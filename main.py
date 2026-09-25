"""
int_mlp_train.py
================
Pure-integer MLP policy trained with QEggRoll (EGGROLL for int8 parameters).

Architecture
------------
  Input  →  [Linear → clipped_add(residual) → EGG_LN] × n_layer  →  Linear head  →  logits

Nonlinearities come entirely from int8 overflow/clipping; sigmoid/tanh are
identity functions exactly as in the EGG language model.

Usage
-----
  python int_mlp_train.py               # defaults
  python int_mlp_train.py --help        # all flags via tyro
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
import tqdm
from dataclasses import dataclass
from typing import NamedTuple, Optional
from functools import partial


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Args:
    seed:            int   = 0
    dtype:           str   = "int8"
    in_dim:          int   = 2       # e.g. 4 for CartPole obs
    out_dim:         int   = 2       # e.g. 2 for CartPole actions
    hidden_dim:      int   = 64
    n_layer:         int   = 2
    population_size: int   = 4096
    num_epochs:      int   = 5000
    sigma_shift:     int   = 1      # perturbation scale: >> (FIXED_POINT + sigma_shift)
    alpha:           float = 0.5 # 0.05    # significance level for update threshold
    noise_reuse:     int   = 1
    rank:            int   = 1
    use_clt:         bool  = False
    fast_fitness:    bool  = False
    noise_seed:      int   = 42
    noise_size_exp:  int = 28      # Original paper user 2^30 for BIG_RAND_MATRIX
    update_threshold: int = 1

# gets args from command line in one line :)
if __name__ == "__main__":
    args = tyro.cli(Args)
else:
    args = Args()  # use defaults when imported as a module

DTYPE      = jnp.dtype(args.dtype)
MAX        = int(jnp.iinfo(DTYPE).max)          # 127 for int8
FIXED_POINT = 4                                  # weights represent true_val * 2^4
FBIT       = 4                                   # fixed-point bits for fitness tables
LOGMAX     = 7                                   # log2(MAX+1) ≈ 7 for int8

# ─────────────────────────────────────────────────────────────────────────────
# Shared data structures  (identical to run.py / notebook)
# ─────────────────────────────────────────────────────────────────────────────

PARAM     = 0
MM_PARAM  = 1
EXCLUDED  = 3

class CommonInit(NamedTuple):
    frozen_params: any
    params:        any
    scan_map:      any
    es_map:        any

class CommonParams(NamedTuple):
    noiser:              any
    frozen_noiser_params: any
    noiser_params:       any
    frozen_params:       any
    params:              any
    es_tree_key:         any
    iterinfo:            any


# ─────────────────────────────────────────────────────────────────────────────
# Tree / key utilities  (from eggroll.ipynb notebook)
# ─────────────────────────────────────────────────────────────────────────────

def recursive_scan_split(param, base_key, scan_tuple):
    if len(scan_tuple) == 0:
        return base_key
    split_keys = jax.random.split(base_key, param.shape[scan_tuple[0]])
    return jax.vmap(recursive_scan_split, in_axes=(None, 0, None))(param, split_keys, scan_tuple[1:])

def simple_es_tree_key(params, base_key, scan_map):
    vals, treedef = jax.tree.flatten(params)
    all_keys      = jax.random.split(base_key, len(vals))
    partial_key_tree = jax.tree.unflatten(treedef, all_keys)
    return jax.tree.map(recursive_scan_split, params, partial_key_tree, scan_map)

def merge_inits(**kwargs):
    params = {}; frozen_params = {}; scan_map = {}; es_map = {}
    for k in kwargs:
        params[k]   = kwargs[k].params
        scan_map[k] = kwargs[k].scan_map
        es_map[k]   = kwargs[k].es_map
        if kwargs[k].frozen_params is not None:
            frozen_params[k] = kwargs[k].frozen_params
    return CommonInit(frozen_params or None, params, scan_map, es_map)

def merge_frozen(common, **kwargs):
    fp = (common.frozen_params or {}) | kwargs
    return common._replace(frozen_params=fp)

def call_submodule(cls, name, common_params, *args, **kwargs):
    sub_common_params = common_params._replace(
        frozen_params = common_params.frozen_params[name] if common_params.frozen_params and name in common_params.frozen_params else None,
        params        = common_params.params[name],
        es_tree_key   = common_params.es_tree_key[name],
    )
    return cls._forward(sub_common_params, *args, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Base Model class
# ─────────────────────────────────────────────────────────────────────────────

class Model:
    @classmethod
    def rand_init(cls, key, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def forward(cls, noiser, frozen_noiser_params, noiser_params,
                frozen_params, params, es_tree_key, iterinfo, *args, **kwargs):
        common_params = CommonParams(noiser, frozen_noiser_params, noiser_params,
                                     frozen_params, params, es_tree_key, iterinfo)
        return cls._forward(common_params, *args, **kwargs)

    @classmethod
    def _forward(cls, cp, *args, **kwargs):
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
# Atomic model primitives
# ─────────────────────────────────────────────────────────────────────────────

class Parameter(Model):
    @classmethod
    def rand_init(cls, key, shape, scale, raw_value, dtype, **_):
        params = raw_value.astype(dtype) if raw_value is not None \
                 else (jax.random.normal(key, shape) * scale).astype(dtype)
        return CommonInit(None, params, (), PARAM)

    @classmethod
    def _forward(cls, common_params, **_):
        return common_params.noiser.get_noisy_standard(
            common_params.frozen_noiser_params, common_params.noiser_params,
            common_params.params, common_params.es_tree_key, common_params.iterinfo)

class MM(Model):
    """Matrix-multiply weight, initialised in fixed-point int8."""
    @classmethod
    def rand_init(cls, key, in_dim, out_dim, dtype, **_):
        params = jnp.round(
            jax.random.normal(key, (out_dim, in_dim)) * (2 ** FIXED_POINT)
        ).astype(dtype)
        return CommonInit(None, params, (), MM_PARAM)

    @classmethod
    def _forward(cls, common_params, x, **_):
        return common_params.noiser.do_mm(
            common_params.frozen_noiser_params, common_params.noiser_params,
            common_params.params, common_params.frozen_params,
            common_params.es_tree_key, common_params.iterinfo, x)
class BlockMM(Model):
    @classmethod
    def rand_init(cls, key, in_dim, out_dim, dtype, block_size=16, **_):
        if in_dim % block_size != 0:
            block_size = in_dim

        num_blocks = in_dim // block_size

        weight = jnp.round(
            jax.random.normal(key, (out_dim, in_dim)) * (2 ** FIXED_POINT)
        ).astype(dtype)

        block_mults = jnp.ones((out_dim, num_blocks), dtype=jnp.int32)
        block_shifts = jnp.ones((out_dim, num_blocks), dtype=jnp.int32) * FIXED_POINT

        merged = merge_inits(
            weight=CommonInit(None, weight, (), MM_PARAM),
            block_mults=CommonInit(None, block_mults, (), PARAM),
            block_shifts=CommonInit(None, block_shifts, (), PARAM)
        )


        return merge_frozen(merged, block_size=block_size)
    @classmethod
    def _forward(cls, common_params, x, **_):
        return common_params.noiser.do_mm(
            common_params.frozen_noiser_params,
            common_params.noiser_params,
            common_params.params,
            common_params.frozen_params,
            common_params.es_tree_key,
            common_params.iterinfo,
            x
        )
# ─────────────────────────────────────────────────────────────────────────────
# Composite model: Linear
# ─────────────────────────────────────────────────────────────────────────────

class Linear(Model):
    @classmethod
    def rand_init(cls, key, in_dim, out_dim, dtype, use_bias=False, **_):
        parts = dict(weight=BlockMM.rand_init(key, in_dim, out_dim, dtype))
        if use_bias:
            parts["bias"] = Parameter.rand_init(
                None, None, None, jnp.zeros(out_dim, dtype=dtype), dtype)
        return merge_inits(**parts)

    @classmethod
    def _forward(cls, common_param, x, **_):
        out = call_submodule(BlockMM, "weight", common_param, x)
        if "bias" in common_param.params:
            out = out + call_submodule(Parameter, "bias", common_param)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Integer layer norm  (lookup-table division, no floats)
# ─────────────────────────────────────────────────────────────────────────────

class EGG_LN(Model):
    @classmethod
    def rand_init(cls, key, hidden_dim, dtype, **_):
        return merge_inits(
            weight=Parameter.rand_init(
                None, None, None,
                jnp.ones(hidden_dim, dtype=dtype) * (2 ** FIXED_POINT),
                dtype))

    @classmethod
    def _forward(cls, common_params, x):
        weight   = call_submodule(Parameter, "weight", common_params).astype(jnp.int32)
        abs_sum  = jnp.clip(
            jnp.dot(jnp.abs(x), jnp.ones_like(x), preferred_element_type=jnp.int32),
            min=1) // x.size
        numerator = (x * weight).astype(jnp.int16).view(jnp.uint16)
        return common_params.noiser_params["DIVISION"][abs_sum][numerator]


# ─────────────────────────────────────────────────────────────────────────────
# Integer utility
# ─────────────────────────────────────────────────────────────────────────────

def clipped_add(*arrays):
    """Sum int8 tensors through int32, clip back to int8 range."""
    return jnp.clip(
        sum(a.astype(jnp.int32) for a in arrays), -MAX, MAX
    ).astype(DTYPE)


# ─────────────────────────────────────────────────────────────────────────────
# Pure-integer MLP  (no activation functions — nonlinearity from int overflow)
# ─────────────────────────────────────────────────────────────────────────────

class IntMLP(Model):
    """
    n_layer residual blocks:  x = clipped_add(Linear(EGG_LN(x)), x)
    followed by a linear output head.

    No sigmoid / tanh / relu — int8 clipping is the nonlinearity.
    """
    @classmethod
    def rand_init(cls, key, in_dim, out_dim, hidden_dim, n_layer, dtype, **_):
        keys = jax.random.split(key, n_layer + 2)
        blocks = {}
        for i in range(n_layer):
            blocks[f"ln{i}"]    = EGG_LN.rand_init(None,  hidden_dim, dtype)
            blocks[f"linear{i}"] = Linear.rand_init(keys[i], hidden_dim, hidden_dim, dtype)
        # Input projection (in_dim → hidden_dim) and output head
        merged = merge_inits(
            proj=Linear.rand_init(keys[-2], in_dim, hidden_dim, dtype),
            head=Linear.rand_init(keys[-1], hidden_dim, out_dim, dtype),
            **blocks,
        )
        return merge_frozen(merged, n_layer=n_layer)

    @classmethod
    def _forward(cls, common_params, x):
        n_layer = common_params.frozen_params["n_layer"]
        # Input projection
        x = call_submodule(Linear, "proj", common_params, x)
        # Residual blocks
        for i in range(n_layer):
            residual = x
            x = call_submodule(EGG_LN, f"ln{i}", common_params, x)
            x = jnp.clip(x, 0, MAX).astype(DTYPE)   # integer ReLU (pqn: relu after layer norm)
            x = call_submodule(Linear, f"linear{i}", common_params, x)
            #x = clipped_add(x, residual)
        # Output head (logits)
        return call_submodule(Linear, "head", common_params, x)


# ─────────────────────────────────────────────────────────────────────────────
# QEggRoll noiser  (integer EGGROLL from run.py, stripped to MLP needs)
# ─────────────────────────────────────────────────────────────────────────────

def _fold_in(base_key_int32, new_int32):
    x = new_int32
    x = ((x >> 16) ^ x) * 0x45d9f3b
    return base_key_int32[0] ^ x

def _get_common_start_idx(frozen_noiser_params, iterinfo, key):
    epoch, thread_id = iterinfo
    true_epoch      = 0 if frozen_noiser_params["noise_reuse"] == 0 \
                      else epoch // frozen_noiser_params["noise_reuse"]
    true_thread_idx = thread_id >> 1
    actual_key      = _fold_in(
        jax.random.key_data(jax.random.fold_in(key, true_epoch)),
        true_thread_idx)
    start_idx = actual_key & (frozen_noiser_params["noise_size"] - 1)
    sign      = jnp.where(thread_id % 2 == 0, 1, -1)
    return start_idx, sign

def _get_lora_update_params(BIG_RAND_MATRIX, frozen_noiser_params, iterinfo, param, key):
    a, b = param.shape
    r = frozen_noiser_params["rank"]
    idx, sign = _get_common_start_idx(frozen_noiser_params, iterinfo, key)
    lora = jax.lax.dynamic_slice_in_dim(BIG_RAND_MATRIX, idx, (a + b) * r).reshape((a + b, r))
    B = lora[:b]        # (b, r)
    A = lora[b:] * sign # (a, r)
    return A, B

def _get_nonlora_update_params(BIG_RAND_MATRIX, fnp, iterinfo, param, key):
    idx, sign = _get_common_start_idx(fnp, iterinfo, key)
    updates   = jax.lax.dynamic_slice_in_dim(BIG_RAND_MATRIX, idx, param.size * 2) \
                    .reshape(param.shape + (2,)).astype(jnp.int32)
    return jnp.prod(updates, axis=-1) * sign

# ── per-parameter update helpers ──

def _common_update(frozen_noiser_params, noiser_params, param, Z, pop_size):
    threshold = noiser_params["update_threshold"] * int(np.sqrt(pop_size))
    if frozen_noiser_params["use_clt"]:
        threshold *= 4 ** FIXED_POINT
    param_int32  = param.astype(jnp.int32)
    max_step     = frozen_noiser_params.get("max_update_step", 1)
    below_thresh = jnp.abs(Z) < threshold
    if max_step <= 1:
        step = jnp.where(
            below_thresh,
            jnp.zeros_like(Z),
            jnp.where(Z > 0, jnp.ones_like(Z), -jnp.ones_like(Z))
        )
    else:
        # Allow ±max_step when signal is strong; same noise gate as pm1.
        capped = jnp.clip(
            jnp.round(Z.astype(jnp.float32) / threshold).astype(jnp.int32),
            -max_step, max_step,
        )
        step = jnp.where(below_thresh, jnp.zeros_like(Z), capped)
    return jnp.clip(param_int32 + step, -MAX, MAX).astype(param.dtype)

def _lora_update(frozen_noiser_params, noiser_params, param, key, scores, iterinfo):
    update_batch = frozen_noiser_params["update_batch_size"]
    split_ii     = jax.tree.map(lambda x: x.reshape(-1, update_batch), iterinfo)
    split_sc     = scores.reshape(-1, update_batch)

    def scan_fn(Z, inputs):
        ii, sc = inputs
        A, B   = jax.vmap(
            partial(_get_lora_update_params, noiser_params["BIG_RAND_MATRIX"], frozen_noiser_params),
            in_axes=(0, None, None))(ii, param, key)
        sc_b   = sc.reshape(sc.shape + (1, 1))
        if frozen_noiser_params["use_clt"]:
            A = sc_b * A
        else:
            A = sc_b * jnp.sign(A);  B = jnp.sign(B)
        return Z + jnp.einsum("nir,njr->ij", A, B, preferred_element_type=jnp.int32), 0

    Z, _ = jax.lax.scan(scan_fn,
                        jnp.zeros_like(param, dtype=jnp.int32),
                        (split_ii, split_sc))
    return _common_update(frozen_noiser_params, noiser_params, param, Z, scores.size)

def _full_update(fnp, noiser_params, param, key, scores, iterinfo):
    update_batch = fnp["update_batch_size"]
    split_ii     = jax.tree.map(lambda x: x.reshape(-1, update_batch), iterinfo)
    split_sc     = scores.reshape(-1, update_batch)

    def scan_fn(Z, inputs):
        ii, sc = inputs
        upd    = jax.vmap(
            partial(_get_nonlora_update_params, noiser_params["BIG_RAND_MATRIX"], fnp),
            in_axes=(0, None, None))(ii, param, key)
        sc_b   = sc.reshape(sc.shape + (1,) * len(param.shape))
        A      = sc_b * (upd.astype(jnp.int32) if fnp["use_clt"] else jnp.sign(upd).astype(jnp.int32))
        return Z + jnp.sum(A, axis=0), 0

    Z, _ = jax.lax.scan(scan_fn,
                        jnp.zeros_like(param, dtype=jnp.int32),
                        (split_ii, split_sc))
    return _common_update(fnp, noiser_params, param, Z, scores.size)


class QEggRoll:
    @classmethod
    def init_noiser(cls, params, sigma_shift, update_threshold, *,
                    dtype="int8", noise_seed=0, noise_reuse=1, rank=1,
                    use_clt=True, fast_fitness=True, update_batch_size=64,
                    noise_size=2**args.noise_size_exp, max_update_step=1):
        # Precompute lookup table for integer layer norm division:
        # DIVISION[abs_mean][numerator_uint16] -> int8
        division = jnp.clip(
            jnp.arange(2**16).astype(jnp.int16)[None, :]
            // jnp.maximum(jnp.arange(2**8).astype(jnp.uint8)[:, None], 1),
            -MAX, MAX
        ).astype(jnp.int8)

        frozen = {
            "noise_reuse":      noise_reuse,
            "rank":             rank,
            "use_clt":          use_clt,
            "fast_fitness":     fast_fitness,
            "update_batch_size": update_batch_size,
            "max_update_step":  max_update_step,
            "noise_size":       noise_size,
        }

        state = {
            "BIG_RAND_MATRIX": (
                jax.random.normal(jax.random.key(noise_seed), noise_size)
                * (2 ** FIXED_POINT)
            ).astype(dtype),
            "sigma_shift":      sigma_shift,
            "update_threshold": update_threshold,
            "DIVISION":         division,
            # "update_batch_size": update_batch_size, # moved into frozen
        }

        return frozen, state

    @classmethod
    def do_mm(cls, frozen_noiser_params, noiser_params, params, frozen_params, base_key, iterinfo, x):
        weight = params["weight"]

        block_mults = cls.get_noisy_standard(
            frozen_noiser_params, noiser_params, params["block_mults"], base_key["block_mults"], iterinfo
        )
        block_shifts = cls.get_noisy_standard(
            frozen_noiser_params, noiser_params, params["block_shifts"], base_key["block_shifts"], iterinfo
        )

        block_size = frozen_params["block_size"]
        out_dim, in_dim = weight.shape
        num_blocks = in_dim // block_size

        x_blocks = x.reshape(num_blocks, block_size).astype(jnp.int32)
        w_blocks = weight.reshape(out_dim, num_blocks, block_size).astype(jnp.int32)

        base_activation = jnp.zeros(out_dim, dtype=jnp.int32)


        for b in range(num_blocks):

            p_b = jnp.dot(x_blocks[b], w_blocks[:, b, :].T, preferred_element_type=jnp.int32)

            # Scale and accumulate immediately to prevent massive intermediate tensors
            scaled_b = (p_b * block_mults[:, b]) >> block_shifts[:, b]
            base_activation += scaled_b

        perturb_activation = 0
        if iterinfo is not None:
            weight_key = base_key["weight"]

            A, B = _get_lora_update_params(
                noiser_params["BIG_RAND_MATRIX"], frozen_noiser_params, iterinfo, weight, weight_key
            )
            raw_perturb = jnp.dot(x, B, preferred_element_type=jnp.int32) @ A.T.astype(jnp.int32)

            fan_in_shift = int(np.log2(np.sqrt(in_dim)))
            global_perturb_shift = FIXED_POINT + noiser_params["sigma_shift"] + fan_in_shift

            perturb_activation = raw_perturb >> global_perturb_shift

        final_activation = base_activation + perturb_activation

        return jnp.clip(final_activation, -MAX, MAX).astype(weight.dtype)
    @classmethod
    def get_noisy_standard(cls, fnp, np_, param, base_key, iterinfo):
        if iterinfo is None:
            return param
        base    = param.astype(jnp.int32)
        perturb = _get_nonlora_update_params(
            np_["BIG_RAND_MATRIX"], fnp, iterinfo, param, base_key)
        return jnp.clip(
            base + (perturb >> (FIXED_POINT + np_["sigma_shift"])),
            -MAX, MAX
        ).astype(param.dtype)

    @classmethod
    def convert_fitnesses(cls, fnp, np_, raw_scores):
        """
        Antithetic pairs: even index is +perturbation, odd is −perturbation.
        fast_fitness=True  → sign(diff), i.e. ±1 int8.
        fast_fitness=False → normalised fixed-point diff.
        """
        pairs = raw_scores.reshape(-1, 2)
        if fnp["fast_fitness"]:
            return jnp.sign(pairs[:, 0] - pairs[:, 1]).astype(DTYPE)
        diff = (pairs[:, 0] - pairs[:, 1]).astype(jnp.float32)
        rms  = jnp.sqrt(jnp.mean(diff ** 2)) + 1e-8
        return jnp.clip((diff / rms * (2 ** FBIT)).astype(jnp.int32), -MAX, MAX).astype(DTYPE)

    @classmethod
    def _do_update(cls, fnp, np_, param, base_key, fitnesses, iterinfos, map_cls):
        fn = [_full_update, _lora_update, _lora_update,
              lambda *a, **k: a[2]][map_cls]   # EXCLUDED → no-op
        if len(base_key.shape) == 0:
            return fn(fnp, np_, param, base_key, fitnesses, iterinfos)
        return jax.vmap(fn, in_axes=(None, None, 0, 0, None, None))(
            fnp, np_, param, base_key, fitnesses, iterinfos)

    @classmethod
    def do_updates(cls, fnp, np_, params, base_keys, fitnesses, iterinfos, es_map):
        # Only use the even-index iterinfos (one per antithetic pair)
        ii = jax.tree.map(lambda x: x[::2], iterinfos)
        new_params = jax.tree.map(
            lambda p, k, m: cls._do_update(fnp, np_, p, k, fitnesses, ii, m),
            params, base_keys, es_map)
        return np_, new_params


# ─────────────────────────────────────────────────────────────────────────────
# Fitness function  (negative MSE in fixed-point integers)
# ─────────────────────────────────────────────────────────────────────────────

@jax.jit
def compute_fitness(logits_int8, targets_int8):
    """
    Fitness = number of correct predictions (higher is better).
    Uses argmax on logits vs argmax on targets.
    Returns int32 scalar.
    """
    pred  = jnp.argmax(logits_int8, axis=-1)
    true  = jnp.argmax(targets_int8, axis=-1)
    return jnp.sum(pred == true).astype(jnp.int32)


# ─────────────────────────────────────────────────────────────────────────────
# Toy dataset: learn the sign function  y = sign(x)  in int8 fixed-point
# ─────────────────────────────────────────────────────────────────────────────

def make_dataset(n_samples, in_dim, out_dim, rng):
    x_float = rng.standard_normal((n_samples, in_dim)).astype(np.float32)
    W_true  = rng.standard_normal((out_dim, in_dim)).astype(np.float32)
    labels  = (x_float @ W_true.T > 0).astype(np.int32)  # (n, out_dim) binary

    # Encode x in fixed-point int8
    x_int8 = np.clip(
        np.round(x_float * (2 ** FIXED_POINT)), -MAX, MAX
    ).astype(np.int8)

    # Encode y as one-hot int8: +64 for true class, 0 elsewhere
    # (arbitrary scale, argmax just needs the true class to be largest)
    y_int8 = np.zeros((n_samples, out_dim), dtype=np.int8)
    for i in range(n_samples):
        true_class = np.argmax(labels[i])
        y_int8[i, true_class] = 64

    return jnp.array(x_int8), jnp.array(y_int8)


def make_xor_dataset(n_samples, rng):
    # 2D XOR: label is 1 if x[0] and x[1] have the same sign
    x = rng.standard_normal((n_samples, 2)).astype(np.float32)
    labels = ((x[:, 0] > 0) == (x[:, 1] > 0)).astype(np.int32)
    x_int8 = np.clip(np.round(x * (2 ** FIXED_POINT)), -MAX, MAX).astype(np.int8)
    y_int8 = np.zeros((n_samples, 2), dtype=np.int8)
    y_int8[np.arange(n_samples), labels] = 64

    return jnp.array(x_int8), jnp.array(y_int8)


def make_xor_dataset(n_samples, rng):
    x = rng.standard_normal((n_samples, 2)).astype(np.float32)
    labels = ((x[:, 0] > 0) == (x[:, 1] > 0)).astype(np.int32)

    # Scale by MAX/2 instead of FIXED_POINT to use more of the int8 range
    x_int8 = np.clip(np.round(x * 48), -127, 127).astype(np.int8)
    y_int8 = np.zeros((n_samples, 2), dtype=np.int8)
    y_int8[np.arange(n_samples), labels] = 64
    return jnp.array(x_int8), jnp.array(y_int8)

def make_spiral_dataset(n_samples, rng):
    n = n_samples // 2

    # Class 0: first spiral arm
    theta0 = rng.uniform(0, 4 * np.pi, n).astype(np.float32)
    r0     = theta0 / (4 * np.pi)
    x0     = np.stack([r0 * np.cos(theta0), r0 * np.sin(theta0)], axis=1)

    # Class 1: second spiral arm (offset by pi)
    theta1 = rng.uniform(0, 4 * np.pi, n).astype(np.float32)
    r1     = theta1 / (4 * np.pi)
    x1     = np.stack([r1 * np.cos(theta1 + np.pi),
                       r1 * np.sin(theta1 + np.pi)], axis=1)

    # Add noise
    x0 += rng.normal(0, 0.05, x0.shape).astype(np.float32)
    x1 += rng.normal(0, 0.05, x1.shape).astype(np.float32)

    x      = np.concatenate([x0, x1], axis=0)
    labels = np.array([0]*n + [1]*n, dtype=np.int32)

    # Shuffle
    idx = rng.permutation(n_samples)
    x, labels = x[idx], labels[idx]

    # Encode x: spirals live in [-1,1], scale to use most of int8 range
    x_int8 = np.clip(np.round(x * 80), -127, 127).astype(np.int8)

    y_int8 = np.zeros((n_samples, 2), dtype=np.int8)
    y_int8[np.arange(n_samples), labels] = 64

    return jnp.array(x_int8), jnp.array(y_int8)

# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def main():
    rng = np.random.default_rng(args.seed)

    # ── Dataset ──────────────────────────────────────────────────────────────
    #train_x, train_y = make_dataset(2048, args.in_dim, args.out_dim, rng)
    #test_x,  test_y  = make_dataset(512,  args.in_dim, args.out_dim, rng)

    train_x, train_y = make_spiral_dataset(2048, rng)
    test_x,  test_y  = make_spiral_dataset(512, rng)

    # ── Model init ───────────────────────────────────────────────────────────
    key = jax.random.key(args.seed)
    model_key, es_key = jax.random.split(key)

    frozen_params, params, scan_map, es_map = IntMLP.rand_init(
        model_key,
        in_dim=args.in_dim,
        out_dim=args.out_dim,
        hidden_dim=args.hidden_dim,
        n_layer=args.n_layer,
        dtype=args.dtype,
    )
    es_tree_key = simple_es_tree_key(params, es_key, scan_map)

    num_params = jax.tree.reduce(
        lambda a, b: a + b, jax.tree.map(lambda x: x.size, params))
    print(f"Parameters: {num_params:,}")

    # ── Noiser init ───────────────────────────────────────────────────────────
    # update_threshold: z-score threshold for accepting a nudge (scipy-free version)
    # Using normal quantile approx: threshold ≈ sqrt(2)*erfinv(1-alpha)
    '''
    threshold = int(round(
        float(jnp.array(1.96 * (2 ** FBIT), dtype=jnp.float32))  # ~95% CI
    ))
    '''
    threshold = args.update_threshold
    update_batch_size = max(2, args.population_size // 16)
    frozen_noiser_params, noiser_params = QEggRoll.init_noiser(
        params,
        sigma_shift      = args.sigma_shift,
        update_threshold = threshold,
        dtype            = args.dtype,
        noise_seed       = args.noise_seed,
        noise_reuse      = args.noise_reuse,
        rank             = args.rank,
        use_clt          = args.use_clt,
        fast_fitness     = args.fast_fitness,
        update_batch_size = update_batch_size,
    )

    N = args.population_size
    assert N % 2 == 0, "population_size must be even (antithetic pairs)"

    # ── JIT-compiled kernels ──────────────────────────────────────────────────
    # Training forward: vmap over N perturbations
    jit_forward = jax.jit(jax.vmap(
        lambda noiser_params, params, iterinfo, x:
            IntMLP.forward(QEggRoll, frozen_noiser_params, noiser_params,
                           frozen_params, params, es_tree_key, iterinfo, x),
        in_axes=(None, None, 0, 0)
    ))

    # Eval forward: no perturbatipon (iterinfo=None), vmap over batch
    jit_eval = jax.jit(jax.vmap(
        lambda noiser_params, params, x:
            IntMLP.forward(QEggRoll, frozen_noiser_params, noiser_params,
                           frozen_params, params, es_tree_key, None, x),
        in_axes=(None, None, 0)
    ))

    jit_update = jax.jit(
        lambda noiser_params, params, fitnesses, iterinfos:
            QEggRoll.do_updates(frozen_noiser_params, noiser_params, params,
                                es_tree_key, fitnesses, iterinfos, es_map)
    )

    # ── Warmup / compile ─────────────────────────────────────────────────────
    print("Compiling... ", end="", flush=True)
    dummy_iterinfos = (jnp.zeros(N, dtype=jnp.int32), jnp.arange(N, dtype=jnp.int32))
    dummy_x  = jnp.zeros((N, args.in_dim), dtype=DTYPE)
    _ = jax.block_until_ready(jit_forward(noiser_params, params, dummy_iterinfos, dummy_x))
    _ = jax.block_until_ready(jit_eval(noiser_params, params,
                                       jnp.zeros((4, args.in_dim), dtype=DTYPE)))
    dummy_fitnesses = jnp.zeros(N // 2, dtype=DTYPE)
    _ = jax.block_until_ready(jit_update(noiser_params, params, dummy_fitnesses, dummy_iterinfos))
    print("done.")

    test_input = jnp.array([[16, 16]], dtype=jnp.int8)   # x = [1.0, 1.0]
    test_out = jit_eval(noiser_params, params, test_input)
    print(f"Test output for [1,1]: {test_out}")

    test_input2 = jnp.array([[-16, 16]], dtype=jnp.int8)  # x = [-1.0, 1.0]
    test_out2 = jit_eval(noiser_params, params, test_input2)
    print(f"Test output for [-1,1]: {test_out2}")

    # ── Epoch loop ────────────────────────────────────────────────────────────
    for epoch in tqdm.trange(args.num_epochs): # tqdm.trange makes the loading bar on terminal
        # Sample a batch; repeat it N times (one per population member)
        idx      = rng.integers(0, len(train_x), size=N)
        batch_x  = train_x[idx]          # (N, in_dim) int8
        batch_y  = train_y[idx]          # (N, out_dim) int8

        iterinfo = (jnp.full(N, epoch, dtype=jnp.int32), jnp.arange(N, dtype=jnp.int32))

        # TODO: Remove later
        if epoch < 500:
            noiser_params["sigma_shift"] = 1
        else:
            noiser_params["sigma_shift"] = 3

        # Forward pass with perturbations
        logits = jit_forward(noiser_params, params, iterinfo, batch_x)  # (N, out_dim)

        # Fitness: negative MSE per population member
        raw_scores = jax.vmap(compute_fitness)(logits, batch_y)          # (N,)

        # Convert to antithetic ±1 fitnesses
        fitnesses = QEggRoll.convert_fitnesses(
            frozen_noiser_params, noiser_params, raw_scores)             # (N//2,)

        # Parameter update
        noiser_params, params = jit_update(noiser_params, params, fitnesses, iterinfo)

        # ── Logging ──────────────────────────────────────────────────────────
        if epoch % 20 == 0 or epoch == args.num_epochs - 1:
            eval_logits  = jit_eval(noiser_params, params, test_x)       # (n_test, out_dim)
            pred_class   = jnp.argmax(eval_logits, axis=-1)
            true_class   = jnp.argmax(test_y, axis=-1)
            accuracy     = jnp.mean(pred_class == true_class).item()
            avg_fitness  = jnp.mean(raw_scores.astype(jnp.float32)).item()
            print(f"  epoch {epoch:4d} | avg_fitness {avg_fitness:8.1f} "
                  f"| test_acc {accuracy:.3f}")

            # Extra debbuging (TODO: Delete this.)
            '''
            sorted_idx = jnp.argsort(raw_scores)[::-1]  # best to worst
            print(f"    top-5 scores: {raw_scores[sorted_idx[:5]]}")
            print(f"    bottom-5 scores: {raw_scores[sorted_idx[-5:]]}")

            print(f"    logits sample (first 5): {eval_logits[:5]}")
            print(f"    pred[:5]: {pred_class[:5]}  true[:5]: {true_class[:5]}")
            print(f"    raw_scores min/max/mean: "
                f"{raw_scores.min():.1f} / {raw_scores.max():.1f} / "
                f"{raw_scores.mean():.1f}")
            print(f"    fitnesses unique values: {jnp.unique(fitnesses)}")
            print(f"    params sample (head weight first row): "
                f"{params['head']['weight'][0, :4]}")

            if 'prev_param' in dir():
                n_changed = sum(
                        int(jnp.sum(p1 != p2))
                        for p1, p2 in zip(
                            jax.tree.leaves(prev_params),
                            jax.tree.leaves(params)
                        )
                    )
                print(f"    params changed this epoch: {n_changed}")

            prev_params = jax.tree.map(lambda x: x.copy(), params)
            '''

    print("\nDone.")

    quadrants = [
        ([ 48,  48], 1, "(+,+) → class 1"),
        ([-48,  48], 0, "(-,+) → class 0"),
        ([ 48, -48], 0, "(+,-) → class 0"),
        ([-48, -48], 1, "(-,-) → class 1"),
        ]
    for x_vals, true_label, desc in quadrants:
        inp = jnp.array([x_vals], dtype=jnp.int8)
        out = jit_eval(noiser_params, params, inp)
        pred = int(jnp.argmax(out[0]))
        correct = "✓" if pred == true_label else "✗"
        print(f"  {correct} {desc} | logits {out[0]} | pred={pred}")


if __name__ == "__main__":

    main()