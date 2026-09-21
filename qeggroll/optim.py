import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from .config import FIXED_POINT, MAX, DTYPE, FBIT

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
    B = lora[:b]
    A = lora[b:] * sign
    return A, B

def _get_nonlora_update_params(BIG_RAND_MATRIX, fnp, iterinfo, param, key):
    idx, sign = _get_common_start_idx(fnp, iterinfo, key)
    updates   = jax.lax.dynamic_slice_in_dim(BIG_RAND_MATRIX, idx, param.size * 2) \
                    .reshape(param.shape + (2,)).astype(jnp.int32)
    return jnp.prod(updates, axis=-1) * sign

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
                    noise_size=2**28, max_update_step=1):

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
        }

        return frozen, state

    @classmethod
    def do_mm(cls, frozen_noiser_params, noiser_params, param, base_key, iterinfo, x):
        base = jnp.dot(x, param.T, preferred_element_type=jnp.int32)
        if iterinfo is not None:
            A, B = _get_lora_update_params(noiser_params["BIG_RAND_MATRIX"], frozen_noiser_params, iterinfo, param, base_key)
            perturb = jnp.dot(x, B, preferred_element_type=jnp.int32) @ A.T.astype(jnp.int32)
            base += perturb >> (FIXED_POINT + noiser_params["sigma_shift"])

        return jnp.clip(
                base // ((2 ** FIXED_POINT) * int(np.sqrt(param.shape[-1]))),
                -MAX, MAX
                ).astype(param.dtype)

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
        pairs = raw_scores.reshape(-1, 2)
        if fnp["fast_fitness"]:
            return jnp.sign(pairs[:, 0] - pairs[:, 1]).astype(DTYPE)
        diff = (pairs[:, 0] - pairs[:, 1]).astype(jnp.float32)
        rms  = jnp.sqrt(jnp.mean(diff ** 2)) + 1e-8
        return jnp.clip((diff / rms * (2 ** FBIT)).astype(jnp.int32), -MAX, MAX).astype(DTYPE)

    @classmethod
    def _do_update(cls, fnp, np_, param, base_key, fitnesses, iterinfos, map_cls):
        fn = [_full_update, _lora_update, _lora_update,
              lambda *a, **k: a[2]][map_cls]
        if len(base_key.shape) == 0:
            return fn(fnp, np_, param, base_key, fitnesses, iterinfos)
        return jax.vmap(fn, in_axes=(None, None, 0, 0, None, None))(
            fnp, np_, param, base_key, fitnesses, iterinfos)

    @classmethod
    def do_updates(cls, fnp, np_, params, base_keys, fitnesses, iterinfos, es_map):
        ii = jax.tree.map(lambda x: x[::2], iterinfos)
        new_params = jax.tree.map(
            lambda p, k, m: cls._do_update(fnp, np_, p, k, fitnesses, ii, m),
            params, base_keys, es_map)
        return np_, new_params