import jax
import jax.numpy as jnp
from .config import PARAM, MM_PARAM, FIXED_POINT, MAX, DTYPE
from .utils import CommonInit, CommonParams, merge_inits, merge_frozen, call_submodule

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
            common_params.params, common_params.es_tree_key, common_params.iterinfo, x)

class Linear(Model):
    @classmethod
    def rand_init(cls, key, in_dim, out_dim, dtype, use_bias=False, **_):
        parts = dict(weight=MM.rand_init(key, in_dim, out_dim, dtype))
        if use_bias:
            parts["bias"] = Parameter.rand_init(
                None, None, None, jnp.zeros(out_dim, dtype=dtype), dtype)
        return merge_inits(**parts)

    @classmethod
    def _forward(cls, common_param, x, **_):
        out = call_submodule(MM, "weight", common_param, x)
        if "bias" in common_param.params:
            out = out + call_submodule(Parameter, "bias", common_param)
        return out

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

class IntMLP(Model):
    """
    n_layer residual blocks: x = clipped_add(Linear(EGG_LN(x)), x)
    followed by a linear output head. No standard activation functions — int8 clipping is the nonlinearity.
    """
    @classmethod
    def rand_init(cls, key, in_dim, out_dim, hidden_dim, n_layer, dtype, **_):
        keys = jax.random.split(key, n_layer + 2)
        blocks = {}
        for i in range(n_layer):
            blocks[f"ln{i}"]    = EGG_LN.rand_init(None,  hidden_dim, dtype)
            blocks[f"linear{i}"] = Linear.rand_init(keys[i], hidden_dim, hidden_dim, dtype)
        merged = merge_inits(
            proj=Linear.rand_init(keys[-2], in_dim, hidden_dim, dtype),
            head=Linear.rand_init(keys[-1], hidden_dim, out_dim, dtype),
            **blocks,
        )
        return merge_frozen(merged, n_layer=n_layer)

    @classmethod
    def _forward(cls, common_params, x):
        n_layer = common_params.frozen_params["n_layer"]
        x = call_submodule(Linear, "proj", common_params, x)
        for i in range(n_layer):
            residual = x
            x = call_submodule(EGG_LN, f"ln{i}", common_params, x)
            x = jnp.clip(x, 0, MAX).astype(DTYPE)
            x = call_submodule(Linear, f"linear{i}", common_params, x)
            # x = clipped_add(x, residual)
        return call_submodule(Linear, "head", common_params, x)