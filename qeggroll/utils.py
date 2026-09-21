import jax
import jax.numpy as jnp
from typing import NamedTuple
from .config import MAX, DTYPE

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

def clipped_add(*arrays):
    """Sum int8 tensors through int32, clip back to int8 range."""
    return jnp.clip(
        sum(a.astype(jnp.int32) for a in arrays), -MAX, MAX
    ).astype(DTYPE)