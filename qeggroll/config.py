import jax.numpy as jnp

# Data Types & Precision
DTYPE = jnp.int8
MAX = int(jnp.iinfo(DTYPE).max)       # 127 for int8
FIXED_POINT = 4                       # Weights represent true_val * 2^4
FBIT = 4                              # Fixed-point bits for fitness tables
LOGMAX = 7                            # log2(MAX+1) ≈ 7 for int8

# ES Parameter Mapping Types
PARAM = 0
MM_PARAM = 1
EXCLUDED = 3