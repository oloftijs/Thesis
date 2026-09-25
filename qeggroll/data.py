import jax
import jax.numpy as jnp
import numpy as np
from .config import MAX, FIXED_POINT

@jax.jit
def compute_fitness(logits_int8, targets_int8):
    pred  = jnp.argmax(logits_int8, axis=-1)
    true  = jnp.argmax(targets_int8, axis=-1)
    return jnp.sum(pred == true).astype(jnp.int32)

def make_dataset(n_samples, in_dim, out_dim, rng):
    x_float = rng.standard_normal((n_samples, in_dim)).astype(np.float32)
    W_true  = rng.standard_normal((out_dim, in_dim)).astype(np.float32)
    labels  = (x_float @ W_true.T > 0).astype(np.int32)

    x_int8 = np.clip(
        np.round(x_float * (2 ** FIXED_POINT)), -MAX, MAX
    ).astype(np.int8)

    y_int8 = np.zeros((n_samples, out_dim), dtype=np.int8)
    for i in range(n_samples):
        true_class = np.argmax(labels[i])
        y_int8[i, true_class] = 64

    return jnp.array(x_int8), jnp.array(y_int8)

def make_xor_dataset(n_samples, rng):
    x = rng.standard_normal((n_samples, 2)).astype(np.float32)
    labels = ((x[:, 0] > 0) == (x[:, 1] > 0)).astype(np.int32)
    x_int8 = np.clip(np.round(x * 48), -127, 127).astype(np.int8)
    y_int8 = np.zeros((n_samples, 2), dtype=np.int8)
    y_int8[np.arange(n_samples), labels] = 64
    return jnp.array(x_int8), jnp.array(y_int8)

def make_spiral_dataset(n_samples, rng):
    n = n_samples // 2
    theta0 = rng.uniform(0, 4 * np.pi, n).astype(np.float32)
    r0     = theta0 / (4 * np.pi)
    x0     = np.stack([r0 * np.cos(theta0), r0 * np.sin(theta0)], axis=1)

    theta1 = rng.uniform(0, 4 * np.pi, n).astype(np.float32)
    r1     = theta1 / (4 * np.pi)
    x1     = np.stack([r1 * np.cos(theta1 + np.pi),
                       r1 * np.sin(theta1 + np.pi)], axis=1)

    x0 += rng.normal(0, 0.05, x0.shape).astype(np.float32)
    x1 += rng.normal(0, 0.05, x1.shape).astype(np.float32)

    x      = np.concatenate([x0, x1], axis=0)
    labels = np.array([0]*n + [1]*n, dtype=np.int32)

    idx = rng.permutation(n_samples)
    x, labels = x[idx], labels[idx]

    x_int8 = np.clip(np.round(x * 80), -127, 127).astype(np.int8)
    y_int8 = np.zeros((n_samples, 2), dtype=np.int8)
    y_int8[np.arange(n_samples), labels] = 64

    return jnp.array(x_int8), jnp.array(y_int8)