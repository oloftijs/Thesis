import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
import tqdm
from dataclasses import dataclass

from qeggroll.nn import IntMLP
from qeggroll.optim import QEggRoll
from qeggroll.data import make_spiral_dataset, compute_fitness
from qeggroll.utils import simple_es_tree_key
from qeggroll.config import DTYPE

@dataclass
class Args:
    seed:            int   = 0
    dtype:           str   = "int8"
    in_dim:          int   = 2
    out_dim:         int   = 2
    hidden_dim:      int   = 64
    n_layer:         int   = 2
    population_size: int   = 4096
    num_epochs:      int   = 5000
    sigma_shift:     int   = 1
    alpha:           float = 0.5
    noise_reuse:     int   = 1
    rank:            int   = 1
    use_clt:         bool  = False
    fast_fitness:    bool  = False
    noise_seed:      int   = 42
    noise_size_exp:  int   = 28
    update_threshold: int  = 1

def main():
    args = tyro.cli(Args)
    rng = np.random.default_rng(args.seed)

    train_x, train_y = make_spiral_dataset(2048, rng)
    test_x,  test_y  = make_spiral_dataset(512, rng)

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

    num_params = jax.tree.reduce(lambda a, b: a + b, jax.tree.map(lambda x: x.size, params))
    print(f"Parameters: {num_params:,}")

    update_batch_size = max(2, args.population_size // 16)
    frozen_noiser_params, noiser_params = QEggRoll.init_noiser(
        params,
        sigma_shift       = args.sigma_shift,
        update_threshold  = args.update_threshold,
        dtype             = args.dtype,
        noise_seed        = args.noise_seed,
        noise_reuse       = args.noise_reuse,
        rank              = args.rank,
        use_clt           = args.use_clt,
        fast_fitness      = args.fast_fitness,
        update_batch_size = update_batch_size,
        noise_size        = 2**args.noise_size_exp,
    )

    N = args.population_size
    assert N % 2 == 0, "population_size must be even (antithetic pairs)"

    jit_forward = jax.jit(jax.vmap(
        lambda np_, p, ii, x: IntMLP.forward(QEggRoll, frozen_noiser_params, np_, frozen_params, p, es_tree_key, ii, x),
        in_axes=(None, None, 0, 0)
    ))

    jit_eval = jax.jit(jax.vmap(
        lambda np_, p, x: IntMLP.forward(QEggRoll, frozen_noiser_params, np_, frozen_params, p, es_tree_key, None, x),
        in_axes=(None, None, 0)
    ))

    jit_update = jax.jit(
        lambda np_, p, fit, ii: QEggRoll.do_updates(frozen_noiser_params, np_, p, es_tree_key, fit, ii, es_map)
    )

    print("Compiling... ", end="", flush=True)
    dummy_iterinfos = (jnp.zeros(N, dtype=jnp.int32), jnp.arange(N, dtype=jnp.int32))
    dummy_x  = jnp.zeros((N, args.in_dim), dtype=DTYPE)
    _ = jax.block_until_ready(jit_forward(noiser_params, params, dummy_iterinfos, dummy_x))
    _ = jax.block_until_ready(jit_eval(noiser_params, params, jnp.zeros((4, args.in_dim), dtype=DTYPE)))
    dummy_fitnesses = jnp.zeros(N // 2, dtype=DTYPE)
    _ = jax.block_until_ready(jit_update(noiser_params, params, dummy_fitnesses, dummy_iterinfos))
    print("done.")

    for epoch in tqdm.trange(args.num_epochs):
        idx      = rng.integers(0, len(train_x), size=N)
        batch_x  = train_x[idx]
        batch_y  = train_y[idx]

        iterinfo = (jnp.full(N, epoch, dtype=jnp.int32), jnp.arange(N, dtype=jnp.int32))
        noiser_params["sigma_shift"] = 1 if epoch < 500 else 3

        logits = jit_forward(noiser_params, params, iterinfo, batch_x)
        raw_scores = jax.vmap(compute_fitness)(logits, batch_y)
        fitnesses = QEggRoll.convert_fitnesses(frozen_noiser_params, noiser_params, raw_scores)
        noiser_params, params = jit_update(noiser_params, params, fitnesses, iterinfo)

        if epoch % 20 == 0 or epoch == args.num_epochs - 1:
            eval_logits  = jit_eval(noiser_params, params, test_x)
            accuracy     = jnp.mean(jnp.argmax(eval_logits, axis=-1) == jnp.argmax(test_y, axis=-1)).item()
            avg_fitness  = jnp.mean(raw_scores.astype(jnp.float32)).item()
            print(f"  epoch {epoch:4d} | avg_fitness {avg_fitness:8.1f} | test_acc {accuracy:.3f}")

    print("\nDone.")
    for x_vals, true_label, desc in [([48, 48], 1, "(+,+)"), ([-48, 48], 0, "(-,+)"), ([48, -48], 0, "(+,-)"), ([-48, -48], 1, "(-,-)")]:
        inp = jnp.array([x_vals], dtype=jnp.int8)
        pred = int(jnp.argmax(jit_eval(noiser_params, params, inp)[0]))
        print(f"  {'✓' if pred == true_label else '✗'} {desc} → class {true_label} | pred={pred}")

if __name__ == "__main__":
    main()