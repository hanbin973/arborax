import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from tests.conftest import BeagleJAX

N_TAXA = 100
N_STATES = 4
N_PATTERNS = 2

EDGE_MAP = {0: (3, 0.1), 1: (3, 0.1), 2: (4, 0.1), 3: (4, 0.1)}
OPS = [
    {"dest": 3, "child1": 0, "child2": 1},
    {"dest": 4, "child1": 3, "child2": 2},
]
EDGE_LENGTHS_JAX = jnp.array([0.1, 0.2, 0.3, 0.4], dtype=jnp.float32)


def _sample_Q_pi(rng):
    base = rng.random((N_STATES, N_STATES), dtype=np.float32)
    Q_np = (base + base.T) / 2.0
    for i in range(N_STATES):
        Q_np[i, i] = 0.0
        row_sum = np.sum(Q_np[i])
        if row_sum > 1e-8:
            Q_np[i] /= row_sum
        Q_np[i, i] = -1.0

    pi_np = rng.dirichlet(np.ones(N_STATES)).astype(np.float32)
    return jnp.array(Q_np, dtype=jnp.float32), jnp.array(pi_np, dtype=jnp.float32)


def _build_problem(seed, use_gpu=False):
    rng = np.random.default_rng(seed)
    Q_jax, pi_jax = _sample_Q_pi(rng)
    tip_data = {
        i: np.ones((N_PATTERNS, N_STATES), dtype=np.float32) for i in range(N_TAXA)
    }
    tip_data[0][0] = [1, 0, 0, 0]
    tip_data[1][0] = [1, 0, 0, 0]
    tip_data[2][0] = [1, 0, 0, 0]
    tip_data[0][1] = [1, 0, 0, 0]
    tip_data[1][1] = [0, 1, 0, 0]
    tip_data[2][1] = [0, 0, 1, 0]

    binder = BeagleJAX(
        tip_count=N_TAXA,
        state_count=N_STATES,
        pattern_count=N_PATTERNS,
        tip_data=tip_data,
        edge_map=EDGE_MAP,
        operations=OPS,
        use_gpu=use_gpu,
        dtype=jnp.float32,
    )

    return binder, Q_jax, pi_jax


def _check_gradients(binder, Q_jax, pi_jax):
    def func_to_check(Q_in):
        return jnp.sum(binder.log_likelihood(EDGE_LENGTHS_JAX, Q_in, pi_jax))

    check_grads(
        func_to_check, (Q_jax,), order=1, modes=["rev"], atol=1e-3, rtol=1e-3, eps=1e-3
    )


def test_q_gradient(use_gpu):
    binder, Q_jax, pi_jax = _build_problem(seed=42, use_gpu=use_gpu)
    _check_gradients(binder, Q_jax, pi_jax)


def test_q_gradient_repeated_runs(seed, use_gpu):
    rng = np.random.default_rng(seed + 1337)
    binder, _, _ = _build_problem(rng.integers(0, 1_000_000), use_gpu=use_gpu)
    for _ in range(5):
        Q_jax, pi_jax = _sample_Q_pi(rng)
        _check_gradients(binder, Q_jax, pi_jax)
