import pytest
import numpy as np
import jax
import jax.numpy as jnp
from jax.test_util import check_grads

from tests.conftest import BeagleJAX


def test_pi_gradient():
    print("\n=== STARTING PI GRADIENT TEST (jax.test_util) ===")

    # 1. Setup Constants
    N_TAXA = 3
    N_STATES = 4
    N_PATTERNS = 2

    # 2. Model Setup
    Q_np = np.ones((4, 4)) * 0.25
    np.fill_diagonal(Q_np, -0.75)

    # Random Pi
    pi_np = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)

    # 3. Topology Setup
    edge_map = {0: (3, 0.1), 1: (3, 0.1), 2: (4, 0.1), 3: (4, 0.1)}
    ops = [{"dest": 3, "child1": 0, "child2": 1}, {"dest": 4, "child1": 3, "child2": 2}]

    # 4. Data Setup
    dummy_tips = {
        i: np.ones((N_PATTERNS, N_STATES), dtype=np.float32) for i in range(N_TAXA)
    }

    # Pattern 0: All As
    dummy_tips[0][0] = [1, 0, 0, 0]
    dummy_tips[1][0] = [1, 0, 0, 0]
    dummy_tips[2][0] = [1, 0, 0, 0]

    # Pattern 1: Mixed
    dummy_tips[0][1] = [1, 0, 0, 0]
    dummy_tips[1][1] = [0, 1, 0, 0]
    dummy_tips[2][1] = [0, 0, 1, 0]

    # 5. Initialize Binder
    binder = BeagleJAX(
        tip_count=N_TAXA,
        state_count=N_STATES,
        pattern_count=N_PATTERNS,
        tip_data=dummy_tips,
        edge_map=edge_map,
        operations=ops,
        dtype=jnp.float32,
    )

    # Fixed inputs
    edge_lengths_jax = jnp.array([0.1, 0.1, 0.1, 0.1], dtype=jnp.float32)
    Q_jax = jnp.array(Q_np, dtype=jnp.float32)
    pi_jax = jnp.array(pi_np, dtype=jnp.float32)

    # 6. Define Function to Test
    # check_grads will differentiate w.r.t all arguments passed to this function.
    # We close over edge_lengths and Q so they are treated as constants.
    def func_to_check(pi):
        # Return sum of log-likelihoods to get a scalar output for gradient check
        return jnp.sum(binder.log_likelihood(edge_lengths_jax, Q_jax, pi))

    print("Checking gradients via jax.test_util.check_grads...")

    # 7. Run Check
    # order=1: Check first derivative
    # modes=['rev']: Check reverse-mode (VJP), which is what we implemented
    # eps: Finite difference step size (larger is better for float32 roundtrips)
    check_grads(
        func_to_check, (pi_jax,), order=1, modes=["rev"], atol=1e-3, rtol=1e-3, eps=1e-3
    )

    print("[SUCCESS] jax.test_util.check_grads passed.")


if __name__ == "__main__":
    test_pi_gradient()
