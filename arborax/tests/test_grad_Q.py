import pytest
import numpy as np
import jax
import jax.numpy as jnp
from jax.test_util import check_grads
from arborax.ops.binder import BeagleJAX

def test_q_gradient():
    print("\n=== STARTING Q GRADIENT TEST (Float32) ===")
    
    # 1. Setup Constants
    N_TAXA = 100
    N_STATES = 4
    N_PATTERNS = 2
    
    # 2. Model Setup (Symmetric Q for stability in random tests)
    np.random.seed(42)
    base = np.random.rand(N_STATES, N_STATES).astype(np.float32)
    Q_np = (base + base.T) / 2.0 
    
    for i in range(N_STATES):
        Q_np[i, i] = 0
        row_sum = np.sum(Q_np[i, :])
        if row_sum > 1e-8:
            Q_np[i, :] /= row_sum 
        Q_np[i, i] = -1.0
        
    pi_np = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
    
    # 3. Topology Setup
    edge_map = {0: (3, 0.1), 1: (3, 0.1), 2: (4, 0.1), 3: (4, 0.1)}
    ops = [{'dest': 3, 'child1': 0, 'child2': 1}, {'dest': 4, 'child1': 3, 'child2': 2}]
    
    # 4. Data Setup
    dummy_tips = {i: np.ones((N_PATTERNS, N_STATES), dtype=np.float32) for i in range(N_TAXA)}
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
        dtype=jnp.float32 # Explicit float32
    )
    
    # Fixed inputs (float32)
    edge_lengths_jax = jnp.array([0.1, 0.2, 0.3, 0.4], dtype=jnp.float32)
    Q_jax = jnp.array(Q_np, dtype=jnp.float32)
    pi_jax = jnp.array(pi_np, dtype=jnp.float32)
    
    # 6. Define Function to Test
    def func_to_check(Q_in):
        return jnp.sum(binder.log_likelihood(edge_lengths_jax, Q_in, pi_jax))

    print("Checking gradients via jax.test_util.check_grads...")
    
    # 7. Run Check
    # We use 1e-3 because intermediate residuals (partials, eigenvecs) are 
    # truncated to float32 when passed through the JAX callback boundary.
    check_grads(
        func_to_check, 
        (Q_jax,), 
        order=1, 
        modes=['rev'], 
        atol=1e-3, 
        rtol=1e-3, 
        eps=1e-3
    )
    
    print("[SUCCESS] Gradient of Q matches numerical reference.")

if __name__ == "__main__":
    test_q_gradient()
