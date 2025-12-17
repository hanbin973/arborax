import pytest
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
from jax.scipy.linalg import expm

from arborax.pure import fast_tree_likelihood_ops_safe, fast_tree_likelihood_ops_callable

# Re-define safe_log for the test module context
MIN_LOG_VAL = -1e18
def safe_log(x):
    return jnp.where(x > 0, jnp.log(x), MIN_LOG_VAL)

# --- 1. Define an Equinox Module for the Transition Function ---
class TimeHomogeneousTransition(eqx.Module):
    rate_matrix: jax.Array

    def __call__(self, t):
        # f(t) -> log(exp(Q * t))
        P = expm(self.rate_matrix * t)
        return safe_log(P)

def test_custom_ops_vs_safe_ops():
    """
    Verifies that the custom ops function (accepting a callable) matches 
    the standard safe ops function (accepting a fixed Q matrix).
    """
    print("\n--- Comparing Custom Callable vs Standard Safe Implementation ---")

    # 1. Data Setup
    seed = 555
    key = jax.random.PRNGKey(seed)
    
    num_leaves = 5
    num_states = 4
    num_sites = 3
    num_nodes = 2 * num_leaves - 1

    # Random Mock Data
    # Operations (Linear tree for simplicity: 0->3, 1->3, 2->4...)
    # We'll just use a dummy valid ops array for a 3-leaf tree to be safe, 
    # or just random valid indices if we trust the math doesn't check topology validity.
    # Let's use a explicit small topology ((0,1), 2) -> Root is 4. Internal is 3.
    # Nodes: 0,1,2 (leaves), 3,4.
    ops = jnp.array([
        [3, 0, 1],
        [4, 3, 2]
    ], dtype=jnp.int32)
    
    # Branch lengths aligned to nodes 0..4
    lengths = jnp.array([0.1, 0.2, 0.3, 0.1, 0.0])

    # Rate Matrix
    A = jax.random.normal(key, (num_states, num_states))
    Q = (A + A.T) / 2
    Q = Q - jnp.diag(jnp.sum(Q, axis=1)) # Row sum 0
    
    # Root & Tips
    pi = jnp.ones(num_states) / num_states
    tips = jax.random.uniform(key, (num_leaves, num_sites, num_states))
    tips = tips / jnp.sum(tips, axis=2, keepdims=True)

    # 2. Initialize the Equinox Module
    # This acts as our "callable" with internal state Q
    transition_module = TimeHomogeneousTransition(rate_matrix=Q)

    # 3. Run Standard Safe Function
    # We map over sites (axis 1 of tips)
    safe_fn = jax.jit(jax.vmap(
        fast_tree_likelihood_ops_safe, 
        in_axes=(None, None, None, None, 1)
    ))
    ll_safe = safe_fn(Q, pi, lengths, ops, tips)

    # 4. Run Custom Function with Module
    # Note: No static_argnums needed for transition_module because it's an Equinox Pytree!
    custom_fn = jax.jit(jax.vmap(
        fast_tree_likelihood_ops_callable, 
        in_axes=(None, None, None, None, 1)
    ))
    ll_custom = custom_fn(transition_module, pi, lengths, ops, tips)

    # 5. Compare
    print(f"Safe:   {ll_safe}")
    print(f"Custom: {ll_custom}")
    
    np.testing.assert_allclose(ll_custom, ll_safe, rtol=1e-6, atol=1e-6)
    print("Assertion Passed!")

if __name__ == "__main__":
    test_custom_ops_vs_safe_ops()
