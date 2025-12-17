import pytest
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import numpy as np
from jax import value_and_grad

from arborax.custom import fast_tree_likelihood_ops_custom
from arborax.pure import fast_tree_likelihood_ops_callable

# -----------------------------------------------------------------------------
# 1. Learnable Equinox Module
# -----------------------------------------------------------------------------

class LearnableJC69(eqx.Module):
    """
    JC69 Model with a learnable rate parameter.
    """
    # Learnable parameter (default 1.0). 
    # We use a raw scalar array.
    rate: jax.Array 

    def __init__(self, init_rate=1.0):
        self.rate = jnp.array(init_rate)

    def __call__(self, t: jax.Array) -> jax.Array:
        # Constrain rate to be positive (softplus or abs) to avoid NaN
        r = jnp.abs(self.rate)
        t = jnp.clip(t, a_min=0.0)
        
        # JC69: rate_factor = -4/3 * r
        rate_factor = -1.3333333 * r
        exp_term = jnp.exp(rate_factor * t)
        
        p_stay = 0.25 + 0.75 * exp_term
        p_change = 0.25 - 0.25 * exp_term
        
        log_diag = jnp.log(p_stay + 1e-30)
        log_off = jnp.log(p_change + 1e-30)
        
        mat = jnp.full((4, 4), log_off)
        eye = jnp.eye(4)
        mat = mat * (1 - eye) + log_diag * eye
        
        return mat

# -----------------------------------------------------------------------------
# 2. Test Fixtures (Same as before)
# -----------------------------------------------------------------------------

@pytest.fixture
def tree_data():
    key = jr.PRNGKey(42)
    k1, k2, k3 = jr.split(key, 3)

    num_tips = 5
    num_states = 4
    num_internal = num_tips - 1
    num_nodes = num_tips + num_internal

    operations = jnp.array([
        [5, 0, 1], [6, 2, 3], [7, 5, 6], [8, 7, 4]
    ])

    branch_lengths = jr.uniform(k1, (num_nodes,)) * 0.5 + 0.01
    leaf_data = jr.uniform(k2, (num_tips, num_states))
    leaf_data /= jnp.sum(leaf_data, axis=1, keepdims=True)
    root_probs = jr.dirichlet(k3, jnp.ones(num_states))

    return {
        "aligned_branch_lengths": branch_lengths,
        "operations": operations,
        "leaf_data": leaf_data,
        "root_probs": root_probs
    }

def print_diff(name, val_custom, val_base):
    diff = np.abs(val_custom - val_base)
    denom = np.abs(val_base) + 1e-9
    rel_err = np.max(diff / denom)
    status = "PASS" if rel_err < 1e-4 else "FAIL"
    print(f"[{status}] {name:<20} | Rel Err: {rel_err:.2e}")

# -----------------------------------------------------------------------------
# 3. New Test for Rate Gradients
# -----------------------------------------------------------------------------

def test_rate_gradient_consistency(tree_data):
    """
    Verifies that gradients flow correctly through the Equinox module parameters (Rate Matrix).
    """
    print(f"\n{'='*10} Rate Parameter Gradient Test {'='*10}")
    
    # Initialize with a specific rate
    model = LearnableJC69(init_rate=2.5)

    # Loss function that takes the MODEL as the first argument
    # (so we can differentiate w.r.t it)
    def loss_fn(m, b_len, r_probs, l_data, impl_fn):
        return impl_fn(
            m,
            r_probs,
            b_len,
            tree_data["operations"],
            l_data
        )

    # 1. Custom VJP Gradients (Arg 0 is the model)
    # eqx.filter_value_and_grad handles the PyTree structure of the module
    func_custom = lambda m: loss_fn(
        m, 
        tree_data["aligned_branch_lengths"], 
        tree_data["root_probs"], 
        tree_data["leaf_data"],
        fast_tree_likelihood_ops_custom
    )
    val_custom, grads_custom = eqx.filter_value_and_grad(func_custom)(model)

    # 2. Baseline Gradients
    # To differentiate the model in the baseline (which treats it as a callable),
    # we pass the same module. 'callable' implementation relies on JAX autodiff,
    # which will trace through the module's __call__ just fine.
    func_base = lambda m: loss_fn(
        m, 
        tree_data["aligned_branch_lengths"], 
        tree_data["root_probs"], 
        tree_data["leaf_data"],
        fast_tree_likelihood_ops_callable
    )
    val_base, grads_base = eqx.filter_value_and_grad(func_base)(model)

    # Compare Values
    print_diff("Log Likelihood", val_custom, val_base)
    
    # Compare Gradient w.r.t Rate
    # The gradient is returned as a matching PyTree (LearnableJC69 instance)
    grad_rate_custom = grads_custom.rate
    grad_rate_base = grads_base.rate
    
    print(f"Custom Rate Grad: {grad_rate_custom}")
    print(f"Base   Rate Grad: {grad_rate_base}")
    
    print_diff("Rate Parameter Grad", grad_rate_custom, grad_rate_base)

    np.testing.assert_allclose(
        grad_rate_custom, grad_rate_base, 
        rtol=1e-4, atol=1e-6, 
        err_msg="Gradient mismatch for Learnable Rate"
    )

# -----------------------------------------------------------------------------
# 4. Standard Tests (Updated to use LearnableJC69)
# -----------------------------------------------------------------------------

def test_standard_gradients(tree_data):
    """Checks branch length, root, and leaf gradients."""
    print(f"\n{'='*10} Standard Gradients Test {'='*10}")
    model = LearnableJC69() # Default rate
    
    def loss_wrapper(b, r, l, fn):
        return fn(model, r, b, tree_data["operations"], l)

    # Differentiate args 0,1,2 (branch, root, leaf)
    # We use jax.value_and_grad here since we aren't diffing the model
    val_c, grads_c = value_and_grad(
        lambda b, r, l: loss_wrapper(b, r, l, fast_tree_likelihood_ops_custom),
        argnums=(0, 1, 2)
    )(tree_data["aligned_branch_lengths"], tree_data["root_probs"], tree_data["leaf_data"])

    val_b, grads_b = value_and_grad(
        lambda b, r, l: loss_wrapper(b, r, l, fast_tree_likelihood_ops_callable),
        argnums=(0, 1, 2)
    )(tree_data["aligned_branch_lengths"], tree_data["root_probs"], tree_data["leaf_data"])

    print_diff("Branch Lengths", grads_c[0], grads_b[0])
    print_diff("Root Probs", grads_c[1], grads_b[1])
    print_diff("Leaf Data", grads_c[2], grads_b[2])

    np.testing.assert_allclose(grads_c[0], grads_b[0], rtol=1e-4, err_msg="Branch Grad Fail")
    np.testing.assert_allclose(grads_c[1], grads_b[1], rtol=1e-4, err_msg="Root Grad Fail")
    np.testing.assert_allclose(grads_c[2], grads_b[2], rtol=1e-4, err_msg="Leaf Grad Fail")
