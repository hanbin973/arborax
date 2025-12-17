import pytest
import numpy as np
import jax
import jax.numpy as jnp
from collections import defaultdict

# Imports from your module structure
from arborax.pure import fast_tree_likelihood_ops, fast_tree_likelihood_ops_safe
from arborax.io import parse_newick

def prepare_inputs_for_jax(edge_indices, edge_lengths):
    """Helper to convert edge lists into JAX-compatible Ops and aligned arrays."""
    num_nodes = edge_indices.max() + 1
    aligned_branch_lengths = np.zeros(num_nodes)
    
    # Map branch lengths to the child node index
    children_indices = edge_indices[:, 1]
    aligned_branch_lengths[children_indices] = edge_lengths
    
    # Build Ops
    children_map = defaultdict(list)
    for (p, c) in edge_indices:
        children_map[p].append(c)
        
    internal_parents = sorted(children_map.keys())
    ops_list = []
    for p in internal_parents:
        children = children_map[p]
        # Ensure binary
        if len(children) != 2:
            raise ValueError(f"Node {p} is not binary.")
        ops_list.append([p, children[0], children[1]])
        
    return np.array(ops_list, dtype=np.int32), aligned_branch_lengths

def test_compare_naive_vs_safe_7_leaves():
    """
    Compares the naive (prob-space) and safe (log-space) likelihood implementations
    using a tree with 7 leaves to ensure they return identical results within tolerance.
    """
    print("\n--- Comparing Naive vs Safe Implementation (7 Leaves) ---")
    
    # 1. Setup Data
    seed = 123
    np.random.seed(seed)
    key = jax.random.PRNGKey(seed)
    
    # Constants
    num_leaves = 7
    num_states = 4 # DNA
    num_sites = 5  # Test with multiple sites
    
    # 7-Leaf Topology (Balanced-ish)
    # Leaves: 0..6
    newick_str = "((0:0.1,1:0.2):0.3,((2:0.1,3:0.2):0.2,(4:0.1,(5:0.2,6:0.1):0.3):0.1):0.2);"
    
    # Parse Topology
    edge_indices, edge_lengths = parse_newick(newick_str, is_file_path=False)
    ops, aligned_lengths = prepare_inputs_for_jax(edge_indices, edge_lengths)
    
    # Generate Random Parameters
    # Rate Matrix Q (Symmetric + Zero row sums)
    A = np.random.rand(num_states, num_states)
    Q = (A + A.T) / 2
    np.fill_diagonal(Q, 0)
    np.fill_diagonal(Q, -Q.sum(axis=1))
    
    # Root Frequencies pi
    pi = np.random.rand(num_states)
    pi /= pi.sum()
    
    # Leaf Data (Random Probabilities)
    tip_partials = np.random.rand(num_leaves, num_sites, num_states)
    tip_partials /= tip_partials.sum(axis=2, keepdims=True)

    # 2. Convert to JAX Arrays
    j_Q = jnp.array(Q)
    j_pi = jnp.array(pi)
    j_ops = jnp.array(ops)
    j_lengths = jnp.array(aligned_lengths)
    j_tips = jnp.array(tip_partials)

    # 3. Define JIT-compiled Functions with VMAP
    # Both functions handle a single site (N_leaves, N_states).
    # We vmap over axis 1 of leaf_data to handle multiple sites.
    
    # Naive (Probability Space)
    naive_fn = jax.jit(jax.vmap(
        fast_tree_likelihood_ops, 
        in_axes=(None, None, None, None, 1)
    ))
    
    # Safe (Log Space)
    safe_fn = jax.jit(jax.vmap(
        fast_tree_likelihood_ops_safe, 
        in_axes=(None, None, None, None, 1)
    ))

    # 4. Compute
    ll_naive = naive_fn(j_Q, j_pi, j_lengths, j_ops, j_tips)
    ll_safe = safe_fn(j_Q, j_pi, j_lengths, j_ops, j_tips)

    # 5. Compare
    # Print results for manual inspection (visible with pytest -s)
    print(f"Naive LogLik (first 3): {ll_naive[:3]}")
    print(f"Safe  LogLik (first 3): {ll_safe[:3]}")
    
    diff = np.abs(ll_naive - ll_safe)
    max_diff = np.max(diff)
    print(f"Max Absolute Difference: {max_diff:.2e}")

    # Assertion
    # We use a slightly looser tolerance because log-sum-exp accumulation order
    # differs numerically from direct multiplication, but they should be very close.
    np.testing.assert_allclose(
        ll_naive, 
        ll_safe, 
        rtol=1e-5, 
        atol=1e-6, 
        err_msg="Naive and Safe implementations diverged!"
    )
