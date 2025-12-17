import pytest
import numpy as np
import jax
import jax.numpy as jnp
from collections import defaultdict

# User-specified imports
import arborax
from arborax.pure import fast_tree_likelihood_ops
from arborax.io import parse_newick

def prepare_inputs_for_jax(edge_indices, edge_lengths):
    """Helper to convert parse_newick output into JAX-compatible Ops."""
    num_nodes = edge_indices.max() + 1
    aligned_branch_lengths = np.zeros(num_nodes)
    children_indices = edge_indices[:, 1]
    aligned_branch_lengths[children_indices] = edge_lengths
    
    children_map = defaultdict(list)
    for (p, c) in edge_indices:
        children_map[p].append(c)
        
    internal_parents = sorted(children_map.keys())
    ops_list = []
    for p in internal_parents:
        ops_list.append([p, children_map[p][0], children_map[p][1]])
        
    return np.array(ops_list, dtype=np.int32), aligned_branch_lengths

def test_jax_vs_reference_implementation():
    print("\n--- Starting JAX vs Arborax Comparison ---") # Visible with -s

    # 1. Setup Data
    seed = 42
    np.random.seed(seed)
    newick_str = "((A:0.1,B:0.2):0.3,C:0.4);"
    num_sites, num_states, num_leaves = 10, 4, 3

    # Generate random params
    A = np.random.rand(num_states, num_states)
    Q = (A + A.T) / 2
    np.fill_diagonal(Q, -Q.sum(axis=1) + Q.diagonal()) # Zero row sums
    pi = np.random.rand(num_states)
    pi /= pi.sum()
    tip_partials = np.random.rand(num_leaves, num_sites, num_states)
    tip_partials /= tip_partials.sum(axis=2, keepdims=True)

    # 2. Reference Run
    edge_indices, edge_lengths = parse_newick(newick_str, is_file_path=False)
    ref_logliks = arborax.loglik(
        tip_partials=tip_partials,
        edge_list=[tuple(e) for e in edge_indices],
        branch_lengths=edge_lengths.tolist(),
        Q=Q, pi=pi
    )

    # 3. JAX Run
    ops, aligned_lengths = prepare_inputs_for_jax(edge_indices, edge_lengths)
    
    # Compile and Run
    jax_loglik_fn = jax.jit(jax.vmap(fast_tree_likelihood_ops, in_axes=(None, None, None, None, 1)))
    jax_logliks = jax_loglik_fn(jnp.array(Q), jnp.array(pi), jnp.array(aligned_lengths), jnp.array(ops), jnp.array(tip_partials))

    # 4. Compare
    diff = np.abs(ref_logliks - jax_logliks)
    print(f"Max Difference: {diff.max():.2e}") # Visible with -s
    
    np.testing.assert_allclose(jax_logliks, ref_logliks, rtol=1e-5, atol=1e-6)
    print("Assertion Passed!") # Visible with -s

if __name__ == "__main__":
    test_jax_vs_reference_implementation()
