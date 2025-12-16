import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Int, Float, PRNGKeyArray
from typing import Any

from arborax import loglik
from arborax.parameters import GTR

class GTRTree(eqx.Module):
    # Field definitions with jaxtyping annotations
    edge_list: Int[Array, "num_edges 2"]
    branch_lengths: Float[Array, "num_edges"]
    transition_kernel: GTR

    def __init__(
            self, 
            edge_list: Int[Array, "E 2"], 
            branch_lengths: Float[Array, "E"],
            num_states: int,
            key: PRNGKeyArray,
            ):
        """
        Initializes the GTRTree module.

        Args:
            edge_list: List or array of edge indices (parent, child).
            branch_lengths: List or array of branch lengths.
            num_states: Number of character states (e.g., 4 for DNA).
            key: JAX PRNG key for initializing the GTR parameters.
        """
        # 1. Convert inputs to JAX arrays
        # We assign to local variables first to perform validation before setting fields
        edge_list_jax = jnp.array(edge_list, dtype=int)
        branch_lengths_jax = jnp.array(branch_lengths, dtype=float)

        # 2. Validation: Check that dimension 0 (number of edges) matches
        if edge_list_jax.shape[0] != branch_lengths_jax.shape[0]:
            raise ValueError(
                f"Shape Mismatch: edge_list has {edge_list_jax.shape[0]} edges, "
                f"but branch_lengths has {branch_lengths_jax.shape[0]} lengths."
            )

        # 3. Assignment
        # Equinox allows standard assignment syntax inside __init__
        self.edge_list = edge_list_jax
        self.branch_lengths = branch_lengths_jax
        self.transition_kernel = GTR(num_states, key=key)

    def __call__(self, tip_partials: Float[Array, "tips states"]) -> Float[Array, ""]:
        """
        Forward pass: Calculates the log-likelihood of the tree.
        
        Args:
            tip_partials: A (num_tips, num_states) array of observed data 
                          (one-hot or probabilistic).
                          
        Returns:
            Scalar log-likelihood.
        """
        # 1. Get current GTR matrices (Q and pi)
        # The transition_kernel handles the parameter constraints internally
        Q, pi = self.transition_kernel()

        # 2. Delegate to the external loglik function
        # We pass self.edge_list and self.branch_lengths which are stored in this Module.
        # Note: If loglik is JAX-compatible, this will be fully differentiable.
        return loglik(
            tip_partials=tip_partials,
            edge_list=self.edge_list,
            branch_lengths=self.branch_lengths,
            Q=Q,
            pi=pi
        )
