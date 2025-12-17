import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Int, Float, PRNGKeyArray
from typing import Any, Tuple

from arborax import loglik, create_arborax_context
from arborax.context import ArboraxContext
from arborax.parameters import GTR

class GTRTree(eqx.Module):
    # Field definitions with jaxtyping annotations
    partials_shape: Tuple[int, int, int] = eqx.field(static=True)
    edge_lengths: Float[Array, "E"]
    context: ArboraxContext = eqx.field(static=True)
    transition_kernel: GTR 

    def __init__(
            self, 
            edge_list: Int[Array, "E 2"], 
            branch_lengths: Float[Array, "E"],
            partials_shape: Tuple[int, int, int],
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
        self.partials_shape = partials_shape
        self.edge_lengths, self.context = create_arborax_context(
                edge_list, branch_lengths, partials_shape
                )
        self.transition_kernel = GTR(partials_shape[2], key=key)

    def __call__(self) -> Float[Array, ""]:
        """
        Forward pass: Calculates the log-likelihood of the tree.
        
        Args:
            tip_partials: A (num_tips, num_states) array of observed data 
                          (one-hot or probabilistic).
                          
        Returns:
            Scalar log-likelihood.
        """
        Q, pi = self.transition_kernel()
        frozen_lengths = jax.lax.stop_gradient(self.edge_lengths)
        return self.context.likelihood_functional(
                Q,
                pi,
                self.edge_lengths
            )

    def bind_partials(self, tip_partials_numpy: np.ndarray):
        """
        Explicitly bind data to the C++ context.
        Call this ONCE before training with a standard NumPy array.
        """
        tip_count = self.partials_shape[0]
        num_states = self.partials_shape[2]

        # Create dict mapping tip_index -> sequence data
        tip_dict = {idx: tip_partials_numpy[idx] for idx in range(tip_count)}

        # We pass dummy Q and pi just to satisfy the function signature.
        # They are IGNORED by likelihood_functional during training.
        dummy_Q = np.eye(num_states)
        dummy_pi = np.ones(num_states) / num_states

        self.context.bind_data(tip_dict, dummy_Q, dummy_pi)
