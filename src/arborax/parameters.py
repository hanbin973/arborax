import jax
import jax.numpy as jnp
import jax.nn as jnn
import equinox as eqx
from typing import Tuple
from jaxtyping import Array, Float, PRNGKeyArray

class GTR(eqx.Module):
    num_states: int = eqx.field(static=True)
    logit_pi: Float[Array, "s"]
    q_factor: Float[Array, "s s"]

    def __init__(self, num_states: int, key: PRNGKeyArray):
        """
        Initializes the GTR substitution model parameters.
        """
        self.num_states = num_states
        
        # Initialize logit_pi as a constant zero vector
        self.logit_pi = jnp.zeros(num_states)
        
        # Initialize q_factor using a standard Gaussian (Normal) distribution
        self.q_factor = jax.random.normal(key, (num_states, num_states))

    def __call__(self) -> Tuple[Float[Array, "s s"], Float[Array, "s"]]:
        """
        Computes the rate probability matrix and root probability.
        """
        # recover equilibrium probability and exchangeability matrix
        pi = jnn.softmax(self.logit_pi)
        
        # Symmetrize q_factor to get exchangeability matrix
        q_triu = jnp.triu(self.q_factor)
        ex_matrix = jnn.softplus(q_triu + q_triu.T)
        
        # modify rate matrix diagonal
        rate_matrix = ex_matrix * pi[None, :]
        rate_matrix = jnp.fill_diagonal(rate_matrix, 0.0, inplace=False)
        rate_matrix = jnp.fill_diagonal(rate_matrix, -rate_matrix.sum(axis=1), inplace=False)
                
        return rate_matrix, pi
