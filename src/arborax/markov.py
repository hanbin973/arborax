import jax
import jax.numpy as jnp
import jax.nn as jnn
import jax.random as jr
import equinox as eqx
from jaxtyping import Array, Float, PRNGKeyArray
from typing import Optional

class GTR(eqx.Module):
    """
    General Time Reversible (GTR) substitution model.
    
    Parameterizes the rate matrix Q via:
    1. Equilibrium probabilities (pi)
    2. Symmetric exchangeability matrix (R)
    
    Q_ij = R_ij * pi_j
    
    Computes log(exp(Q * t)) efficiently via eigendecomposition of the 
    symmetrized matrix.
    """
    num_states: int = eqx.field(static=True)
    logit_stationary_probs: Float[Array, "s"]
    exchangeability_params: Float[Array, "s s"]
    
    def __init__(
        self, 
        num_states: int, 
        key: PRNGKeyArray,
        init_rate_matrix: Optional[Float[Array, "s s"]] = None,
    ):
        self.num_states = num_states
        if init_rate_matrix is None:
            # Initialize stationary distribution to roughly uniform
            self.logit_stationary_probs = jnp.zeros(num_states)
            
            # Initialize exchangeability parameters (roughly uniform rates)
            self.exchangeability_params = jr.normal(key, shape=(num_states, num_states))
        else:
            raise NotImplementedError("Custom rate matrix initialization not implemented.")

    @property
    def stationary_probs(self) -> Float[Array, "s"]:
        """Returns the equilibrium probabilities (pi)."""
        return jnn.softmax(self.logit_stationary_probs)

    def __call__(self, t: Float[Array, ""]) -> Float[Array, "s s"]:
        """
        Computes the log-transition matrix log(P(t)) = log(exp(Q * t)).
        """
        # 1. Recover Parameters
        pi = self.stationary_probs
        sqrt_pi = jnp.sqrt(pi)
        inv_sqrt_pi = 1.0 / sqrt_pi

        # Symmetrize exchangeability matrix (R)
        R_param_sym = jnp.triu(self.exchangeability_params) + jnp.tril(self.exchangeability_params.T)
        R = jnn.softplus(R_param_sym)
        
        # 2. Construct Rate Matrix Q
        # Definition: Q_ij = R_ij * pi_j
        Q = R * pi[None, :]
        
        # Zero diagonal and set to -row_sums
        Q = jnp.fill_diagonal(Q, 0.0, inplace=False)
        row_sums = Q.sum(axis=1)
        Q = jnp.fill_diagonal(Q, -row_sums, inplace=False)
        
        # 3. Eigendecomposition of Symmetrized Matrix S
        # S = D^(1/2) Q D^(-1/2)
        # This matrix is symmetric, guaranteeing real eigenvalues/vectors
        S = sqrt_pi[:, None] * Q * inv_sqrt_pi[None, :]
        
        eigenvalues, eigenvectors = jnp.linalg.eigh(S)
        
        # 4. Compute P(t) = exp(Q * t)
        # Since eigenvalues <= 0, exp(eigenvalues * t) is in (0, 1].
        # No shift is needed; overflow is impossible. 
        # Underflow to 0 is acceptable (fast decaying modes).
        
        scaled_evals = eigenvalues * t
        
        # Reconstruct: V @ diag(exp(evals)) @ V.T
        # Broadcasting: eigenvectors * exp_val[None,:] scales columns by eigenvalues
        weighted_vectors = eigenvectors * jnp.exp(scaled_evals)[None, :]
        inner_matrix = weighted_vectors @ eigenvectors.T
        
        # Transform back to asymmetric space
        # P = D^(-1/2) * inner * D^(1/2)
        P = inv_sqrt_pi[:, None] * inner_matrix * sqrt_pi[None, :]
        
        # 5. Log Transformation
        # Clip to a small epsilon to prevent log(0) or log(negative_noise)
        # 1e-30 is safe for float32/64 to avoid -inf while preserving "practically zero"
        P_clipped = jnp.maximum(P, 1e-30)
        
        return jnp.log(P_clipped)
