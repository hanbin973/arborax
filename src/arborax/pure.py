import numpy as np

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn
from jax.scipy.linalg import expm
from jax.scipy.special import logsumexp
import equinox as eqx

from arborax.markov import GTR
from arborax.io import prepare_inputs_for_jax
from arborax.custom import fast_tree_likelihood_ops_custom

# ==========================================
# 1. Standard Implementation (Probability Space)
#    Fast, but prone to underflow on large trees.
# ==========================================

@jax.jit
def fast_tree_likelihood_ops(rate_matrix, root_probs, aligned_branch_lengths, operations, leaf_data):
    """
    Computes tree likelihood using Felsenstein's Pruning Algorithm in probability space.
    Best for small trees or short branches where underflow is not an issue.
    """
    num_nodes = aligned_branch_lengths.shape[0]
    num_leaves = leaf_data.shape[0]
    num_states = rate_matrix.shape[0]

    # Precompute transition matrices: P(t) = exp(Q * t)
    # transitions[i] corresponds to the branch ABOVE node i
    transitions = jax.vmap(lambda t: expm(rate_matrix * t))(aligned_branch_lengths)

    # Initialize Buffer
    buffer = jnp.zeros((num_nodes, num_states))
    buffer = buffer.at[:num_leaves].set(leaf_data)

    def execute_op(current_buffer, op):
        parent_idx, left_idx, right_idx = op

        # Gather inputs
        L_left = current_buffer[left_idx]
        L_right = current_buffer[right_idx]
        P_left = transitions[left_idx]
        P_right = transitions[right_idx]

        # Pruning Step: parent = (P_left @ L_left) * (P_right @ L_right)
        from_left = jnp.dot(P_left, L_left)
        from_right = jnp.dot(P_right, L_right)

        parent_clv = from_left * from_right

        # Update buffer
        new_buffer = current_buffer.at[parent_idx].set(parent_clv)
        return new_buffer, None

    final_buffer, _ = jax.lax.scan(execute_op, buffer, operations)

    # Root Likelihood
    root_idx = operations[-1, 0]
    root_clv = final_buffer[root_idx]

    # Final dot product with root equilibrium frequencies
    return jnp.log(jnp.dot(root_clv, root_probs))


# ==========================================
# 2. Safe Implementation (Log Space)
#    Slower due to logsumexp, but numerically stable for all tree sizes.
# ==========================================

MIN_LOG_VAL = -1e18

def safe_log(x):
    """Computes log(x) with a floor to prevent -inf."""
    return jnp.where(x > 0, jnp.log(x), MIN_LOG_VAL)

@jax.jit
def fast_tree_likelihood_ops_safe(rate_matrix, root_probs, aligned_branch_lengths, operations, leaf_data):
    """
    Computes tree likelihood in LOG-SPACE.
    Prevents underflow for large trees by using the Log-Sum-Exp trick.
    """
    num_nodes = aligned_branch_lengths.shape[0]
    num_leaves = leaf_data.shape[0]
    num_states = rate_matrix.shape[0]

    # 1. Precompute Log-Transition Matrices
    #    We compute P=expm(Q*t) normally, then take safe_log(P).
    def get_log_transition(t):
        P = expm(rate_matrix * t)
        return safe_log(P)

    log_transitions = jax.vmap(get_log_transition)(aligned_branch_lengths)

    # 2. Initialize Buffer in Log Space
    log_buffer = jnp.full((num_nodes, num_states), MIN_LOG_VAL)

    #    Convert leaf data (probs) to log-probs
    log_leaf_vals = safe_log(leaf_data)
    log_buffer = log_buffer.at[:num_leaves].set(log_leaf_vals)

    # 3. Define Log-Space Op Kernel
    def execute_log_op(current_log_buffer, op):
        parent_idx, left_idx, right_idx = op

        # Fetch data
        log_L_left = current_log_buffer[left_idx]   # (S,)
        log_L_right = current_log_buffer[right_idx] # (S,)
        log_P_left = log_transitions[left_idx]      # (S, S)
        log_P_right = log_transitions[right_idx]    # (S, S)

        # Pruning Step in Log Space (LogSumExp instead of Dot Product)
        # Broadcasting: log_P (S,S) + log_L (S,) adds log_L to every row of log_P.
        # Summing over axis 1 collapses the child state j, leaving parent state i.
        log_from_left = logsumexp(log_P_left + log_L_left, axis=1)
        log_from_right = logsumexp(log_P_right + log_L_right, axis=1)

        # Combine branches: Product in probability space = Sum in log space
        log_parent_clv = log_from_left + log_from_right

        new_buffer = current_log_buffer.at[parent_idx].set(log_parent_clv)
        return new_buffer, None

    final_log_buffer, _ = jax.lax.scan(execute_log_op, log_buffer, operations)

    # 4. Root Likelihood
    root_idx = operations[-1, 0]
    log_root_clv = final_log_buffer[root_idx]
    log_root_probs = safe_log(root_probs)

    # Final dot product in log space
    total_log_likelihood = logsumexp(log_root_clv + log_root_probs)

    return total_log_likelihood

@jax.jit
def fast_tree_likelihood_ops_callable(log_transition_fn, root_probs, aligned_branch_lengths, operations, leaf_data):
    """
    Computes tree likelihood in LOG-SPACE using a CUSTOM transition function.

    Args:
        log_transition_fn: A callable function `f(t) -> (S, S)` that returns
                           the log-transition matrix for a given branch length t.
        root_probs: (S,) Equilibrium/Root probabilities (in normal probability space).
        aligned_branch_lengths: (N,) Branch lengths aligned to node indices.
        operations: (N_internal, 3) Ops array.
        leaf_data: (L, S) Leaf data (tip partials) in normal probability space.
    """
    num_nodes = aligned_branch_lengths.shape[0]
    num_leaves = leaf_data.shape[0]
    num_states = root_probs.shape[0]

    # 1. Precompute Log-Transition Matrices using the Callable
    #    We vmap the user-provided function over the branch lengths.
    log_transitions = jax.vmap(log_transition_fn)(aligned_branch_lengths)

    # 2. Initialize Buffer in Log Space
    log_buffer = jnp.full((num_nodes, num_states), MIN_LOG_VAL)

    #    Convert leaf data (probs) to log-probs
    log_leaf_vals = safe_log(leaf_data)
    log_buffer = log_buffer.at[:num_leaves].set(log_leaf_vals)

    # 3. Define Log-Space Op Kernel (Identical to previous safe version)
    def execute_log_op(current_log_buffer, op):
        parent_idx, left_idx, right_idx = op

        # Fetch data
        log_L_left = current_log_buffer[left_idx]
        log_L_right = current_log_buffer[right_idx]
        log_P_left = log_transitions[left_idx]
        log_P_right = log_transitions[right_idx]

        # Pruning Step: logsumexp(log_P + log_L)
        log_from_left = logsumexp(log_P_left + log_L_left, axis=1)
        log_from_right = logsumexp(log_P_right + log_L_right, axis=1)

        # Combine: Sum in log space
        log_parent_clv = log_from_left + log_from_right

        new_buffer = current_log_buffer.at[parent_idx].set(log_parent_clv)
        return new_buffer, None

    # 4. Run Scan
    final_log_buffer, _ = jax.lax.scan(execute_log_op, log_buffer, operations)

    # 5. Root Likelihood
    root_idx = operations[-1, 0]
    log_root_clv = final_log_buffer[root_idx]
    log_root_probs = safe_log(root_probs)

    total_log_likelihood = logsumexp(log_root_clv + log_root_probs)

    return total_log_likelihood

class TreeLikelihood(eqx.Module):
    """
    A differentiable layer that binds a learnable GTR model to a fixed tree topology.
    """
    gtr: GTR
    operations: jax.Array
    aligned_branch_lengths: jax.Array
    
    def __init__(
        self, 
        gtr: GTR, 
        edge_indices: jax.Array, 
        edge_lengths: jax.Array
    ):
        """
        Args:
            gtr: An instance of arborax.markov.GTR.
            edge_indices: (E, 2) array of [parent, child] indices.
            edge_lengths: (E,) array of branch lengths.
        """
        self.gtr = gtr
        
        # Delegate topology compilation to the IO module
        # fast_tree_likelihood_ops_callable expects integer ops and float lengths
        ops, lengths = prepare_inputs_for_jax(edge_indices, edge_lengths)
        
        # Store as constant JAX arrays (frozen part of the model)
        self.operations = jnp.array(ops, dtype=jnp.int32)
        self.aligned_branch_lengths = jnp.array(lengths, dtype=jnp.float32)

    def __call__(self, leaf_data: jax.Array) -> jax.Array:
        """
        Computes the log-likelihood of the leaf data.
        
        Args:
            leaf_data: Shape (NumLeaves, NumStates) or (NumLeaves, NumSites, NumStates).
        """
        # 1. Freeze branch lengths (Topology is fixed during optimization)
        lengths = jax.lax.stop_gradient(self.aligned_branch_lengths)
        
        # 2. Prepare the Kernel Inputs
        transition_fn = self.gtr
        root_probs = self.gtr.stationary_probs
        
        # 3. Handle Vectorization (Multiple Sites)
        if leaf_data.ndim == 3:
            # Shape: (Leaves, Sites, States) -> vmap over axis 1 (Sites)
            # We map only leaf_data; everything else is broadcasted.
            likelihood_fn = jax.vmap(
                fast_tree_likelihood_ops_callable, 
                in_axes=(None, None, None, None, 1)
            )
            
            per_site_ll = likelihood_fn(
                transition_fn, root_probs, lengths, self.operations, leaf_data
            )
            
            return jnp.mean(per_site_ll)
            
        else:
            # Shape: (Leaves, States) -> Single site
            return fast_tree_likelihood_ops_custom(
                transition_fn, root_probs, lengths, self.operations, leaf_data
            )

class KLD(eqx.Module):
    tree_likelihood: TreeLikelihood
    num_mc: int = eqx.field(static=True)

    def __init__(
        self, 
        gtr, 
        edge_indices, 
        edge_lengths, 
        num_mc: int
    ):
        """
        Args:
            gtr: GTR model instance.
            edge_indices: Topology indices.
            edge_lengths: Branch lengths.
            num_mc: Number of Monte Carlo samples to draw per leaf.
        """
        self.tree_likelihood = TreeLikelihood(gtr, edge_indices, edge_lengths)
        self.num_mc = num_mc

    def __call__(self, leaf_data: jax.Array, key: jax.random.PRNGKey) -> jax.Array:
        """
        Samples states at the tips and computes the average log-likelihood.

        Args:
            leaf_data: (num_leaves, num_states) matrix of probabilities.
            key: PRNGKey for the sampling process.
        """
        num_leaves, num_states = leaf_data.shape

        # 1. Prepare Logits
        # Logits for categorical sampling are just the log-probabilities.
        # We add epsilon to prevent log(0) -> -inf issues.
        logits = jnp.log(leaf_data + 1e-30)

        # 2. Vectorized Categorical Sampling
        # By passing shape=(num_mc, num_leaves), JAX treats the leading 
        # dimension of 'logits' (num_leaves) as a batch.
        # Output shape: (num_mc, num_leaves)
        sample_indices = jr.categorical(key, logits, shape=(self.num_mc, num_leaves))

        # 3. Reshape and Transform
        # We need (num_leaves, num_mc, num_states) for TreeLikelihood.
        # First, transpose to (num_leaves, num_mc)
        sample_indices = jnp.transpose(sample_indices)

        # 4. Convert to One-Hot
        # Output shape: (num_leaves, num_mc, num_states)
        sampled_one_hot = jnn.one_hot(sample_indices, num_classes=num_states)

        # 5. Compute Likelihood
        # This calls TreeLikelihood.__call__, which detects ndim=3 and 
        # internally uses vmap(fast_tree_likelihood_ops_callable, in_axes=(..., 1))
        # and returns jnp.mean(per_site_ll).
        return - self.tree_likelihood(sampled_one_hot)
