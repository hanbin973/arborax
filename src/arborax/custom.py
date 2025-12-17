import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jax import custom_vjp

MIN_LOG_VAL = -1e18

def safe_log(x):
    """Computes log(x) with a floor to prevent -inf."""
    return jnp.where(x > 0, jnp.log(x), MIN_LOG_VAL)

# -----------------------------------------------------------------------------
# 1. The Core Tree Logic (Differentiable Boundary)
# -----------------------------------------------------------------------------

@custom_vjp
def _tree_traversal_core(log_buffer_init, log_transitions, operations, root_probs_log):
    """
    The inner core function that we want to define a custom gradient for.
    It takes the initialized leaf buffer and transitions and returns the total log-likelihood.
    """
    # --- Forward Scan Definition ---
    def execute_log_op(current_log_buffer, op):
        parent_idx, left_idx, right_idx = op
        
        # Gather inputs
        log_L_left = current_log_buffer[left_idx]
        log_L_right = current_log_buffer[right_idx]
        log_P_left = log_transitions[left_idx]
        log_P_right = log_transitions[right_idx]
        
        # Compute messages (Pruning)
        # log_msg = logsumexp(log_P + log_child, axis=1)
        # Broadcasting: (S, S) + (S,) -> (S, S) sum over child states (axis 1)
        log_from_left = logsumexp(log_P_left + log_L_left, axis=1)
        log_from_right = logsumexp(log_P_right + log_L_right, axis=1)
        
        log_parent_clv = log_from_left + log_from_right
        
        # Update buffer
        new_buffer = current_log_buffer.at[parent_idx].set(log_parent_clv)
        return new_buffer, None

    # Run Scan
    final_log_buffer, _ = jax.lax.scan(execute_log_op, log_buffer_init, operations)
    
    # Root Likelihood
    root_idx = operations[-1, 0]
    log_root_clv = final_log_buffer[root_idx]
    
    total_log_likelihood = logsumexp(log_root_clv + root_probs_log)
    
    return total_log_likelihood

# -----------------------------------------------------------------------------
# 2. Forward Pass Implementation (Returns Residuals)
# -----------------------------------------------------------------------------

def _tree_core_fwd(log_buffer_init, log_transitions, operations, root_probs_log):
    # We basically run the exact same logic as above, but we save the 'final_log_buffer'
    # so we can use it in the backward pass. 
    
    def execute_log_op(current_log_buffer, op):
        parent_idx, left_idx, right_idx = op
        log_L_left = current_log_buffer[left_idx]
        log_L_right = current_log_buffer[right_idx]
        log_P_left = log_transitions[left_idx]
        log_P_right = log_transitions[right_idx]
        
        log_from_left = logsumexp(log_P_left + log_L_left, axis=1)
        log_from_right = logsumexp(log_P_right + log_L_right, axis=1)
        
        log_parent_clv = log_from_left + log_from_right
        new_buffer = current_log_buffer.at[parent_idx].set(log_parent_clv)
        return new_buffer, None

    final_log_buffer, _ = jax.lax.scan(execute_log_op, log_buffer_init, operations)
    
    root_idx = operations[-1, 0]
    total_log_likelihood = logsumexp(final_log_buffer[root_idx] + root_probs_log)
    
    # Save for backward: final_log_buffer, inputs
    return total_log_likelihood, (final_log_buffer, log_transitions, operations, root_probs_log)

# -----------------------------------------------------------------------------
# 3. Custom Backward Pass (Ancestral Gradient Reconstruction)
# -----------------------------------------------------------------------------

def _tree_core_bwd(residuals, g_log_lik):
    """
    Manual backward pass.
    g_log_lik: Scalar gradient of the total log likelihood.
    """
    final_log_buffer, log_transitions, operations, root_probs_log = residuals
    num_nodes, num_states = final_log_buffer.shape
    
    # Initialize Gradient Buffer for CLVs (Partial Likelihoods)
    # We will accumulate gradients here as we traverse down from root.
    grad_log_buffer = jnp.zeros((num_nodes, num_states))
    
    # --- 1. Gradient at Root ---
    root_idx = operations[-1, 0]
    log_root_clv = final_log_buffer[root_idx]
    
    # d(LL) / d(Root_CLV) = g * softmax(Root_CLV + Root_Probs)
    # d(LL) / d(Root_Probs) = g * softmax(Root_CLV + Root_Probs)
    common_softmax = jax.nn.softmax(log_root_clv + root_probs_log)
    grad_root_clv = g_log_lik * common_softmax
    grad_root_probs_log = g_log_lik * common_softmax
    
    # Set root gradient in buffer
    grad_log_buffer = grad_log_buffer.at[root_idx].set(grad_root_clv)

    # --- 2. Backward Scan (Root -> Leaves) ---
    # We iterate operations in REVERSE.
    reverse_ops = operations[::-1]

    def backward_step(current_grad_buffer, op):
        parent_idx, left_idx, right_idx = op
        
        # Retrieve computed values from forward pass (Rematerialization)
        # We need these to compute the local Jacobians (softmaxes)
        log_L_left = final_log_buffer[left_idx]
        log_L_right = final_log_buffer[right_idx]
        log_P_left = log_transitions[left_idx]
        log_P_right = log_transitions[right_idx]
        
        # Get Gradient at Parent
        grad_parent = current_grad_buffer[parent_idx] # (S,)
        
        # --- Gradients for Left Branch ---
        # Forward eq: log_from_left[i] = logsumexp_j(P[i,j] + L_left[j])
        # Local Jacobian: softmax(P[i,:] + L_left)
        
        # We compute the weighted softmax for the transition matrix
        # Shape logic: (S, 1) + (S,) -> (S, S). Softmax over axis 1 (child states)
        # s_left[i, j] = exp(P[i,j] + L[j]) / sum(...)
        s_left = jax.nn.softmax(log_P_left + log_L_left, axis=1)
        s_right = jax.nn.softmax(log_P_right + log_L_right, axis=1)

        # Gradient w.r.t Transition Matrix (Left)
        # dL/dP_ij = grad_parent[i] * s_left[i,j]
        grad_P_left = grad_parent[:, None] * s_left
        
        # Gradient w.r.t Child CLV (Left)
        # dL/dL_j = sum_i (grad_parent[i] * s_left[i,j])
        grad_L_left = jnp.sum(grad_P_left, axis=0)

        # --- Gradients for Right Branch ---
        grad_P_right = grad_parent[:, None] * s_right
        grad_L_right = jnp.sum(grad_P_right, axis=0)
        
        # Update Gradient Buffer for children
        # Note: In a tree, every node is a child exactly once, so we can set/add.
        # 'set' is safer if we assume strict tree (visited once).
        new_grad_buffer = current_grad_buffer.at[left_idx].set(grad_L_left)
        new_grad_buffer = new_grad_buffer.at[right_idx].set(grad_L_right)
        
        # Output the gradients for P_left and P_right to be stacked
        return new_grad_buffer, (grad_P_left, grad_P_right, left_idx, right_idx)

    # Run Reverse Scan
    _, (grads_P_left, grads_P_right, idxs_left, idxs_right) = jax.lax.scan(
        backward_step, grad_log_buffer, reverse_ops
    )
    
    # --- 3. Scatter Transition Gradients back to full array ---
    # We generated gradients for P in the order of operations. We need to map them 
    # back to the original (N, S, S) shape of `log_transitions`.
    
    grad_transitions = jnp.zeros_like(log_transitions)
    
    # We have two streams of gradients (left children and right children)
    grad_transitions = grad_transitions.at[idxs_left].set(grads_P_left)
    grad_transitions = grad_transitions.at[idxs_right].set(grads_P_right)
    
    # Return gradients: (log_buffer_init, log_transitions, operations, root_probs_log)
    # operations is integer, so no gradient (None)
    # log_buffer_init gradient is zero for internal nodes, but valid for leaves.
    # The 'final_grad_buffer' contains gradients for all nodes. 
    # But strictly, the function input was 'log_buffer_init'.
    # We can just return the relevant slice or the whole thing? 
    # Actually, JAX handles the slice if we return the full shape.
    
    return _, grad_transitions, None, grad_root_probs_log

# Register the VJP
_tree_traversal_core.defvjp(_tree_core_fwd, _tree_core_bwd)

# -----------------------------------------------------------------------------
# 4. Public API Wrapper
# -----------------------------------------------------------------------------

@jax.jit
def fast_tree_likelihood_ops_custom(log_transition_fn, root_probs, aligned_branch_lengths, operations, leaf_data):
    """
    Optimized Tree Likelihood using Custom VJP.
    """
    num_nodes = aligned_branch_lengths.shape[0]
    num_leaves = leaf_data.shape[0]
    num_states = root_probs.shape[0]

    # 1. Precompute Log-Transition Matrices (Standard AD works here)
    #    The custom VJP starts *after* this, isolating the tree structure overhead.
    log_transitions = jax.vmap(log_transition_fn)(aligned_branch_lengths)

    # 2. Initialize Buffer
    log_buffer = jnp.full((num_nodes, num_states), MIN_LOG_VAL)
    log_leaf_vals = safe_log(leaf_data)
    log_buffer = log_buffer.at[:num_leaves].set(log_leaf_vals)
    
    # 3. Root Probs Log
    root_probs_log = safe_log(root_probs)

    # 4. Call Core (Custom VJP handles the recursion)
    return _tree_traversal_core(log_buffer, log_transitions, operations, root_probs_log)
