import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from ..lowlevel.beagle_cffi import BeagleLikelihoodCalculator

class BeagleJAX:
    def __init__(self, 
                 tip_count, 
                 state_count, 
                 pattern_count, 
                 tip_data, 
                 edge_map, 
                 operations,
                 pattern_weights=None,
                 use_gpu=False,
                 dtype=jnp.float32): # Default back to float32
        
        self.tip_count = tip_count
        self.node_count = 2 * tip_count - 1
        self.operations = operations
        self.pattern_count = pattern_count
        self.dtype = dtype
        self.state_count = state_count
        
        # 1. Initialize Backend
        self.backend = BeagleLikelihoodCalculator(
            tip_count=tip_count,
            state_count=state_count,
            pattern_count=pattern_count,
            use_gpu=use_gpu
        )
        self.backend.set_tip_partials(tip_data)
        self.backend.set_pattern_weights(pattern_weights)
        
        sorted_nodes = sorted([k for k in edge_map.keys()])
        self.prob_indices = np.array(sorted_nodes, dtype=np.int32)
        self.root_index = operations[-1]['dest']
        self.c_operations = self.backend.prepare_operations(operations)
        self.operation_count = len(operations)
        
        self.parents = np.full(self.node_count, -1, dtype=np.int32)
        for child, (parent, _) in edge_map.items():
            self.parents[child] = parent

        self._log_likelihood_vjp = jax.custom_vjp(self._log_likelihood_impl)
        self._log_likelihood_vjp.defvjp(self._log_likelihood_fwd, self._log_likelihood_bwd)

    def log_likelihood(self, edge_lengths, Q, pi, tip_partials=None):
        return self._log_likelihood_vjp(edge_lengths, Q, pi, tip_partials)

    def _log_likelihood_impl(self, edge_lengths, Q, pi, tip_partials=None):
        output_shape = jax.ShapeDtypeStruct((self.pattern_count,), self.dtype)
        if tip_partials is None:
            return jax.pure_callback(self._host_simple_fwd, output_shape, edge_lengths, Q, pi)
        else:
            def _dynamic_wrapper(el, q, p, tips):
                self._update_tips_host(tips)
                return self._host_simple_fwd(el, q, p)
            return jax.pure_callback(_dynamic_wrapper, output_shape, edge_lengths, Q, pi, tip_partials)

    # -----------------------------------------------------------------------
    # FORWARD PASS
    # -----------------------------------------------------------------------
    def _log_likelihood_fwd(self, edge_lengths, Q, pi, tip_partials=None):
        shape_ll = jax.ShapeDtypeStruct((self.pattern_count,), self.dtype)
        
        # Use standard dtype for residuals (float32)
        res_dtype = self.dtype 
        
        shape_all_partials = jax.ShapeDtypeStruct((self.node_count, self.pattern_count, self.state_count), res_dtype)
        shape_all_scales = jax.ShapeDtypeStruct((self.node_count, self.pattern_count), res_dtype)
        shape_V = jax.ShapeDtypeStruct((self.state_count, self.state_count), res_dtype)
        shape_InvV = jax.ShapeDtypeStruct((self.state_count, self.state_count), res_dtype)
        shape_Evals = jax.ShapeDtypeStruct((self.state_count,), res_dtype)

        if tip_partials is None:
            host_fn = self._host_fwd_full
            args = (edge_lengths, Q, pi)
        else:
            host_fn = self._host_fwd_full_dynamic
            args = (edge_lengths, Q, pi, tip_partials)

        site_ll, all_partials, all_scales, V, V_inv, evals = jax.pure_callback(
            host_fn,
            (shape_ll, shape_all_partials, shape_all_scales, shape_V, shape_InvV, shape_Evals),
            *args
        )
        
        residuals = (all_partials, all_scales, V, V_inv, evals, edge_lengths, pi)
        return site_ll, residuals

    # -----------------------------------------------------------------------
    # BACKWARD PASS
    # -----------------------------------------------------------------------
    def _log_likelihood_bwd(self, residuals, grad_output):
        # All inputs are in self.dtype (e.g. float32)
        all_partials, all_scales, V, V_inv, evals, edge_lengths, pi = residuals
        
        # 1. Initialize Adjoints at Root
        root_p = all_partials[self.root_index]
        denom = jnp.dot(root_p, pi) 
        
        adj_root = (grad_output / denom)[:, None] * pi
        
        adjoints = jnp.zeros_like(all_partials)
        adjoints = adjoints.at[self.root_index].set(adj_root)
        
        grad_pi = jnp.sum((grad_output / denom)[:, None] * root_p, axis=0)
        grad_Q_accum = jnp.zeros_like(V)
        
        # 2. Traversal
        for op in reversed(self.operations):
            dest = op['dest']
            child1 = op['child1']
            child2 = op['child2']
            
            adj_dest = adjoints[dest]
            
            L1 = all_partials[child1]
            L2 = all_partials[child2]
            t1 = edge_lengths[child1]
            t2 = edge_lengths[child2]
            
            P1 = self._compute_P(t1, V, V_inv, evals)
            P2 = self._compute_P(t2, V, V_inv, evals)

            # Scale Correction (Fixed Logic)
            log_C = all_scales[child1] + all_scales[child2] - all_scales[dest]
            C = jnp.exp(log_C)[:, None] 
            
            T1 = jnp.dot(L1, P1.T) 
            T2 = jnp.dot(L2, P2.T)
            
            # Branch 1
            adj_T1 = adj_dest * T2 * C
            adj_child1 = jnp.dot(adj_T1, P1)
            adjoints = adjoints.at[child1].set(adj_child1)
            
            grad_P1 = jnp.einsum('pi,pj->ij', adj_T1, L1)
            grad_Q_accum += self._van_loan_grad(grad_P1, t1, V, V_inv, evals)
            
            # Branch 2
            adj_T2 = adj_dest * T1 * C
            adj_child2 = jnp.dot(adj_T2, P2)
            adjoints = adjoints.at[child2].set(adj_child2)
            
            grad_P2 = jnp.einsum('pi,pj->ij', adj_T2, L2)
            grad_Q_accum += self._van_loan_grad(grad_P2, t2, V, V_inv, evals)

        return None, grad_Q_accum, grad_pi, None

    def _compute_P(self, t, V, V_inv, evals):
        exp_lam_t = jnp.exp(evals * t)
        return (V * exp_lam_t) @ V_inv

    def _van_loan_grad(self, grad_P, t, V, V_inv, evals):
        # General Van Loan: V^-T @ [ (V^T @ grad_P @ V^-T) * G ] @ V^T
        V_inv_T = V_inv.T
        V_T = V.T
        
        term = V_T @ grad_P @ V_inv_T
        
        exp_lam_t = jnp.exp(evals * t)
        G_diag = t * exp_lam_t
        
        diff_exp = exp_lam_t[:, None] - exp_lam_t[None, :]
        diff_lam = evals[:, None] - evals[None, :]
        mask = jnp.abs(diff_lam) < 1e-9
        safe_denom = jnp.where(mask, 1.0, diff_lam)
        G_off = diff_exp / safe_denom
        G = jnp.where(mask, jnp.diag(G_diag), G_off)
        
        term = term * G
        return V_inv_T @ term @ V_T

    # -----------------------------------------------------------------------
    # HOST CALLBACKS
    # -----------------------------------------------------------------------
    def _host_simple_fwd(self, edge_lengths, Q, pi):
        self._update_beagle_state(edge_lengths, Q, pi)
        pi_f64 = np.array(pi).astype(np.float64)
        return self.backend.calculate_site_log_likelihoods(self.root_index, pi=pi_f64).astype(self.dtype)

    def _update_beagle_state(self, edge_lengths_np, Q_np, pi_np):
        edge_lengths_f64 = np.array(edge_lengths_np).astype(np.float64)
        Q_f64 = np.array(Q_np).astype(np.float64)
        pi_f64 = np.array(pi_np).astype(np.float64)
        self.backend.set_model_matrix(Q_f64, pi_f64)
        self.backend.update_transition_matrices(edge_lengths_f64, self.prob_indices)
        self.backend.update_partials(self.c_operations, operation_count=self.operation_count)

    def _host_fwd_full(self, edge_lengths, Q, pi):
        self._update_beagle_state(edge_lengths, Q, pi)
        
        pi_f64 = np.array(pi).astype(np.float64)
        site_ll = self.backend.calculate_site_log_likelihoods(self.root_index, pi=pi_f64)
        all_partials, all_scales = self.backend.get_all_partials_and_scales(self.node_count)
        
        Q_f64 = np.array(Q).astype(np.float64)
        evals, V = np.linalg.eig(Q_f64)
        V_inv = np.linalg.inv(V)
        
        # Cast everything to self.dtype (float32) before returning to JAX
        return (site_ll.astype(self.dtype), 
                all_partials.astype(self.dtype), 
                all_scales.astype(self.dtype),
                V.astype(self.dtype), 
                V_inv.astype(self.dtype), 
                evals.astype(self.dtype))

    def _update_tips_host(self, tip_partials_np):
        tip_partials_f64 = np.array(tip_partials_np).astype(np.float64)
        for i in range(len(tip_partials_f64)):
            self.backend.set_tip_partials({i: tip_partials_f64[i]})

    def _host_fwd_full_dynamic(self, edge_lengths, Q, pi, tip_partials):
        self._update_tips_host(tip_partials)
        return self._host_fwd_full(edge_lengths, Q, pi)
