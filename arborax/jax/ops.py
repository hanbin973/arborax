import jax
import jax.numpy as jnp
import numpy as np
import os
from jax.scipy.linalg import expm
from ..lowlevel.beagle_cffi import BeagleLikelihoodCalculator

def get_jax_dtype():
    if jax.config.read("jax_enable_x64"): return jnp.float64, np.float64
    return jnp.float32, np.float32

class ArboraxContext:
    def __init__(self, tip_count, operations, pattern_count=1, use_gpu=None):
        if use_gpu is None: use_gpu = os.environ.get("ARBORAX_USE_GPU", "0") == "1"
        self.calc = BeagleLikelihoodCalculator(tip_count, pattern_count=pattern_count, use_gpu=use_gpu)
        self.operations = operations
        self.root_index = operations[-1]['dest']
        self.pattern_count = pattern_count
        self.num_nodes = self.calc.node_count
        self.Q = None; self.pi = None
        self.node_parent_map = {} 
        for op in operations:
            dest = op['dest']; c1 = op['child1']; c2 = op['child2']
            self.node_parent_map[c1] = (dest, c2)
            self.node_parent_map[c2] = (dest, c1)

    def __hash__(self): return id(self)
    def __eq__(self, other): return self is other
    def bind_data(self, tip_partials_dict, Q, p_root):
        self.calc.set_tip_partials(tip_partials_dict)
        self.Q = Q; self.pi = p_root
    def likelihood(self, edge_lengths):
        if self.Q is None: raise RuntimeError("Data not bound")
        return self.likelihood_functional(self.Q, self.pi, edge_lengths)
    def likelihood_functional(self, Q, pi, edge_lengths):
        P_stack = jax.vmap(lambda t: expm(Q * t))(edge_lengths)
        return _beagle_likelihood_op(P_stack, pi, self)

jax.tree_util.register_pytree_node(ArboraxContext, lambda x: ((), x), lambda x, _: x)

def _beagle_fwd_callback(P_stack, pi, context):
    _, np_dtype = get_jax_dtype()
    P_np = np.array(P_stack, dtype=np.float64)
    pi_np = np.array(pi, dtype=np.float64)
    
    context.calc.set_transition_matrices(P_np, len(P_np))
    context.calc.update_partials(context.operations)
    logl = context.calc.calculate_site_log_likelihoods(context.root_index, pi_np)
    
    all_partials = context.calc.get_all_partials(context.num_nodes)
    all_scales = context.calc.get_all_scale_factors(context.num_nodes)
    
    return (
        np.array(logl, dtype=np_dtype),
        np.array(all_partials, dtype=np_dtype),
        np.array(all_scales, dtype=np_dtype)
    )

def _compute_ancestral_partials_np(partials, P_stack, pi, operations, num_nodes):
    n_sites = partials.shape[1]; n_states = 4
    L_up = np.zeros((num_nodes, n_sites, n_states))
    L_up[operations[-1]['dest']] = pi
    
    for op in reversed(operations):
        parent = op['dest']; c1 = op['child1']; c2 = op['child2']
        P1 = P_stack[c1]; P2 = P_stack[c2]
        
        msg_c2 = np.dot(partials[c2], P2.T)
        msg_c1 = np.dot(partials[c1], P1.T)
        ctx_c1 = L_up[parent] * msg_c2
        ctx_c2 = L_up[parent] * msg_c1
        L_up[c1] = np.dot(ctx_c1, P1)
        L_up[c2] = np.dot(ctx_c2, P2)
    return L_up

def _beagle_bwd_callback(grad_logl, partials, scales, P_stack, pi, context):
    _, np_dtype = get_jax_dtype()
    partials = np.array(partials, dtype=np.float64)
    scales = np.array(scales, dtype=np.float64)
    P_stack = np.array(P_stack, dtype=np.float64)
    pi = np.array(pi, dtype=np.float64)
    grad_logl = np.array(grad_logl, dtype=np.float64)

    # FIX: Unscale by multiplying by exp(-log_scale)
    partials_raw = partials * np.exp(-scales)[:, :, None]

    L_up = _compute_ancestral_partials_np(partials_raw, P_stack, pi, context.operations, context.num_nodes)
    
    num_edges = len(P_stack)
    grad_P = np.zeros_like(P_stack)
    
    root_down = partials_raw[context.root_index]
    site_L = np.dot(root_down, pi)
    
    site_weights = np.zeros_like(site_L)
    valid = site_L != 0
    site_weights[valid] = grad_logl[valid] / site_L[valid]
    site_weights = site_weights[:, None, None]
    
    for edge_idx in range(num_edges):
        child = edge_idx
        if child == context.root_index or child not in context.node_parent_map: continue
        parent = context.node_parent_map[child][0]
        
        up = L_up[parent]
        down = partials_raw[child]
        outer = up[:, :, None] * down[:, None, :]
        grad_P[edge_idx] = np.sum(outer * site_weights, axis=0)
        
    grad_pi = np.sum(root_down * site_weights[:, 0], axis=0)
    
    return (np.array(grad_P, dtype=np_dtype), np.array(grad_pi, dtype=np_dtype))

@jax.custom_vjp
def _beagle_likelihood_op(P_stack, pi, context):
    j_dtype, _ = get_jax_dtype()
    result_shape = jax.ShapeDtypeStruct((context.pattern_count,), j_dtype)
    def _callback(P, p): return _beagle_fwd_callback(P, p, context)[0]
    return jax.pure_callback(_callback, result_shape, P_stack, pi)

def _beagle_fwd(P_stack, pi, context):
    j_dtype, _ = get_jax_dtype()
    logl, partials, scales = jax.pure_callback(
        lambda P, p: _beagle_fwd_callback(P, p, context),
        (
            jax.ShapeDtypeStruct((context.pattern_count,), j_dtype),
            jax.ShapeDtypeStruct((context.num_nodes, context.pattern_count, 4), j_dtype),
            jax.ShapeDtypeStruct((context.num_nodes, context.pattern_count), j_dtype)
        ),
        P_stack, pi
    )
    return logl, (partials, scales, P_stack, pi, context)

def _beagle_bwd(res, g):
    j_dtype, _ = get_jax_dtype()
    partials, scales, P_stack, pi, context = res
    d_P, d_pi = jax.pure_callback(
        lambda g_in, parts, sc, P, p: _beagle_bwd_callback(g_in, parts, sc, P, p, context),
        (
            jax.ShapeDtypeStruct(P_stack.shape, j_dtype),
            jax.ShapeDtypeStruct(pi.shape, j_dtype)
        ),
        g, partials, scales, P_stack, pi
    )
    return (d_P, d_pi, None)

_beagle_likelihood_op.defvjp(_beagle_fwd, _beagle_bwd)
