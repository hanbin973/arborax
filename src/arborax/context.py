import os

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import expm

from .beagle_cffi import BeagleLikelihoodCalculator


def get_jax_dtype():
    if jax.config.read("jax_enable_x64"):
        return jnp.float64, np.float64
    return jnp.float32, np.float32


class ArboraxContext:
    def __init__(self, tip_count, operations, pattern_count=1, use_gpu=None):
        if use_gpu is None:
            use_gpu = os.environ.get("ARBORAX_USE_GPU", "0") == "1"
        self.calc = BeagleLikelihoodCalculator(
            tip_count, pattern_count=pattern_count, use_gpu=use_gpu
        )
        self.operations = operations
        self.root_index = operations[-1]["dest"]
        self.pattern_count = pattern_count
        self.num_nodes = self.calc.node_count
        self.state_count = self.calc.state_count
        self.Q = None
        self.pi = None
        self.node_parent_map = {}
        for op in operations:
            dest = op["dest"]
            c1 = op["child1"]
            c2 = op["child2"]
            self.node_parent_map[c1] = (dest, c2)
            self.node_parent_map[c2] = (dest, c1)
        self.root_preorder_index = self.calc.preorder_buffer_index(self.root_index)
        self._preorder_operations_ptr, self._preorder_operation_count = (
            self._prepare_preorder_operations()
        )

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other

    def bind_data(self, tip_partials_dict, Q, p_root):
        self.calc.set_tip_partials(tip_partials_dict)
        self.Q = Q
        self.pi = p_root

    def _prepare_preorder_operations(self):
        op_dicts = self._build_preorder_operation_dicts()
        if not op_dicts:
            return None, 0
        ptr = self.calc.prepare_operations(op_dicts)
        return ptr, len(op_dicts)

    def _build_preorder_operation_dicts(self):
        parent_children = {}
        for op in self.operations:
            parent_children[op["dest"]] = (op["child1"], op["child2"])

        preorder_ops = []
        stack = [self.root_index]
        visited = set()

        while stack:
            parent = stack.pop()
            if parent in visited:
                continue
            visited.add(parent)
            children = parent_children.get(parent)
            if not children:
                continue
            child1, child2 = children
            parent_pre_idx = self.calc.preorder_buffer_index(parent)
            preorder_ops.append(self._make_preorder_op(parent_pre_idx, child1, child2))
            preorder_ops.append(self._make_preorder_op(parent_pre_idx, child2, child1))

            if child1 in parent_children:
                stack.append(child1)
            if child2 in parent_children:
                stack.append(child2)

        return preorder_ops

    def _make_preorder_op(self, parent_pre_idx, target_child, sibling_child):
        return {
            "dest": self.calc.preorder_buffer_index(target_child),
            "destScaleWrite": -1,
            "destScaleRead": -1,
            "child1": parent_pre_idx,
            "child1Matrix": target_child,
            "child2": sibling_child,
            "child2Matrix": sibling_child,
        }

    def likelihood(self, edge_lengths):
        if self.Q is None:
            raise RuntimeError("Data not bound")
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

    all_partials, all_scales = context.calc.get_all_partials_and_scales(
        context.num_nodes
    )

    return (
        np.array(logl, dtype=np_dtype),
        np.array(all_partials, dtype=np_dtype),
        np.array(all_scales, dtype=np_dtype),
    )


def _beagle_bwd_callback(grad_logl, partials, scales, P_stack, pi, context):
    _, np_dtype = get_jax_dtype()
    partials = np.array(partials, dtype=np.float64)
    scales = np.array(scales, dtype=np.float64)
    P_stack = np.array(P_stack, dtype=np.float64)
    pi = np.array(pi, dtype=np.float64)
    grad_logl = np.array(grad_logl, dtype=np.float64)

    root_idx = context.root_index
    root_partials = partials[root_idx]
    site_probs = np.dot(root_partials, pi)
    site_weights = np.zeros_like(site_probs)
    valid = site_probs != 0
    site_weights[valid] = grad_logl[valid] / site_probs[valid]
    grad_pi = np.sum(site_weights[:, None] * root_partials, axis=0)

    calc = context.calc
    if context._preorder_operation_count > 0:
        root_pre = np.broadcast_to(pi, (context.pattern_count, context.state_count))
        calc.set_partials(context.root_preorder_index, root_pre)
        calc.update_pre_partials(
            context._preorder_operations_ptr,
            operation_count=context._preorder_operation_count,
        )
        preorder_partials = calc.get_all_pre_partials(context.num_nodes)
    else:
        preorder_partials = np.zeros_like(partials)
        preorder_partials[root_idx] = np.broadcast_to(
            pi, (context.pattern_count, context.state_count)
        )

    grad_P = np.zeros_like(P_stack)

    num_edges = len(P_stack)
    sibling_msgs = np.zeros_like(partials)
    for node_idx in range(num_edges):
        sibling_msgs[node_idx] = np.dot(partials[node_idx], P_stack[node_idx].T)

    for edge_idx in range(num_edges):
        child = edge_idx
        if child == root_idx or child not in context.node_parent_map:
            continue
        parent, sibling = context.node_parent_map[child]
        adj_parent = site_weights[:, None] * preorder_partials[parent]
        log_C = scales[child] + scales[sibling] - scales[parent]
        C = np.exp(log_C)[:, None]
        msg_sibling = sibling_msgs[sibling]
        adj_msg = adj_parent * msg_sibling * C
        grad_P[edge_idx] = np.einsum("pi,pj->ij", adj_msg, partials[child])

    return (np.array(grad_P, dtype=np_dtype), np.array(grad_pi, dtype=np_dtype))


@jax.custom_vjp
def _beagle_likelihood_op(P_stack, pi, context):
    j_dtype, _ = get_jax_dtype()
    result_shape = jax.ShapeDtypeStruct((context.pattern_count,), j_dtype)

    def _callback(P, p):
        return _beagle_fwd_callback(P, p, context)[0]

    return jax.pure_callback(_callback, result_shape, P_stack, pi)


def _beagle_fwd(P_stack, pi, context):
    j_dtype, _ = get_jax_dtype()
    shape_partials = jax.ShapeDtypeStruct(
        (context.num_nodes, context.pattern_count, context.state_count), j_dtype
    )
    shape_scales = jax.ShapeDtypeStruct(
        (context.num_nodes, context.pattern_count), j_dtype
    )
    logl, partials, scales = jax.pure_callback(
        lambda P, p: _beagle_fwd_callback(P, p, context),
        (
            jax.ShapeDtypeStruct((context.pattern_count,), j_dtype),
            shape_partials,
            shape_scales,
        ),
        P_stack,
        pi,
    )
    return logl, (partials, scales, P_stack, pi, context)


def _beagle_bwd(res, g):
    j_dtype, _ = get_jax_dtype()
    partials, scales, P_stack, pi, context = res
    d_P, d_pi = jax.pure_callback(
        lambda g_in, parts, sc, P, p: _beagle_bwd_callback(
            g_in, parts, sc, P, p, context
        ),
        (
            jax.ShapeDtypeStruct(P_stack.shape, j_dtype),
            jax.ShapeDtypeStruct(pi.shape, j_dtype),
        ),
        g,
        partials,
        scales,
        P_stack,
        pi,
    )
    return (d_P, d_pi, None)


_beagle_likelihood_op.defvjp(_beagle_fwd, _beagle_bwd)
