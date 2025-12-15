import os
import sys

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

        self.tip_count = tip_count
        self.pattern_count = pattern_count
        self.use_gpu = use_gpu

        self.operations = operations
        self.root_index = operations[-1]["dest"]
        self.num_nodes = 2 * tip_count - 1
        self.state_count = None

        self.Q = None
        self.pi = None
        self._tip_partials = None

        self.node_parent_map = {}
        for op in operations:
            dest = op["dest"]
            c1 = op["child1"]
            c2 = op["child2"]
            self.node_parent_map[c1] = (dest, c2)
            self.node_parent_map[c2] = (dest, c1)
        self._calc_pool = []

    def close(self):
        while self._calc_pool:
            calc = self._calc_pool.pop()
            calc.close()

    def __del__(self):
        if getattr(sys, "is_finalizing", lambda: False)():
            return
        self.close()

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other

    def bind_data(self, tip_partials_dict, Q, p_root):
        if not tip_partials_dict:
            raise ValueError("tip_partials_dict must not be empty")

        tip_arrays: dict[int, np.ndarray] = {}
        inferred_state_count = None
        for idx, partials in tip_partials_dict.items():
            arr = np.asarray(partials, dtype=np.float64)
            if arr.shape[0] != self.pattern_count:
                raise ValueError(
                    f"Tip {idx} has {arr.shape[0]} patterns, "
                    f"expected {self.pattern_count}"
                )
            if inferred_state_count is None:
                inferred_state_count = arr.shape[1]
            elif arr.shape[1] != inferred_state_count:
                raise ValueError("All tips must share the same state dimension")
            tip_arrays[idx] = arr

        if len(tip_arrays) != self.tip_count:
            raise ValueError(
                f"Expected partials for {self.tip_count} tips, "
                f"received {len(tip_arrays)}"
            )

        self._tip_partials = tip_arrays
        self.state_count = inferred_state_count
        self.Q = Q
        self.pi = p_root

    def _ensure_bound(self):
        if self._tip_partials is None or self.state_count is None:
            raise RuntimeError("Tip data has not been bound. Call bind_data first.")

    def _build_preorder_operation_dicts(self, calc):
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
            parent_pre_idx = calc.preorder_buffer_index(parent)
            preorder_ops.append(
                self._make_preorder_op(calc, parent_pre_idx, child1, child2)
            )
            preorder_ops.append(
                self._make_preorder_op(calc, parent_pre_idx, child2, child1)
            )

            if child1 in parent_children:
                stack.append(child1)
            if child2 in parent_children:
                stack.append(child2)

        return preorder_ops

    def _make_preorder_op(self, calc, parent_pre_idx, target_child, sibling_child):
        return {
            "dest": calc.preorder_buffer_index(target_child),
            "destScaleWrite": -1,
            "destScaleRead": -1,
            "child1": parent_pre_idx,
            "child1Matrix": target_child,
            "child2": sibling_child,
            "child2Matrix": sibling_child,
        }

    def _create_calculator(self):
        self._ensure_bound()
        return BeagleLikelihoodCalculator(
            self.tip_count,
            state_count=self.state_count,
            pattern_count=self.pattern_count,
            use_gpu=self.use_gpu,
        )

    def _acquire_calculator(self):
        if self._calc_pool:
            return self._calc_pool.pop()
        return self._create_calculator()

    def _release_calculator(self, calc):
        keep_alive = getattr(calc, "_keep_alive", None)
        if keep_alive is not None:
            keep_alive.clear()
        self._calc_pool.append(calc)

    def _run_beagle(self, P_np, pi_np):
        calc = self._acquire_calculator()
        try:
            calc.set_tip_partials(self._tip_partials)
            calc.set_transition_matrices(P_np, len(P_np))
            ops_ptr = calc.prepare_operations(self.operations)
            calc.update_partials(ops_ptr, operation_count=len(self.operations))
            logl = calc.calculate_site_log_likelihoods(self.root_index, pi_np)
            all_partials, all_scales = calc.get_all_partials_and_scales(self.num_nodes)
            preorder = self._compute_preorder_partials(calc, pi_np)
            return logl, all_partials, all_scales, preorder
        finally:
            self._release_calculator(calc)

    def _compute_preorder_partials(self, calc, pi_np):
        preorder_ops = self._build_preorder_operation_dicts(calc)
        shape = (self.num_nodes, self.pattern_count, self.state_count)
        broadcast_root = np.broadcast_to(pi_np, (self.pattern_count, self.state_count))

        if not preorder_ops:
            upstream = np.zeros(shape, dtype=np.float64)
            upstream[self.root_index] = broadcast_root
            return upstream

        root_pre_idx = calc.preorder_buffer_index(self.root_index)
        calc.set_partials(root_pre_idx, broadcast_root)
        calc.update_pre_partials(preorder_ops)
        return calc.get_all_pre_partials(self.num_nodes)

    def likelihood(self, edge_lengths):
        if self.Q is None:
            raise RuntimeError("Data not bound")
        return self.likelihood_functional(self.Q, self.pi, edge_lengths)

    def likelihood_functional(self, Q, pi, edge_lengths):
        self._ensure_bound()
        P_stack = jax.vmap(lambda t: expm(Q * t))(edge_lengths)
        return _beagle_likelihood_op(P_stack, pi, self)


jax.tree_util.register_pytree_node(ArboraxContext, lambda x: ((), x), lambda x, _: x)


def _beagle_fwd_callback(P_stack, pi, context):
    _, np_dtype = get_jax_dtype()
    P_np = np.array(P_stack, dtype=np.float64)
    pi_np = np.array(pi, dtype=np.float64)

    logl, partials, scales, preorder = context._run_beagle(P_np, pi_np)

    return (
        np.array(logl, dtype=np_dtype),
        np.array(partials, dtype=np_dtype),
        np.array(scales, dtype=np_dtype),
        np.array(preorder, dtype=np_dtype),
    )


def _beagle_bwd_callback(
    grad_logl, partials, scales, preorder_partials, P_stack, pi, context
):
    _, np_dtype = get_jax_dtype()
    partials = np.array(partials, dtype=np.float64)
    scales = np.array(scales, dtype=np.float64)
    preorder_partials = np.array(preorder_partials, dtype=np.float64)
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

    return jax.pure_callback(
        _callback, result_shape, P_stack, pi, vmap_method="sequential"
    )


def _beagle_fwd(P_stack, pi, context):
    j_dtype, _ = get_jax_dtype()
    shape_partials = jax.ShapeDtypeStruct(
        (context.num_nodes, context.pattern_count, context.state_count), j_dtype
    )
    shape_scales = jax.ShapeDtypeStruct(
        (context.num_nodes, context.pattern_count), j_dtype
    )
    shape_preorder = jax.ShapeDtypeStruct(
        (context.num_nodes, context.pattern_count, context.state_count), j_dtype
    )
    logl, partials, scales, preorder = jax.pure_callback(
        lambda P, p: _beagle_fwd_callback(P, p, context),
        (
            jax.ShapeDtypeStruct((context.pattern_count,), j_dtype),
            shape_partials,
            shape_scales,
            shape_preorder,
        ),
        P_stack,
        pi,
        vmap_method="sequential",
    )
    return logl, (partials, scales, preorder, P_stack, pi, context)


def _beagle_bwd(res, g):
    j_dtype, _ = get_jax_dtype()
    partials, scales, preorder_partials, P_stack, pi, context = res
    d_P, d_pi = jax.pure_callback(
        lambda g_in, parts, sc, pre, P, p: _beagle_bwd_callback(
            g_in, parts, sc, pre, P, p, context
        ),
        (
            jax.ShapeDtypeStruct(P_stack.shape, j_dtype),
            jax.ShapeDtypeStruct(pi.shape, j_dtype),
        ),
        g,
        partials,
        scales,
        preorder_partials,
        P_stack,
        pi,
        vmap_method="sequential",
    )
    return (d_P, d_pi, None)


_beagle_likelihood_op.defvjp(_beagle_fwd, _beagle_bwd)
