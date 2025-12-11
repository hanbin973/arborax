import numpy as np
from jax.scipy.linalg import expm

from arborax.jax_ops import ArboraxContext
from tests.test_reference import _generate_problem


def _manual_preorder_partials(
    partials, scales, P_stack, pi_vec, operations, root_index
):
    num_nodes, pattern_count, state_count = partials.shape
    upstream = np.zeros_like(partials, dtype=np.float64)
    pi_broadcast = np.broadcast_to(pi_vec, (pattern_count, state_count))
    upstream[root_index] = pi_broadcast

    for op in reversed(operations):
        parent = op["dest"]
        child1 = op["child1"]
        child2 = op["child2"]

        log_C = scales[child1] + scales[child2] - scales[parent]
        C = np.exp(log_C)[:, None]

        msg1 = np.dot(partials[child1], P_stack[child1].T)
        msg2 = np.dot(partials[child2], P_stack[child2].T)

        ctx1 = upstream[parent] * msg2 * C
        ctx2 = upstream[parent] * msg1 * C

        upstream[child1] = np.dot(ctx1, P_stack[child1])
        upstream[child2] = np.dot(ctx2, P_stack[child2])

    return upstream


def _build_preorder_operations(calc, operations):
    parent_to_children = {op["dest"]: (op["child1"], op["child2"]) for op in operations}
    pre_ops = []
    stack = [operations[-1]["dest"]]
    visited = set()

    while stack:
        parent = stack.pop()
        if parent in visited:
            continue
        visited.add(parent)

        children = parent_to_children.get(parent)
        if not children:
            continue

        child1, child2 = children
        pre_ops.append(
            {
                "dest": calc.preorder_buffer_index(child1),
                "destScaleWrite": -1,
                "destScaleRead": -1,
                "child1": calc.preorder_buffer_index(parent),
                "child1Matrix": child1,
                "child2": child2,
                "child2Matrix": child2,
            }
        )
        pre_ops.append(
            {
                "dest": calc.preorder_buffer_index(child2),
                "destScaleWrite": -1,
                "destScaleRead": -1,
                "child1": calc.preorder_buffer_index(parent),
                "child1Matrix": child2,
                "child2": child1,
                "child2Matrix": child1,
            }
        )

        if child1 in parent_to_children:
            stack.append(child1)
        if child2 in parent_to_children:
            stack.append(child2)

    return pre_ops


def _transition_stack(Q, edge_lengths):
    Q_np = np.array(Q, dtype=np.float64)
    lengths = np.array(edge_lengths, dtype=np.float64)
    num_nodes = lengths.shape[0]
    P_stack = np.zeros((num_nodes, Q_np.shape[0], Q_np.shape[1]), dtype=np.float64)
    for idx in range(num_nodes):
        P_stack[idx] = np.array(expm(Q_np * lengths[idx]), dtype=np.float64)
    return P_stack


def test_preorder_partials_match_beagle(seed):
    problem = _generate_problem(seed)
    context: ArboraxContext = problem["context"]

    Q = np.array(problem["Q"], dtype=np.float64)
    pi = np.array(problem["pi"], dtype=np.float64)
    edge_lengths = np.array(problem["edge_lengths"], dtype=np.float64)

    calc = context.calc
    P_stack = _transition_stack(Q, edge_lengths)

    calc.set_model_matrix(Q, pi)
    calc.set_transition_matrices(P_stack, len(P_stack))
    calc.update_partials(context.operations, len(context.operations))

    partials, scales = calc.get_all_partials_and_scales(context.num_nodes)

    manual_pre = _manual_preorder_partials(
        partials, scales, P_stack, pi, context.operations, context.root_index
    )

    root_pre_idx = calc.preorder_buffer_index(context.root_index)
    root_pre = np.broadcast_to(pi, (context.pattern_count, context.state_count))
    calc.set_partials(root_pre_idx, root_pre)

    preorder_ops = _build_preorder_operations(calc, context.operations)
    calc.update_pre_partials(preorder_ops, len(preorder_ops))
    beagle_pre = calc.get_all_pre_partials(context.num_nodes)

    np.testing.assert_allclose(
        beagle_pre,
        manual_pre,
        atol=1e-6,
        rtol=1e-6,
        err_msg="Pre-order partials mismatch between BEAGLE and reference traversal",
    )
