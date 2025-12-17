from collections import defaultdict
from typing import Sequence, Union, Tuple

import numpy as np

from .beagle_cffi import BeagleLikelihoodCalculator
from .context import ArboraxContext

__version__ = "0.1.0"


def create_arborax_context(
    edge_list: Sequence[Tuple[int, int]],
    branch_lengths: Sequence[float],
    partials_shape: Tuple[int, int, int]
) -> Tuple[ArboraxContext, np.ndarray]:
    """
    Pre-initializes the ArboraxContext and maps branch lengths to the correct node order.

    Args:
        edge_list: Sequence of (parent, child) indices.
        branch_lengths: Sequence of branch lengths corresponding to edge_list.
        partials_shape: Tuple of (num_tips, num_sites, num_states).

    Returns:
        context: An initialized ArboraxContext with operations scheduled.
        edge_lengths_arr: A numpy array of branch lengths indexed by child node ID.
    """
    # 1. Unpack Dimensions
    # Shape is (tips, sites, states) -> (tip_count, pattern_count, states)
    tip_count, pattern_count, _ = partials_shape

    # 2. Validate inputs (NumPy/Python side)
    edge_array = np.asarray(edge_list, dtype=np.int32)
    branch_lengths_arr = np.asarray(branch_lengths, dtype=np.float64)

    if edge_array.ndim != 2 or edge_array.shape[1] != 2:
        raise ValueError("edge_list must be shape (E, 2)")

    expected_edge_count = 2 * tip_count - 2
    if len(edge_array) != expected_edge_count:
        raise ValueError(f"Expected {expected_edge_count} edges, got {len(edge_array)}")

    if len(edge_array) != len(branch_lengths_arr):
        raise ValueError("branch_lengths length must match edge_list length")

    # 3. Build Tree Topology (Static Analysis)
    # This generates the traversal order (operations) and total node count
    tree_info = _build_tree_from_edges(edge_array, tip_count)
    operations = tree_info["operations"]
    node_count = tree_info["node_count"]

    # 4. Map Branch Lengths
    # Maps the input branch lengths (ordered by edge_list) to the internal
    # array format (ordered by node index) required by the C++ engine.
    edge_lengths_vec = _edge_lengths_from_pairs(
        node_count, edge_array, branch_lengths_arr
    )
    edge_lengths_arr = np.asarray(edge_lengths_vec, dtype=np.float64)

    # 5. Initialize Context
    context = ArboraxContext(
        tip_count=tip_count,
        operations=operations,
        pattern_count=pattern_count,
    )

    return edge_lengths_arr, context

def loglik(
    tip_partials: Union[np.ndarray, dict[int, np.ndarray]],
    edge_list: Sequence[tuple[int, int]],
    branch_lengths: Sequence[float],
    Q: np.ndarray,
    pi: np.ndarray,
) -> np.ndarray:
    tip_partials_arr = _normalize_tip_partials(tip_partials)
    tip_count, pattern_count, _ = tip_partials_arr.shape

    edge_array = np.asarray(edge_list, dtype=np.int32)
    if edge_array.ndim != 2 or edge_array.shape[1] != 2:
        raise ValueError("edge_list must be shape (E, 2) of parent/child indices")

    branch_lengths_arr = np.asarray(branch_lengths, dtype=np.float64)
    if branch_lengths_arr.ndim != 1:
        raise ValueError("branch_lengths must be a 1-D array")

    if len(edge_array) != len(branch_lengths_arr):
        raise ValueError("branch_lengths must match the number of edges")

    expected_edge_count = 2 * tip_count - 2
    if len(edge_array) != expected_edge_count:
        raise ValueError(
            f"expected {expected_edge_count} edges for {tip_count} tips, "
            f"got {len(edge_array)}"
        )

    tree_info = _build_tree_from_edges(edge_array, tip_count)
    edge_lengths_vec = _edge_lengths_from_pairs(
        tree_info["node_count"], edge_array, branch_lengths_arr
    )
    operations = tree_info["operations"]

    tip_dict = {idx: tip_partials_arr[idx] for idx in range(tip_count)}

    context = ArboraxContext(
        tip_count=tip_count,
        operations=operations,
        pattern_count=pattern_count,
    )
    context.bind_data(tip_dict, Q, pi)
    edge_lengths_arr = np.asarray(edge_lengths_vec, dtype=Q.dtype)
    return context.likelihood_functional(Q, pi, edge_lengths_arr)

def _normalize_tip_partials(
    tip_partials: Union[np.ndarray, dict[int, np.ndarray]],
) -> np.ndarray:
    if isinstance(tip_partials, dict):
        if not tip_partials:
            raise ValueError("tip_partials dictionary must not be empty")
        keys = sorted(tip_partials.keys())
        expected = list(range(len(keys)))
        if keys != expected:
            raise ValueError(
                "tip_partials dict must contain consecutive tip indices starting at 0"
            )
        arrays = [np.asarray(tip_partials[k]) for k in keys]
        return np.stack(arrays, axis=0)
    arr = np.asarray(tip_partials)
    if arr.ndim != 3:
        raise ValueError("tip_partials array must have shape (tips, patterns, states)")
    return arr


def _build_tree_from_edges(edge_array: np.ndarray, tip_count: int) -> dict[str, object]:
    parent_to_children: dict[int, list[int]] = defaultdict(list)
    children = set()
    nodes = set()
    for parent, child in edge_array:
        parent_to_children[parent].append(child)
        children.add(child)
        nodes.add(parent)
        nodes.add(child)

    for parent, kids in parent_to_children.items():
        if len(kids) != 2:
            raise ValueError(f"Node {parent} must have exactly two children")

    node_count = 2 * tip_count - 1
    expected_nodes = set(range(node_count))
    if nodes != expected_nodes:
        missing = expected_nodes - nodes
        extra = nodes - expected_nodes
        raise ValueError(
            f"Tree nodes must be labeled 0..{node_count-1}. "
            f"Missing: {sorted(missing)}, extra: {sorted(extra)}"
        )

    tips = [node for node in expected_nodes if node not in parent_to_children]
    if len(tips) != tip_count:
        raise ValueError("Number of tip nodes does not match tip partials")

    roots = [node for node in parent_to_children if node not in children]
    if len(roots) != 1:
        raise ValueError("Tree must have exactly one root")
    root = roots[0]

    operations: list[dict[str, int]] = []

    def emit(node: int):
        kids = parent_to_children.get(node)
        if not kids:
            return
        child1, child2 = kids
        emit(child1)
        emit(child2)
        operations.append({"dest": node, "child1": child1, "child2": child2})

    emit(root)

    return {"operations": operations, "node_count": node_count}


def _edge_lengths_from_pairs(
    node_count: int, edges: np.ndarray, lengths: np.ndarray
) -> np.ndarray:
    edge_lengths = np.zeros(node_count, dtype=lengths.dtype)
    for (parent, child), length in zip(edges, lengths):
        edge_lengths[child] = length
    return edge_lengths
