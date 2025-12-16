import dendropy
import numpy as np
from typing import Tuple

def parse_newick(
        newick_data: str, 
        is_file_path: bool = False,
        ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parses a Newick tree, converts it to binary, and returns edge lists and lengths.

    Args:
        newick_data: The Newick string or a file path to a Newick file.
        is_file_path: Set to True if newick_data is a path, False if it is a raw string.

    Returns:
        edge_indices: A numpy array of shape (E, 2) containing [parent_idx, child_idx].
        edge_lengths: A numpy array of shape (E,) containing the length of each edge.
    """
    # 1. Load the tree
    if is_file_path:
        tree = dendropy.Tree.get(path=newick_data, schema="newick")
    else:
        tree = dendropy.Tree.get(data=newick_data, schema="newick")

    # 2. Convert to binary tree
    # This splits multi-furcating nodes (polytomies) by inserting 
    # zero-length branches to create a strictly binary structure.
    tree.resolve_polytomies()

    # 3. Map nodes to integer indices
    # We create a dictionary mapping the Dendropy Node object to a unique integer (0 to N-1)
    # We map based on a standard traversal (preorder is common)
    node_to_idx = {node: i for i, node in enumerate(tree)}

    edge_list = []
    length_list = []

    # 4. Traverse to collect edges
    # In Dendropy, edges are attributes of the node they lead TO (the child).
    for node in tree:
        # The root node has no parent, so it represents the start of the tree, 
        # not an edge connecting two nodes. We skip it.
        if node.parent_node is not None:
            p_idx = node_to_idx[node.parent_node]
            c_idx = node_to_idx[node]
            
            # Handle cases where edge length might be None (topology only)
            length = node.edge.length if node.edge.length is not None else 0.0
            
            edge_list.append([p_idx, c_idx])
            length_list.append(length)

    # Convert to NumPy arrays
    edge_indices = np.array(edge_list, dtype=int)
    edge_lengths = np.array(length_list, dtype=float)

    return edge_indices, edge_lengths
