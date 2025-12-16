import dendropy
import numpy as np
from typing import Tuple

def parse_newick(
        newick_data: str, 
        is_file_path: bool = False,
        ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parses a Newick tree, converts it to binary, and returns edge lists and lengths.
    
    Indexing Strategy:
    - Leaves are indexed 0 to (num_leaves - 1).
    - Internal nodes are indexed num_leaves to (num_nodes - 1).
    - Uses postorder traversal (children before parents) to assign internal indices.

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
    node_to_idx = {}
    
    leaves = []
    internal_nodes = []

    # Iterate in Postorder: Visits children before parents.
    # This naturally handles "leaves first" for subtrees, but we explicitly
    # split lists to enforce the 0..L-1 constraint for all leaves.
    for node in tree.postorder_node_iter():
        if node.is_leaf():
            leaves.append(node)
        else:
            internal_nodes.append(node)

    # Assign indices: Leaves first (0 to L-1)
    for i, node in enumerate(leaves):
        node_to_idx[node] = i
        
    # Assign indices: Internal nodes next (L to N-1)
    # Since we collected them in postorder, these indices will generally 
    # increase from the bottom of the tree up to the root.
    num_leaves = len(leaves)
    for i, node in enumerate(internal_nodes):
        node_to_idx[node] = i + num_leaves

    edge_list = []
    length_list = []

    # 4. Traverse to collect edges
    # We can iterate postorder or preorder here; the edge list content is the same.
    # Postorder is safe.
    for node in tree.postorder_node_iter():
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