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

import numpy as np
from collections import defaultdict

def prepare_inputs_for_jax(edge_indices, edge_lengths):
    """
    Converts edge lists from parse_newick into JAX-compatible Ops and aligned arrays.
    
    Args:
        edge_indices: (E, 2) array of [parent, child] from parse_newick.
        edge_lengths: (E,) array of branch lengths.
        
    Returns:
        ops: (N_internal, 3) array of [parent, child_left, child_right].
             Sorted by parent index to ensure valid post-order execution.
        aligned_branch_lengths: (N_nodes,) array. 
             aligned_branch_lengths[i] is the length of the branch ABOVE node i.
             The root (last index) will have length 0.0.
    """
    # 1. Determine total number of nodes (N)
    # The maximum index in edge_indices is the root (or largest internal node)
    num_nodes = edge_indices.max() + 1
    
    # 2. Align branch lengths to node indices
    # We want an array where array[i] = length of branch ending at node i.
    aligned_branch_lengths = np.zeros(num_nodes)
    
    # edge_indices[:, 1] is the column of 'child' indices. 
    # The branch length corresponds to the branch leading TO the child.
    children_indices = edge_indices[:, 1]
    aligned_branch_lengths[children_indices] = edge_lengths
    
    # 3. Group children by parent to build Ops
    # We use a dictionary to collect [child1, child2] for each parent.
    # Since resolve_polytomies() was called, strict binary is expected.
    children_map = defaultdict(list)
    for (p, c) in edge_indices:
        children_map[p].append(c)
        
    # 4. Construct the Ops array
    # We must iterate through parents in strictly increasing order.
    # Your parse_newick assigns IDs L to N-1 in postorder, so sorting 
    # parents ascendingly ensures we process children before parents.
    internal_parents = sorted(children_map.keys())
    
    ops_list = []
    for p in internal_parents:
        children = children_map[p]
        
        # Sanity check for binary tree
        if len(children) != 2:
            raise ValueError(f"Node {p} is not binary. Found children: {children}")
            
        # Append [parent, left, right]
        ops_list.append([p, children[0], children[1]])
        
    ops = np.array(ops_list, dtype=np.int32)
    
    return ops, aligned_branch_lengths
