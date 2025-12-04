import re

class Node:
    """
    Represents a node in the phylogenetic tree.
    """
    def __init__(self, id=None, name=None, length=0.0):
        self.id = id
        self.name = name
        self.length = length
        self.left = None
        self.right = None
        self.is_tip = False

def parse_newick_to_beagle_nodes(newick_str):
    """
    Parses a Newick string and assigns Beagle-compatible indices.
    
    Beagle Indexing Convention:
    - Tips: 0 to N-1
    - Internals: N to 2N-2
    - Root: 2N-2
    
    Returns: 
        (list of Node objects, int tip_count)
    """
    # Clean string
    newick = newick_str.strip().rstrip(';')
    tokens = re.split(r'([(),:])', newick)
    tokens = [t for t in tokens if t.strip()]

    stack = []
    current_node = None
    tips = []
    
    # 1. Build basic tree structure
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token == '(':
            node = Node()
            if stack:
                parent = stack[-1]
                if not parent.left: parent.left = node
                else: parent.right = node
            stack.append(node)
        elif token == ',':
            pass 
        elif token == ')':
            current_node = stack.pop()
        elif token == ':':
            i += 1
            length_str = tokens[i]
            target = current_node if current_node else stack[-1].right if stack[-1].right else stack[-1].left
            target.length = float(length_str)
        else:
            # Label (Tip)
            name = token
            node = Node(name=name)
            node.is_tip = True
            tips.append(node)
            parent = stack[-1]
            if not parent.left: parent.left = node
            else: parent.right = node
            current_node = node 
        i += 1

    # 2. Assign IDs (Beagle Convention)
    tip_map = {t.name: i for i, t in enumerate(tips)}
    for t in tips:
        t.id = tip_map[t.name]
    
    # Assign Internal IDs (Sequential post-order approximation)
    internal_counter = len(tips)
    all_nodes = tips[:]
    
    def traverse_assign(node):
        nonlocal internal_counter
        if node.is_tip: return
        
        if node.left: traverse_assign(node.left)
        if node.right: traverse_assign(node.right)
        
        if node.id is None:
            node.id = internal_counter
            internal_counter += 1
            all_nodes.append(node)

    # Find root (the one not in children)
    # Note: A real implementation needs a more robust root finder.
    # This assumes the last node processed in the stack was root or connected to it.
    # For this snippet, we assume the last node in 'all_nodes' after traverse is root.
    
    # Warning: This is a simplified parser. Production code should use DendroPy
    # to ensure correct post-order traversal for 'operations'.
    return all_nodes, len(tips)
