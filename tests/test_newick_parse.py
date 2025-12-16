import numpy as np
import pytest
from arborax.io import parse_newick


@pytest.fixture
def polytomy_newick():
    """
    A Newick string with a polytomy (A, B, C share one parent).
    After binarization, this requires adding a new internal node and a 0-length edge.
    Structure:
       /-- A (0.1)
    --|--- B (0.2)
       \-- C (0.3)
    Parent connects to Root with 0.5
    Root connects to D with 0.8
    """
    return "((A:0.1,B:0.2,C:0.3):0.5,D:0.8);"

def test_parse_and_vectorize_structure(polytomy_newick):
    """
    Tests that the function returns the correct array shapes and 
    successfully converts a polytomy to a binary tree.
    """
    indices, lengths = parse_newick(polytomy_newick)

    # 1. Check Output Types
    assert isinstance(indices, np.ndarray)
    assert isinstance(lengths, np.ndarray)
    
    # 2. Check Tree Properties
    # Original leaves: A, B, C, D (4 leaves)
    # A binary tree with N leaves has 2*N - 1 total nodes.
    # Total nodes = 2*4 - 1 = 7 nodes.
    # Total edges = Total nodes - 1 (since it's a tree and we exclude root's parent) = 6 edges.
    
    expected_edges = 6
    assert indices.shape == (expected_edges, 2)
    assert lengths.shape == (expected_edges,)

def test_edge_lengths_preserved(polytomy_newick):
    """
    Tests that the original edge lengths exist in the output, 
    plus the new 0.0 length edge from binarization.
    """
    _, lengths = parse_newick(polytomy_newick)
    
    # We expect the original lengths: 0.1, 0.2, 0.3, 0.5, 0.8
    # PLUS at least one 0.0 length edge created to resolve the split of A,B,C
    expected_values = [0.1, 0.2, 0.3, 0.5, 0.8]
    
    for val in expected_values:
        # Check if the value exists in the array (using approx for float comparison)
        assert np.any(np.isclose(lengths, val)), f"Expected length {val} not found in output"

    # Check that a zero-length edge was added for binarization
    assert np.any(np.isclose(lengths, 0.0)), "Binarization should have added a 0.0 length edge"

def test_connectivity_integrity(polytomy_newick):
    """
    Tests that valid parent-child relationships are returned.
    """
    indices, _ = parse_newick(polytomy_newick)
    
    # 1. No self-loops (parent != child)
    assert np.all(indices[:, 0] != indices[:, 1])
    
    # 2. Max index should be 6 (since there are 7 nodes, 0-6)
    assert indices.max() == 6
    assert indices.min() == 0
