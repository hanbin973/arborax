import pytest
import numpy as np
from scipy.linalg import expm
from arborax.beagle_cffi import BeagleLikelihoodCalculator


def pure_python_likelihood(tip_data, edge_map, Q, pi, pattern_weights):
    """
    A recursive Felsenstein pruning algorithm in pure Python/SciPy.
    edge_map: dict {child_node_idx: (parent_node_idx, branch_length)}
    tip_data: dict {tip_idx: partial_likelihood_array}
    """

    node_partials = {}
    for k, v in tip_data.items():
        node_partials[k] = v

    def get_P(t):
        return expm(Q * t)

    # Node 5: Children 0, 1
    P_0 = get_P(edge_map[0][1])
    P_1 = get_P(edge_map[1][1])
    L_0 = P_0.dot(node_partials[0])
    L_1 = P_1.dot(node_partials[1])
    node_partials[5] = L_0 * L_1

    # Node 6: Children 5, 2
    P_5 = get_P(edge_map[5][1])
    P_2 = get_P(edge_map[2][1])
    L_5 = P_5.dot(node_partials[5])
    L_2 = P_2.dot(node_partials[2])
    node_partials[6] = L_5 * L_2

    # Node 7: Children 3, 4
    P_3 = get_P(edge_map[3][1])
    P_4 = get_P(edge_map[4][1])
    L_3 = P_3.dot(node_partials[3])
    L_4 = P_4.dot(node_partials[4])
    node_partials[7] = L_3 * L_4

    # Node 8 (Root): Children 6, 7
    P_6 = get_P(edge_map[6][1])
    P_7 = get_P(edge_map[7][1])
    L_6 = P_6.dot(node_partials[6])
    L_7 = P_7.dot(node_partials[7])
    root_partials = L_6 * L_7

    site_L = np.dot(root_partials, pi)
    log_L = np.log(site_L)
    return np.sum(log_L * pattern_weights)


def test_5taxa_likelihood():
    print("\n=== STARTING BEAGLE 5-TAXA TEST ===")

    N_TAXA = 5
    N_STATES = 4
    N_PATTERNS = 1

    # Jukes-Cantor Model
    Q = np.ones((4, 4)) * 0.25
    np.fill_diagonal(Q, -0.75)
    pi = np.array([0.25, 0.25, 0.25, 0.25])

    # Dummy Data
    np.random.seed(42)
    tip_data = {}
    for i in range(N_TAXA):
        state = np.zeros(4)
        state[np.random.randint(0, 4)] = 1.0
        tip_data[i] = state

    # Topology: (((0, 1)5, 2)6, (3, 4)7)8
    edge_map = {
        0: (5, 0.1),
        1: (5, 0.2),
        2: (6, 0.15),
        3: (7, 0.3),
        4: (7, 0.1),
        5: (6, 0.05),
        6: (8, 0.1),
        7: (8, 0.1),
    }

    # --- Run Python Implementation ---
    py_logL = pure_python_likelihood(tip_data, edge_map, Q, pi, np.array([1.0]))

    # --- Run BEAGLE Implementation ---
    beagle = BeagleLikelihoodCalculator(
        tip_count=N_TAXA, state_count=N_STATES, pattern_count=N_PATTERNS, use_gpu=False
    )

    beagle.set_tip_partials(tip_data)
    beagle.set_model_matrix(Q, pi)

    ops = [
        {"dest": 5, "child1": 0, "child2": 1},
        {"dest": 6, "child1": 5, "child2": 2},
        {"dest": 7, "child1": 3, "child2": 4},
        {"dest": 8, "child1": 6, "child2": 7},
    ]

    edge_lengths = []
    matrix_indices = []

    # Sort ensures we map indices consistently
    sorted_nodes = sorted([k for k in edge_map.keys()])
    for node_idx in sorted_nodes:
        length = edge_map[node_idx][1]
        edge_lengths.append(length)
        matrix_indices.append(node_idx)

    beagle.update_transition_matrices(
        edge_lengths=np.array(edge_lengths),
        probability_indices=np.array(matrix_indices),
    )

    beagle.update_partials(ops)
    beagle_logL = beagle.calculate_root_log_likelihood(root_index=8)

    # --- Compare ---
    print(f"Python: {py_logL}")
    print(f"Beagle: {beagle_logL}")

    np.testing.assert_allclose(
        beagle_logL, py_logL, atol=1e-6, err_msg="Likelihoods do not match!"
    )


def _random_rate_matrix(rng):
    rates = rng.uniform(0.05, 1.5, size=(4, 4))
    rates = (rates + rates.T) / 2.0
    np.fill_diagonal(rates, 0.0)
    row_sums = rates.sum(axis=1)
    np.fill_diagonal(rates, -row_sums)
    return rates


def _random_tip_vectors(tip_count, rng):
    tip_data = {}
    for i in range(tip_count):
        vec = rng.uniform(0.01, 2.0, size=4)
        vec /= vec.sum()
        tip_data[i] = vec
    return tip_data


def test_5taxa_randomized_likelihood(seed):
    rng = np.random.default_rng(seed)

    N_TAXA = 5
    N_STATES = 4

    Q = _random_rate_matrix(rng)
    pi = rng.dirichlet(np.ones(N_STATES))
    tip_data = _random_tip_vectors(N_TAXA, rng)

    edge_map = {
        0: (5, rng.uniform(0.01, 0.3)),
        1: (5, rng.uniform(0.01, 0.3)),
        2: (6, rng.uniform(0.01, 0.3)),
        3: (7, rng.uniform(0.01, 0.3)),
        4: (7, rng.uniform(0.01, 0.3)),
        5: (6, rng.uniform(0.01, 0.3)),
        6: (8, rng.uniform(0.01, 0.3)),
        7: (8, rng.uniform(0.01, 0.3)),
    }

    pattern_weights = np.array([1.0])

    py_logL = pure_python_likelihood(tip_data, edge_map, Q, pi, pattern_weights)

    beagle = BeagleLikelihoodCalculator(
        tip_count=N_TAXA, state_count=N_STATES, pattern_count=1, use_gpu=False
    )
    beagle_tip_partials = {i: v.reshape(1, -1) for i, v in tip_data.items()}
    beagle.set_tip_partials(beagle_tip_partials)
    beagle.set_model_matrix(Q, pi)

    ops = [
        {"dest": 5, "child1": 0, "child2": 1},
        {"dest": 6, "child1": 5, "child2": 2},
        {"dest": 7, "child1": 3, "child2": 4},
        {"dest": 8, "child1": 6, "child2": 7},
    ]
    edge_lengths = np.zeros(9)
    matrix_indices = np.arange(9, dtype=np.int32)
    for child, (_, length) in edge_map.items():
        edge_lengths[child] = length

    beagle.update_transition_matrices(edge_lengths, matrix_indices)
    beagle.update_partials(ops)
    beagle_logL = beagle.calculate_root_log_likelihood(root_index=8)

    np.testing.assert_allclose(
        beagle_logL, py_logL, atol=1e-5, err_msg="Randomized likelihood mismatch"
    )


if __name__ == "__main__":
    test_5taxa_likelihood()
