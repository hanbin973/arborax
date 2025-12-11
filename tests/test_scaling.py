import numpy as np
import pytest
from scipy.linalg import expm

from arborax.beagle_cffi import BeagleLikelihoodCalculator


def python_scaled_likelihood(tip_data, operations, edge_map, Q, pi, pattern_weights):
    """
    A Python implementation of Felsenstein's pruning algorithm that
    DYNAMICALLY RESCALES at every node to prevent underflow.

    Returns:
        (log_likelihood, node_partials_dict, node_cumulative_scales_dict)
    """

    # 1. Helper: Transition Matrix P(t)
    memoized_P = {}

    def get_P(length):
        if length not in memoized_P:
            memoized_P[length] = expm(Q * length)
        return memoized_P[length]

    # 2. Initialize State
    node_state = {}  # Stores the SCALED partials array (normalized)
    node_cumulative_scales = {}  # Stores the CUMULATIVE log-scale factor for the subtree

    # Load tips
    # Tips are unscaled, so cumulative log-scale is 0.0
    for k, v in tip_data.items():
        node_state[k] = v
        node_cumulative_scales[k] = 0.0

    # 3. Traverse Operations (Post-Order)
    for op in operations:
        dest = op["dest"]
        child1 = op["child1"]
        child2 = op["child2"]

        len1 = edge_map[child1][1]
        len2 = edge_map[child2][1]

        P1 = get_P(len1)
        P2 = get_P(len2)

        L1 = node_state[child1]
        L2 = node_state[child2]

        # Compute Raw Partials using SCALED children
        # This result is "locally raw" but implicitly scaled by (S_c1 + S_c2)
        branch1_prob = P1.dot(L1)
        branch2_prob = P2.dot(L2)
        raw_partials = branch1_prob * branch2_prob

        # --- SCALING STEP ---
        max_val = np.max(raw_partials)

        if max_val > 0:
            scaler = 1.0 / max_val
            scaled_partials = raw_partials * scaler
            local_log_scaler = np.log(max_val)
        else:
            scaled_partials = raw_partials
            local_log_scaler = -np.inf

        # Store Scaled Partials
        node_state[dest] = scaled_partials

        # Calculate and Store CUMULATIVE Scale
        # S_dest = s_local + S_child1 + S_child2
        cumulative_scale = (
            local_log_scaler
            + node_cumulative_scales[child1]
            + node_cumulative_scales[child2]
        )
        node_cumulative_scales[dest] = cumulative_scale

    # 4. Root Calculation
    root_idx = operations[-1]["dest"]
    root_partials = node_state[root_idx]

    site_L = np.dot(root_partials, pi)

    # The total likelihood is the Scaled Root Likelihood + Total Cumulative Scale at Root
    log_L = np.log(site_L) + node_cumulative_scales[root_idx]

    return np.sum(log_L * pattern_weights), node_state, node_cumulative_scales


def test_large_tree_scaling(use_gpu):
    print("\n=== STARTING LARGE TREE SCALING TEST ===")

    # 1. Constants
    N_TAXA = 50
    N_STATES = 4

    # 2. Model
    Q = np.ones((4, 4)) * 0.25
    np.fill_diagonal(Q, -0.75)
    pi = np.array([0.25, 0.25, 0.25, 0.25])

    # 3. Data
    np.random.seed(123)
    tip_data = {}
    for i in range(N_TAXA):
        state = np.zeros(4)
        state[np.random.randint(0, 4)] = 1.0
        tip_data[i] = state

    # 4. Build Ladder Topology
    edge_map = {}
    ops = []
    current_node = N_TAXA

    # First pair
    edge_map[0] = (current_node, 0.1)
    edge_map[1] = (current_node, 0.1)
    ops.append({"dest": current_node, "child1": 0, "child2": 1})

    prev_internal = current_node
    current_node += 1

    # Add remaining
    for i in range(2, N_TAXA):
        edge_map[prev_internal] = (current_node, 0.1)
        edge_map[i] = (current_node, 0.1)
        ops.append({"dest": current_node, "child1": prev_internal, "child2": i})
        prev_internal = current_node
        current_node += 1

    root_idx = prev_internal

    # 5. Run Python Implementation
    print(f"Computing Python Likelihood for {N_TAXA} taxa ladder tree...")
    py_logL, py_partials, py_cum_scales = python_scaled_likelihood(
        tip_data, ops, edge_map, Q, pi, np.array([1.0])
    )
    print(f"Python LogL: {py_logL}")

    # 6. Run BEAGLE Implementation
    print("Computing BEAGLE Likelihood...")
    beagle = BeagleLikelihoodCalculator(
        tip_count=N_TAXA, state_count=N_STATES, pattern_count=1, use_gpu=use_gpu
    )

    beagle.set_tip_partials(tip_data)
    beagle.set_model_matrix(Q, pi)

    max_idx = root_idx
    edge_lengths = np.zeros(max_idx + 1, dtype=np.float64)
    prob_indices = np.arange(max_idx + 1, dtype=np.int32)

    for child, (parent, length) in edge_map.items():
        edge_lengths[child] = length

    beagle.update_transition_matrices(edge_lengths, prob_indices)
    beagle.update_partials(ops)

    beagle_logL = beagle.calculate_root_log_likelihood(root_idx)
    print(f"Beagle LogL: {beagle_logL}")

    # 7. Verify Final Results
    np.testing.assert_allclose(beagle_logL, py_logL, atol=1e-5)

    # 8. --- UPDATED: Compare Reconstructed ORIGINAL Log-Partials ---
    print(
        "\nVerifying reconstructed internal partials (log(partial) + cumulative_log(scale))..."
    )

    # Check ALL internal nodes
    nodes_to_check = range(N_TAXA, root_idx + 1)

    for node in nodes_to_check:
        # A. Retrieve BEAGLE State
        # b_partials is the SCALED array
        # b_log_scale is the CUMULATIVE scale factor (as provided by getScaleFactors for that node buffer)
        b_partials, b_log_scales = beagle.get_partials_and_scale(node)

        # B. Retrieve Python State
        p_partials = py_partials[node]
        p_cum_scale = py_cum_scales[node]  # This is now the cumulative sum

        # C. Reconstruct Original Log Likelihoods
        # ln(L_real) = ln(L_scaled) + ln(CumulativeScale)
        with np.errstate(divide="ignore"):
            b_recon_log = np.log(b_partials.flatten()) + b_log_scales[0]
            p_recon_log = np.log(p_partials) + p_cum_scale

        # D. Compare
        finite_mask = np.isfinite(b_recon_log)

        if not np.all(np.isneginf(b_recon_log) == np.isneginf(p_recon_log)):
            print(f"Mismatch in zero-probability states at node {node}")
            # Debug print
            print("Beagle Recon:", b_recon_log)
            print("Python Recon:", p_recon_log)
            assert False, f"Zero-probability structure mismatch at node {node}"

        if np.any(finite_mask):
            np.testing.assert_allclose(
                b_recon_log[finite_mask],
                p_recon_log[finite_mask],
                atol=1e-5,
                err_msg=f"Reconstructed log-partial mismatch at node {node}",
            )

        print(f"  Node {node}: OK")

    print("[SUCCESS] All reconstructed partials match.")


if __name__ == "__main__":
    test_large_tree_scaling()
