import numpy as np
import jax
import jax.numpy as jnp
from arborax.binder import BeagleJAX
from arborax.beagle_cffi import BeagleLikelihoodCalculator


def test_batch_site_likelihood():
    print("\n=== STARTING BATCH SITE PATTERN TEST (Float32) ===")

    # 1. Setup Constants
    N_TAXA = 3
    N_STATES = 4
    N_PATTERNS = 10  # Increased to 10 sites

    # 2. Define Model (Jukes-Cantor)
    Q = np.ones((4, 4)) * 0.25
    np.fill_diagonal(Q, -0.75)
    pi = np.array([0.25, 0.25, 0.25, 0.25])

    # 3. Define Topology (Simple V shape + stem)
    # (0, 1) -> 3, (3, 2) -> 4 (Root)
    edge_map = {0: (3, 0.1), 1: (3, 0.1), 2: (4, 0.1), 3: (4, 0.1)}

    ops = [{"dest": 3, "child1": 0, "child2": 1}, {"dest": 4, "child1": 3, "child2": 2}]

    # Flatten edges for JAX input (Using float32)
    edge_lengths_jax = jnp.array([0.1, 0.1, 0.1, 0.1], dtype=jnp.float32)
    Q_jax = jnp.array(Q, dtype=jnp.float32)
    pi_jax = jnp.array(pi, dtype=jnp.float32)

    # 4. Define Data for N_PATTERNS Sites
    # Generate random deterministic data for reproducibility
    np.random.seed(42)
    tip_partials_np = np.zeros((N_TAXA, N_PATTERNS, N_STATES), dtype=np.float32)

    print(f"Generating {N_PATTERNS} random site patterns...")
    for p in range(N_PATTERNS):
        for t in range(N_TAXA):
            # Randomly assign a state (A, C, G, or T)
            state_idx = np.random.randint(0, 4)
            tip_partials_np[t, p, state_idx] = 1.0

    # 5. Run BATCHED JAX Binder
    print(f"Initializing JAX Binder...")

    dummy_tips = {i: np.ones((N_PATTERNS, N_STATES)) for i in range(N_TAXA)}

    binder = BeagleJAX(
        tip_count=N_TAXA,
        state_count=N_STATES,
        pattern_count=N_PATTERNS,
        tip_data=dummy_tips,
        edge_map=edge_map,
        operations=ops,
        use_gpu=False,
        dtype=jnp.float32,
    )

    print("Calling log_likelihood with dynamic tip partials...")
    tip_partials_jax = jnp.array(tip_partials_np)

    jit_log_likelihood = jax.jit(binder.log_likelihood)

    batched_result = jit_log_likelihood(
        edge_lengths_jax, Q_jax, pi_jax, tip_partials_jax
    )

    print(f"Batched Result Shape: {batched_result.shape}")

    # 6. Verify against INDIVIDUAL runs
    print("\nVerifying against individual single-pattern runs...")
    print(
        f"{'Pattern':<8} | {'Batched (JAX)':<15} | {'Reference (Seq)':<15} | {'Difference':<12}"
    )
    print("-" * 56)

    expected_lls = []

    for p_idx in range(N_PATTERNS):
        single_site_tips = {}
        for t in range(N_TAXA):
            # Cast to float64 for reference calculation
            single_site_tips[t] = tip_partials_np[t, p_idx, :].astype(np.float64)

        ref_beagle = BeagleLikelihoodCalculator(N_TAXA, N_STATES, pattern_count=1)
        ref_beagle.set_tip_partials(single_site_tips)
        ref_beagle.set_model_matrix(Q, pi)

        indices = np.arange(4, dtype=np.int32)
        lens = np.array([0.1, 0.1, 0.1, 0.1])
        ref_beagle.update_transition_matrices(lens, indices)
        ref_beagle.update_partials(ops)

        ll = ref_beagle.calculate_root_log_likelihood(root_index=4)
        expected_lls.append(ll)

        batch_val = float(batched_result[p_idx])
        diff = abs(batch_val - ll)
        print(f"{p_idx:<8} | {batch_val:<15.6f} | {ll:<15.6f} | {diff:<12.2e}")

    expected_lls = np.array(expected_lls, dtype=np.float32)

    np.testing.assert_allclose(
        batched_result,
        expected_lls,
        atol=1e-5,
        err_msg="Batched JAX result does not match individual BEAGLE runs!",
    )

    print("\n[SUCCESS] Batched site calculation matches individual runs.")


def _random_rate_matrix(rng):
    rates = rng.uniform(0.05, 1.5, size=(4, 4))
    rates = (rates + rates.T) / 2.0
    np.fill_diagonal(rates, 0.0)
    row_sums = rates.sum(axis=1)
    np.fill_diagonal(rates, -row_sums)
    return rates


def test_batch_site_likelihood_randomized(seed):
    rng = np.random.default_rng(seed)
    N_TAXA = 3
    N_STATES = 4
    N_PATTERNS = 6

    Q = _random_rate_matrix(rng)
    pi = rng.dirichlet(np.ones(N_STATES))

    edge_map = {
        0: (3, rng.uniform(0.05, 0.2)),
        1: (3, rng.uniform(0.05, 0.2)),
        2: (4, rng.uniform(0.05, 0.2)),
        3: (4, rng.uniform(0.05, 0.2)),
    }
    ops = [{"dest": 3, "child1": 0, "child2": 1}, {"dest": 4, "child1": 3, "child2": 2}]

    tip_partials = rng.uniform(0.01, 1.5, size=(N_TAXA, N_PATTERNS, N_STATES))
    tip_partials /= tip_partials.sum(axis=2, keepdims=True)
    tip_partials = tip_partials.astype(np.float32)

    dummy_tips = {
        i: np.ones((N_PATTERNS, N_STATES), dtype=np.float32) for i in range(N_TAXA)
    }
    binder = BeagleJAX(
        tip_count=N_TAXA,
        state_count=N_STATES,
        pattern_count=N_PATTERNS,
        tip_data=dummy_tips,
        edge_map=edge_map,
        operations=ops,
        dtype=jnp.float32,
    )

    edge_lengths = np.zeros(len(edge_map), dtype=np.float64)
    for child, (_, length) in edge_map.items():
        edge_lengths[child] = length

    edge_lengths_jax = jnp.array(edge_lengths, dtype=jnp.float32)
    Q_jax = jnp.array(Q, dtype=jnp.float32)
    pi_jax = jnp.array(pi, dtype=jnp.float32)
    tip_partials_jax = jnp.array(tip_partials, dtype=jnp.float32)

    batched_ll = binder.log_likelihood(
        edge_lengths_jax, Q_jax, pi_jax, tip_partials_jax
    )

    expected_lls = []
    for p_idx in range(N_PATTERNS):
        tip_dict = {
            i: tip_partials[i, p_idx, :].reshape(1, -1).astype(np.float64)
            for i in range(N_TAXA)
        }
        ref = BeagleLikelihoodCalculator(N_TAXA, N_STATES, pattern_count=1)
        ref.set_tip_partials(tip_dict)
        ref.set_model_matrix(Q, pi)
        indices = np.arange(len(edge_map), dtype=np.int32)
        ref.update_transition_matrices(edge_lengths, indices)
        ref.update_partials(ops)
        expected_lls.append(ref.calculate_root_log_likelihood(root_index=4))

    expected_lls = np.array(expected_lls, dtype=np.float32)
    np.testing.assert_allclose(
        batched_ll,
        expected_lls,
        atol=1e-4,
        err_msg="Randomized batched results do not match sequential evaluations",
    )


if __name__ == "__main__":
    test_batch_site_likelihood()
