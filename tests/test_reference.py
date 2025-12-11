import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from jax import lax

from arborax import loglik
from arborax.context import ArboraxContext


def _random_rate_matrix(state_count, rng):
    rates = rng.uniform(0.05, 1.5, size=(state_count, state_count))
    rates = (rates + rates.T) / 2.0
    np.fill_diagonal(rates, 0.0)
    row_sums = rates.sum(axis=1)
    np.fill_diagonal(rates, -row_sums)
    return rates.astype(np.float32)


def _random_pi(state_count, rng):
    vec = rng.uniform(0.05, 1.0, size=state_count)
    vec /= vec.sum()
    return vec.astype(np.float32)


def _random_tip_partials(tip_count, pattern_count, state_count, rng):
    tip_data = {}
    for tip in range(tip_count):
        raw = rng.uniform(0.05, 1.5, size=(pattern_count, state_count))
        normalized = raw / raw.sum(axis=1, keepdims=True)
        tip_data[tip] = normalized.astype(np.float32)
    return tip_data


def _build_random_tree(tip_count, rng, edge_range=(0.01, 0.25)):
    nodes = list(range(tip_count))
    next_node = tip_count
    operations = []
    edge_map = {}

    while len(nodes) > 1:
        child1, child2 = rng.choice(nodes, size=2, replace=False)
        nodes = [n for n in nodes if n not in (child1, child2)]
        parent = next_node
        next_node += 1
        len1 = rng.uniform(*edge_range)
        len2 = rng.uniform(*edge_range)
        edge_map[child1] = (parent, len1)
        edge_map[child2] = (parent, len2)
        operations.append({"dest": parent, "child1": child1, "child2": child2})
        nodes.append(parent)

    root = nodes[0]
    assert operations[-1]["dest"] == root
    node_count = next_node
    return operations, edge_map, root, node_count


def _edge_lengths_array(edge_map, node_count):
    arr = np.zeros(node_count, dtype=np.float32)
    for child, (_, length) in edge_map.items():
        arr[child] = length
    return arr


def _ops_array(operations):
    return np.array(
        [[op["dest"], op["child1"], op["child2"]] for op in operations], dtype=np.int32
    )


def _tip_tensor(tip_data, node_count):
    first = next(iter(tip_data.values()))
    tensor = np.zeros((node_count, *first.shape), dtype=np.float32)
    for idx, arr in tip_data.items():
        tensor[idx] = arr
    return tensor


def _jax_felsenstein(edge_lengths, Q, pi, tip_tensor, ops_array):
    def body(i, partials):
        dest, child1, child2 = ops_array[i]
        P1 = jsp.linalg.expm(Q * edge_lengths[child1])
        P2 = jsp.linalg.expm(Q * edge_lengths[child2])
        branch1 = jnp.dot(partials[child1], P1.T)
        branch2 = jnp.dot(partials[child2], P2.T)
        dest_partials = branch1 * branch2
        return partials.at[dest].set(dest_partials)

    partials = lax.fori_loop(0, ops_array.shape[0], body, tip_tensor)
    root = ops_array[-1, 0]
    site_probs = jnp.dot(partials[root], pi)
    safe_site = jnp.clip(site_probs, 1e-30, None)
    return jnp.log(safe_site)


def _reference_sum_loglik(edge_lengths, Q, pi, tip_tensor, ops_array):
    return jnp.sum(_jax_felsenstein(edge_lengths, Q, pi, tip_tensor, ops_array))


def _generate_problem(
    seed,
    tip_count=5,
    pattern_count=3,
    state_count=4,
    edge_range=(0.01, 0.25),
    use_gpu=False,
):
    rng = np.random.default_rng(seed)

    operations, edge_map, _, node_count = _build_random_tree(
        tip_count, rng, edge_range=edge_range
    )
    tip_data = _random_tip_partials(tip_count, pattern_count, state_count, rng)

    edge_lengths = _edge_lengths_array(edge_map, node_count)
    ops_array = _ops_array(operations)
    tip_tensor = _tip_tensor(tip_data, node_count)

    Q_np = _random_rate_matrix(state_count, rng)
    pi_np = _random_pi(state_count, rng)

    context = ArboraxContext(
        tip_count=tip_count,
        operations=operations,
        pattern_count=pattern_count,
        use_gpu=use_gpu,
    )
    context.bind_data(tip_data, Q_np, pi_np)

    return {
        "context": context,
        "edge_lengths": jnp.array(edge_lengths, dtype=jnp.float32),
        "Q": jnp.array(Q_np, dtype=jnp.float32),
        "pi": jnp.array(pi_np, dtype=jnp.float32),
        "tip_tensor": jnp.array(tip_tensor, dtype=jnp.float32),
        "ops_array": jnp.array(ops_array, dtype=jnp.int32),
        "edge_map": edge_map,
        "tip_data": tip_data,
    }


def test_random_loglik_matches_reference(seed, use_gpu):
    problem = _generate_problem(seed, use_gpu=use_gpu)

    beagle_ll = jnp.sum(
        problem["context"].likelihood_functional(
            problem["Q"], problem["pi"], problem["edge_lengths"]
        )
    )
    reference_ll = _reference_sum_loglik(
        problem["edge_lengths"],
        problem["Q"],
        problem["pi"],
        problem["tip_tensor"],
        problem["ops_array"],
    )
    np.testing.assert_allclose(
        np.array(beagle_ll), np.array(reference_ll), atol=1e-5, rtol=1e-5
    )


def test_random_gradients_match_reference(seed, use_gpu):
    problem = _generate_problem(seed, use_gpu=use_gpu)

    context = problem["context"]
    edge_lengths_jax = problem["edge_lengths"]
    Q_jax = problem["Q"]
    pi_jax = problem["pi"]
    tip_tensor_jax = problem["tip_tensor"]
    ops_array_jax = problem["ops_array"]

    def beagle_sum_ll(Q):
        return jnp.sum(context.likelihood_functional(Q, pi_jax, edge_lengths_jax))

    def beagle_sum_ll_pi(pi_vec):
        return jnp.sum(context.likelihood_functional(Q_jax, pi_vec, edge_lengths_jax))

    def reference_sum_ll(Q, pi_vec):
        return _reference_sum_loglik(
            edge_lengths_jax, Q, pi_vec, tip_tensor_jax, ops_array_jax
        )

    grad_Q_beagle = jax.grad(beagle_sum_ll)(Q_jax)
    grad_pi_beagle = jax.grad(beagle_sum_ll_pi)(pi_jax)
    grad_Q_ref = jax.grad(lambda Q: reference_sum_ll(Q, pi_jax))(Q_jax)
    grad_pi_ref = jax.grad(lambda pi_vec: reference_sum_ll(Q_jax, pi_vec))(pi_jax)

    np.testing.assert_allclose(
        np.array(grad_Q_beagle), np.array(grad_Q_ref), atol=1e-3, rtol=1e-3
    )
    np.testing.assert_allclose(
        np.array(grad_pi_beagle), np.array(grad_pi_ref), atol=1e-3, rtol=1e-3
    )


def test_zero_length_edges_stability(seed, use_gpu):
    rng = np.random.default_rng(seed)
    tip_count = 4
    pattern_count = 2
    state_count = 4

    operations, edge_map, root, node_count = _build_random_tree(
        tip_count, rng, edge_range=(0.0, 0.2)
    )
    for child in list(edge_map.keys())[:2]:
        parent, _ = edge_map[child]
        edge_map[child] = (parent, 0.0)

    tip_data = _random_tip_partials(tip_count, pattern_count, state_count, rng)
    edge_lengths = _edge_lengths_array(edge_map, node_count)
    ops_array = _ops_array(operations)
    tip_tensor = _tip_tensor(tip_data, node_count)

    Q_np = _random_rate_matrix(state_count, rng)
    pi_np = _random_pi(state_count, rng)

    context = ArboraxContext(
        tip_count=tip_count,
        operations=operations,
        pattern_count=pattern_count,
        use_gpu=use_gpu,
    )
    context.bind_data(tip_data, Q_np, pi_np)

    edge_lengths_jax = jnp.array(edge_lengths, dtype=jnp.float32)
    Q_jax = jnp.array(Q_np, dtype=jnp.float32)
    pi_jax = jnp.array(pi_np, dtype=jnp.float32)
    tip_tensor_jax = jnp.array(tip_tensor, dtype=jnp.float32)
    ops_array_jax = jnp.array(ops_array, dtype=jnp.int32)

    beagle_ll = jnp.sum(context.likelihood_functional(Q_jax, pi_jax, edge_lengths_jax))
    reference_ll = _reference_sum_loglik(
        edge_lengths_jax, Q_jax, pi_jax, tip_tensor_jax, ops_array_jax
    )
    np.testing.assert_allclose(
        np.array(beagle_ll), np.array(reference_ll), atol=1e-4, rtol=1e-4
    )

    grad_Q_beagle = jax.grad(
        lambda Q: jnp.sum(context.likelihood_functional(Q, pi_jax, edge_lengths_jax))
    )(Q_jax)
    grad_Q_ref = jax.grad(
        lambda Q: _reference_sum_loglik(
            edge_lengths_jax, Q, pi_jax, tip_tensor_jax, ops_array_jax
        )
    )(Q_jax)
    np.testing.assert_allclose(
        np.array(grad_Q_beagle), np.array(grad_Q_ref), atol=1e-3, rtol=1e-3
    )


def _edge_list_and_lengths(edge_map):
    edges = []
    lengths = []
    for child in sorted(edge_map.keys()):
        parent, length = edge_map[child]
        edges.append((parent, child))
        lengths.append(length)
    return np.array(edges, dtype=np.int32), np.array(lengths, dtype=np.float32)


def test_loglik_api_matches_context(seed, use_gpu):
    problem = _generate_problem(seed, use_gpu=use_gpu)
    tip_count = problem["context"].calc.tip_count
    tip_array = np.stack([problem["tip_data"][i] for i in range(tip_count)], axis=0)
    edge_list, lengths = _edge_list_and_lengths(problem["edge_map"])

    api_ll = loglik(
        tip_array,
        edge_list=edge_list,
        branch_lengths=lengths,
        Q=np.array(problem["Q"]),
        pi=np.array(problem["pi"]),
    )
    context_ll = problem["context"].likelihood_functional(
        problem["Q"], problem["pi"], problem["edge_lengths"]
    )
    np.testing.assert_allclose(
        np.array(api_ll), np.array(context_ll), atol=1e-5, rtol=1e-5
    )
