import jax
import jax.numpy as jnp
import numpy as np

from arborax import loglik
from tests.test_reference import (
    _edge_list_and_lengths,
    _generate_problem,
    _jax_felsenstein,
)


def _loglik_wrapper(tip_array, edge_list, branch_lengths, Q, pi):
    return loglik(
        tip_partials=tip_array,
        edge_list=edge_list,
        branch_lengths=branch_lengths,
        Q=Q,
        pi=pi,
    )


def test_loglik_matches_reference_algorithm(seed, use_gpu):
    problem = _generate_problem(seed, use_gpu=use_gpu)

    tip_count = problem["context"].tip_count
    tip_array = np.stack([problem["tip_data"][idx] for idx in range(tip_count)], axis=0)
    edge_list, branch_lengths = _edge_list_and_lengths(problem["edge_map"])

    api_ll = _loglik_wrapper(
        tip_array,
        edge_list=edge_list,
        branch_lengths=branch_lengths,
        Q=np.array(problem["Q"]),
        pi=np.array(problem["pi"]),
    )

    reference_ll = _jax_felsenstein(
        problem["edge_lengths"],
        problem["Q"],
        problem["pi"],
        problem["tip_tensor"],
        problem["ops_array"],
    )

    np.testing.assert_allclose(
        np.array(api_ll), np.array(reference_ll), atol=1e-5, rtol=1e-5
    )


def test_loglik_vmap_over_Q(seed, use_gpu):
    """Verify loglik can be vmapped over a batch of rate matrices."""
    problem = _generate_problem(seed, use_gpu=use_gpu)

    tip_count = problem["context"].tip_count
    tip_array = np.stack([problem["tip_data"][idx] for idx in range(tip_count)], axis=0)
    edge_list, branch_lengths = _edge_list_and_lengths(problem["edge_map"])

    base_Q = np.array(problem["Q"])
    pi = np.array(problem["pi"])

    batch = 3
    perturb = np.stack(
        [np.eye(base_Q.shape[0]) * (0.01 * i) for i in range(batch)], axis=0
    )
    Q_batch = base_Q[None, :, :] + perturb

    def batched_fn(Q_matrix):
        return _loglik_wrapper(
            tip_array,
            edge_list=edge_list,
            branch_lengths=branch_lengths,
            Q=Q_matrix,
            pi=pi,
        )

    vmapped = jax.vmap(batched_fn)
    vmapped_vals = vmapped(jnp.array(Q_batch, dtype=jnp.float32))

    expected = []
    for idx in range(batch):
        expected.append(
            _loglik_wrapper(
                tip_array,
                edge_list=edge_list,
                branch_lengths=branch_lengths,
                Q=Q_batch[idx],
                pi=pi,
            )
        )

    np.testing.assert_allclose(
        np.array(vmapped_vals),
        np.stack([np.array(val) for val in expected], axis=0),
        atol=1e-5,
        rtol=1e-5,
    )
