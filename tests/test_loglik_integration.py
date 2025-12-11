import numpy as np

from arborax import loglik
from tests.test_reference import (
    _edge_list_and_lengths,
    _generate_problem,
    _jax_felsenstein,
)


def test_loglik_matches_reference_algorithm(seed, use_gpu):
    problem = _generate_problem(seed, use_gpu=use_gpu)

    tip_count = problem["context"].calc.tip_count
    tip_array = np.stack([problem["tip_data"][idx] for idx in range(tip_count)], axis=0)
    edge_list, branch_lengths = _edge_list_and_lengths(problem["edge_map"])

    api_ll = loglik(
        tip_partials=tip_array,
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
