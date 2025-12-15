import random
from collections.abc import Mapping
from functools import lru_cache

import numpy as np
import pytest

from arborax.beagle_cffi import BeagleLikelihoodCalculator
from arborax.context import ArboraxContext

try:
    import jax.random as jrandom
except ImportError:  # pragma: no cover
    jrandom = None


@pytest.fixture(params=[0, 1, 2, 3, 4])
def seed(request):
    """Global fixture that seeds Python, NumPy, and (optionally) JAX RNGs."""
    seed_value = request.param
    random.seed(seed_value)
    np.random.seed(seed_value)
    if jrandom is not None:
        _ = jrandom.PRNGKey(seed_value)
    return seed_value


@pytest.fixture(params=[True, False])
def use_gpu(request):
    """Run tests against both CPU and GPU BEAGLE backends when available."""
    return request.param


@lru_cache(maxsize=1)
def _beagle_gpu_supports_scale_factors() -> bool:
    try:
        calc = BeagleLikelihoodCalculator(
            tip_count=3, state_count=4, pattern_count=1, use_gpu=True
        )
        try:
            # Any call is sufficient; GPU currently returns -7 (NO_IMPLEMENTATION).
            _ = calc.get_scale_factors(0)
        finally:
            calc.close()
    except RuntimeError as exc:
        if "beagleGetScaleFactors" in str(exc) and "NO_IMPLEMENTATION" in str(exc):
            return False
        # Unexpected failure: surface it rather than silently skipping.
        raise
    return True


@pytest.fixture
def require_beagle_scalers(use_gpu):
    """Skip when BEAGLE scalers can't be retrieved for the selected backend."""
    if use_gpu and not _beagle_gpu_supports_scale_factors():
        pytest.skip(
            "Selected BEAGLE backend does not implement beagleGetScaleFactors "
            "(BeagleGPUImpl::getScaleFactors returns BEAGLE_ERROR_NO_IMPLEMENTATION)."
        )


def build_context_binder(
    tip_data: Mapping[int, np.ndarray],
    operations: list[dict[str, int]],
    pattern_count: int,
    use_gpu: bool = False,
) -> ArboraxContext:
    """Construct an ArboraxContext and bind tip data for use in JAX tests."""
    if use_gpu and not _beagle_gpu_supports_scale_factors():
        pytest.skip(
            "ArboraxContext currently requires beagleGetScaleFactors for its custom VJP, "
            "but the selected BEAGLE backend does not implement it."
        )

    if not tip_data:
        raise ValueError("tip_data must not be empty")

    tip_dict = {
        idx: np.asarray(array, dtype=np.float64) for idx, array in tip_data.items()
    }
    first = next(iter(tip_dict.values()))
    state_count = first.shape[1]
    if any(arr.shape != first.shape for arr in tip_dict.values()):
        raise ValueError("All tip arrays must share the same (patterns, states) shape")

    context = ArboraxContext(
        tip_count=len(tip_dict),
        operations=operations,
        pattern_count=pattern_count,
        use_gpu=use_gpu,
    )
    dummy_Q = np.eye(state_count, dtype=np.float64)
    dummy_pi = np.full(state_count, 1.0 / state_count, dtype=np.float64)
    context.bind_data(tip_dict, dummy_Q, dummy_pi)
    return context
