import random

import numpy as np
import pytest

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
        # Initialize a JAX key so downstream code can derive keys deterministically.
        _ = jrandom.PRNGKey(seed_value)
    return seed_value
