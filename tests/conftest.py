"""Shared test fixtures for active inference visualizer."""

import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    """Deterministic random number generator for reproducible tests."""
    return np.random.default_rng(seed=42)
