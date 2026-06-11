"""Shared fixtures for the test suite."""

import pytest

from src.config import Config


@pytest.fixture
def cfg(tmp_path):
    """A Config whose data/output dirs point at a temporary location."""
    return Config(data_dir=tmp_path / "data", output_dir=tmp_path / "results")
