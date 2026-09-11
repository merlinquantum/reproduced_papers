"""Test configuration: make the project importable and share small fixtures."""

import sys
from pathlib import Path

import pytest

# The project is a flat package tree (lib/, models/, utils/) imported as
# top-level modules, so the project directory has to be on sys.path. pytest only
# adds tests/ itself, hence the explicit insert.
PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

#: Base seed of the shipped configs; used so tests exercise real instances.
BASE_SEED = 101200


@pytest.fixture
def clique_graph():
    """A small Max-Clique instance with a constant diagonal (no augmentation)."""
    from utils.graphs import sample_instance_graph

    return sample_instance_graph(5, BASE_SEED + 300)


@pytest.fixture
def cut_graph():
    """A small Max-Cut instance with a varying diagonal (forces augmentation)."""
    from utils.graphs import sample_instance_graph

    return sample_instance_graph(5, BASE_SEED + 300)


@pytest.fixture
def configs_dir():
    """Path to the shipped run configs."""
    return PROJECT_DIR / "configs"
