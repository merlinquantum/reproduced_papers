import pathlib
import sys

# Ensure this tests directory is on sys.path to import shared helper
_TESTS_DIR = pathlib.Path(__file__).parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from common import _load_impl_module


def test_placeholder():
    # Basic smoke test to ensure template imports and runs
    _ = _load_impl_module()
