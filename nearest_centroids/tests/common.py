import importlib.util
import pathlib
import sys


def _load_impl_module():
    impl_path = pathlib.Path(__file__).resolve().parents[1] / "implementation.py"
    assert impl_path.exists(), "implementation.py missing"
    # Ensure local 'lib' package is importable
    sys.path.insert(0, str(impl_path.parent))
    spec = importlib.util.spec_from_file_location("impl", impl_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
