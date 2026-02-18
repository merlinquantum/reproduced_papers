import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
HF_CACHE_DIR = REPO_ROOT / ".cache" / "huggingface"
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("HF_DATASETS_CACHE", str(HF_CACHE_DIR / "datasets"))

try:
    from papers.DQNN.lib.boson_sampler import BosonSampler
except ModuleNotFoundError:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    for path in (REPO_ROOT, PROJECT_ROOT):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from papers.DQNN.lib.boson_sampler import BosonSampler


@pytest.fixture
def bs_1():
    return BosonSampler(m=9, n=4)


@pytest.fixture
def bs_2():
    return BosonSampler(m=8, n=4)
