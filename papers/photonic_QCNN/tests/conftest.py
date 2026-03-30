import sys
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PAPERS_ROOT = REPO_ROOT / "papers"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PAPERS_ROOT) not in sys.path:
    sys.path.insert(0, str(PAPERS_ROOT))

# Matplotlib relies on pyparsing APIs that emit deprecation warnings under pytest.
warnings.filterwarnings(
    "ignore",
    message=".*deprecated.*",
    module="matplotlib._fontconfig_pattern",
)
warnings.filterwarnings(
    "ignore",
    message=".*enablePackrat.*deprecated.*",
    module="matplotlib._mathtext",
)
