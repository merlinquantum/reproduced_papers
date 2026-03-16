from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_cli_schema():
    schema_path = PROJECT_DIR / "cli.json"
    if not schema_path.exists():
        schema_path = PROJECT_DIR / "configs" / "cli.json"
    with schema_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
