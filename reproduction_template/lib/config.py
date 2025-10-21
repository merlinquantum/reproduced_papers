import json
import logging
from pathlib import Path


def load_config(path: Path):
    """Load a JSON config file into a dict.

    Only JSON is supported in this template.
    The description field contains information to be displayed to the user.
    """
    ext = path.suffix.lower()
    logger = logging.getLogger(__name__)
    logger.info("Loading config file:%s", str(path))
    with path.open("r") as f:
        if ext != ".json":
            raise ValueError(
                f"Unsupported config extension for template (use JSON): {ext}"
            )
        j = json.load(f)
        if "description" not in j:
            raise ValueError("Missing 'description' field in JSON.")
        logger.info(" JSON Description:%s", j["description"])
        return j


def deep_update(base, updates):
    """Recursively update dict `base` with `updates`. Returns the updated dict."""
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = deep_update(base[k], v)
        else:
            base[k] = v
    return base
