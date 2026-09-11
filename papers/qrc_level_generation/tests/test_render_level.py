from __future__ import annotations

import sys

import numpy as np
import pytest
from common import PROJECT_DIR

if str(PROJECT_DIR / "utils") not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR / "utils"))

pytest.importorskip("PIL", reason="pillow not installed")

import render_level  # noqa: E402

pytestmark = pytest.mark.skipif(
    not render_level.LEVEL_IMAGE.exists(), reason="packaged level image missing"
)


def test_atlas_covers_all_features_and_is_consistent():
    # build_atlas raises on any column that disagrees with its feature's
    # first occurrence, so constructing it is itself the consistency check.
    atlas, sequence = render_level.build_atlas()
    assert len(atlas) == 32
    assert len(sequence) == 157


def test_render_original_roundtrip():
    from PIL import Image

    atlas, sequence = render_level.build_atlas()
    rendered = render_level.render(sequence, atlas)
    original = Image.open(render_level.LEVEL_IMAGE).convert("RGB")
    assert rendered.size == original.size
    assert np.array_equal(np.asarray(rendered), np.asarray(original))
