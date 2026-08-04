"""Checks for opt-in external data materialisation."""

from __future__ import annotations

from lib import data as data_mod


def test_deepship_github_sample_download_is_explicit(tmp_path, monkeypatch):
    calls: list[tuple[str, str]] = []

    def fake_urlretrieve(url, filename):
        calls.append((url, str(filename)))
        filename.write_bytes(b"RIFF")

    monkeypatch.setattr(data_mod, "urlretrieve", fake_urlretrieve)

    data_mod.ensure_deepship_github_sample(tmp_path)

    assert len(calls) == 8
    assert (tmp_path / "deepship" / "F" / "41.wav").exists()
    assert calls[0][0].startswith(data_mod.DEEPSHIP_GITHUB_BASE_URL)
