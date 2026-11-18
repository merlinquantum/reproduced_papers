"""Check bounded artifact-only retries and propagation of permanent failures."""

from pathlib import Path

import pytest
from lib.artifacts import atomic_replace


def windows_lock():
    """Return a Windows-style access conflict without requiring Windows."""
    error = PermissionError("simulated file access conflict")
    error.winerror = 5
    return error


def test_transient_lock_retries_the_same_file_and_keeps_atomic_content(
    tmp_path, monkeypatch
):
    source, target = tmp_path / "status.tmp", tmp_path / "status.json"
    source.write_text("new")
    target.write_text("old")
    original = Path.replace
    calls = []
    delays = []

    def locked(self, destination):
        calls.append((self, destination))
        if len(calls) < 3:
            assert target.read_text() == "old"
            raise windows_lock()
        return original(self, destination)

    monkeypatch.setattr(Path, "replace", locked)
    monkeypatch.setitem(atomic_replace.__globals__, "sleep", delays.append)
    atomic_replace(source, target)
    assert target.read_text() == "new"
    assert not source.exists()
    assert calls == [(source, target)] * 3
    assert delays == [0.05, 0.1]


def test_persistent_lock_is_bounded_and_preserves_both_artifacts(tmp_path, monkeypatch):
    source, target = tmp_path / "status.tmp", tmp_path / "status.json"
    source.write_text("new")
    target.write_text("old")
    calls, delays = [], []

    def locked(self, destination):
        calls.append(1)
        raise windows_lock()

    monkeypatch.setattr(Path, "replace", locked)
    monkeypatch.setitem(atomic_replace.__globals__, "sleep", delays.append)
    with pytest.raises(PermissionError):
        atomic_replace(source, target)
    assert len(calls) == 8
    assert sum(delays) == pytest.approx(2.25)
    assert source.read_text() == "new"
    assert target.read_text() == "old"


def test_other_errors_propagate_without_retries(tmp_path, monkeypatch):
    calls = []

    def denied(self, destination):
        calls.append(1)
        raise PermissionError("not a known Windows sharing conflict")

    monkeypatch.setattr(Path, "replace", denied)
    with pytest.raises(PermissionError):
        atomic_replace(tmp_path / "a", tmp_path / "b")
    assert len(calls) == 1
