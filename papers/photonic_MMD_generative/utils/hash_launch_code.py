"""Fingerprint the code copy of every remote launch of this paper (run on a GPU machine).

Usage (as a remote launch; the launch's working directory is its own repo copy)::

    python papers/photonic_MMD_generative/utils/hash_launch_code.py

Walks ``../../`` (the machine's ``launches/`` folder for this paper), and for
each ``<stamp>/repo/papers/photonic_MMD_generative`` hashes every file under
``lib/``, ``utils/`` and ``configs/`` (SHA-256), plus one combined digest per
folder. Writes ``outdir/launch_code_hashes_<hostname>.json`` in the current
launch's paper folder, which is copied back with the evidence.
"""

from __future__ import annotations

import hashlib
import json
import socket
from pathlib import Path

PAPER = "papers/photonic_MMD_generative"
FOLDERS = ("lib", "utils", "configs")


def digest(folder: Path) -> tuple[str, dict]:
    files = {}
    combined = hashlib.sha256()
    for path in sorted(
        p for p in folder.rglob("*") if p.is_file() and "__pycache__" not in p.parts
    ):
        h = hashlib.sha256(path.read_bytes()).hexdigest()
        rel = str(path.relative_to(folder))
        files[rel] = h
        combined.update(f"{rel}:{h}\n".encode())
    return combined.hexdigest(), files


def main() -> None:
    """Hash every launch copy reachable on this machine."""
    here = Path.cwd().resolve()
    launches = here.parent.parent  # .../launches/<stamp>/repo -> .../launches
    result = {"hostname": socket.gethostname(), "launches": {}}
    for repo in sorted(launches.glob("*/repo")):
        paper = repo / PAPER
        if not paper.is_dir():
            continue
        entry = {}
        for name in FOLDERS:
            if (paper / name).is_dir():
                combined, files = digest(paper / name)
                entry[name] = {"combined_sha256": combined, "files": files}
        result["launches"][repo.parent.name] = entry
    out = here / PAPER / "outdir" / f"launch_code_hashes_{socket.gethostname()}.json"
    out.write_text(json.dumps(result, indent=2))
    print(out, len(result["launches"]))


if __name__ == "__main__":
    main()
