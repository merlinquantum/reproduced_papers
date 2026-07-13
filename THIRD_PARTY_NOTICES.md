# Third-Party Notices

This repository contains original code released under the MIT License in the
root `LICENSE` file. It also includes or adapts material from third-party
projects.

This file is informational and does not replace notices in individual source
files or subdirectories. If a file has its own license header, or a directory
ships with its own `LICENSE` or `NOTICE` file, that notice controls for the
material it covers.

## Major upstream projects

### Qiskit

- Upstream: <https://github.com/Qiskit/qiskit>
- License: Apache License 2.0
- Notes: some gate-based reproductions may include code adapted from Qiskit or
  preserve Qiskit-compatible structures. Keep Apache notices where present.

### PennyLane

- Upstream: <https://github.com/PennyLaneAI/pennylane>
- License: Apache License 2.0
- Notes: some gate-based reproductions may include code adapted from PennyLane
  or preserve PennyLane-compatible structures. Keep Apache notices where
  present.

### MerLin

- Upstream: <https://github.com/merlinquantum/merlin>
- License: MIT
- Notes: this repository is part of the MerLin ecosystem and includes code
  written for or adapted from MerLin.

## Additional embedded or legacy material

- Some archived or legacy directories include their own `LICENSE` files. Those
  licenses continue to apply within those directories.
- Example: `papers/qSSL/lib/qnn/` contains files that retain an MIT header from
  upstream material, and `legacy/` contains bundled code with directory-local
  licenses.
- Example: some files under `papers/shared/` retain Apache 2.0 headers from
  upstream material.

## Practical rule

- The root `LICENSE` applies to original repository code unless another notice
  says otherwise.
- Per-file headers and directory-local `LICENSE` or `NOTICE` files apply to the
  specific third-party material they cover.
- When redistributing this repository, preserve the root MIT license and all
  applicable third-party notices.
