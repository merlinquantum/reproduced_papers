from __future__ import annotations

import logging
from pathlib import Path

from .qrb_circuit import QRBCircuitSpec, build_qrb_circuit

logger = logging.getLogger(__name__)


def _build_qrb_circuit(*, kd: int, kh: int):

    kd = int(kd)
    kh = int(kh)
    m = kd + kh
    if kd <= 0 or kh < 0 or m <= 0:
        raise ValueError(f"Invalid kd/kh: kd={kd} kh={kh}")

    return build_qrb_circuit(QRBCircuitSpec(m=m, k=kd))


def _build_qrb_experiment(*, circuit, kd: int, kh: int):
    import perceval as pcvl  # noqa: WPS433

    exp = pcvl.Experiment(circuit)
    m = int(kd) + int(kh)
    exp.with_input(pcvl.BasicState([1, 0] * m))
    for mode in range(circuit.m):
        exp.add(mode, pcvl.Detector.threshold())
    return exp


def export_qrb_circuit_svg(
    *,
    run_dir: Path,
    kd: int,
    kh: int,
    filename: str = "qrb_circuit.svg",
    recursive: bool = True,
) -> Path | None:
    """Render the QRB Perceval circuit used by the photonic QRNN as an SVG.

    Perceval 1.1.0 does not expose `Format.SVG`; however `Format.HTML` returns a
    `DrawsvgWrapper` (for experiments) that contains a `drawsvg.Drawing` object
    that can be saved as an SVG.

    The function is best-effort: it logs and returns None on failure.
    """

    try:
        from perceval.rendering.format import Format  # noqa: WPS433
        from perceval.rendering.pdisplay import pdisplay_experiment  # noqa: WPS433

        circuit = _build_qrb_circuit(kd=kd, kh=kh)
        exp = _build_qrb_experiment(circuit=circuit, kd=kd, kh=kh)

        out_path = (run_dir / filename).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        wrapper = pdisplay_experiment(
            exp,
            output_format=Format.HTML,
            recursive=bool(recursive),
        )
        drawing = getattr(wrapper, "value", None)
        if drawing is None or not hasattr(drawing, "save_svg"):
            raise TypeError(
                f"Unexpected pdisplay HTML result; expected drawsvg.Drawing, got {type(drawing)}"
            )

        drawing.save_svg(str(out_path))
        return out_path
    except Exception:
        logger.warning("Failed to export QRB circuit SVG", exc_info=True)
        return None
