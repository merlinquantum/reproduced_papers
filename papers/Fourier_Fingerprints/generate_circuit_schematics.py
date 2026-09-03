from pathlib import Path

from lib.fourier_1D import PhotonicSpectralModel
from perceval.rendering.pdisplay import pdisplay_to_file

results_dir = Path(__file__).resolve().parent / "results"
results_dir.mkdir(exist_ok=True)

for circuit_index in range(4):
    model = PhotonicSpectralModel(circuit_index=circuit_index)
    pdisplay_to_file(
        model.quantum_layer.circuit,
        str(results_dir / f"circuit_{circuit_index}.png"),
        recursive=True,
    )
