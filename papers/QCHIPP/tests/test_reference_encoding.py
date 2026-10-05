"""A fixed unencoded optical mode must expose relative data phases."""

import torch


def test_reference_mode_removes_the_all_encoded_global_phase_alias():
    from lib.photonic import (
        PhotonicQCNN,
        ReferencePhotonicQCNN,
        residue_angles,
    )

    torch.set_num_threads(1)
    torch.manual_seed(49)
    # Valid categorical codes A=1 and L=10; all local angles differ by pi/2.
    bits = torch.tensor([[0, 0, 0, 0, 1] * 4, [0, 1, 0, 1, 0] * 4], dtype=torch.float32)
    legacy = PhotonicQCNN().eval()
    reference = ReferencePhotonicQCNN().eval()
    angles = residue_angles(bits).reshape(-1, 6)
    with torch.no_grad():
        old = legacy(bits)
        probabilities = reference.stage1.quantum(angles)
    torch.testing.assert_close(old[0], old[1], atol=1e-6, rtol=1e-6)
    assert probabilities.shape == (4, 28)
    assert torch.linalg.vector_norm(probabilities[0] - probabilities[2]).item() > 1e-4
    assert sum(p.numel() for p in reference.parameters()) == 346
