"""Generate evaluation-attempt CSVs from verified public-format checkpoints."""

import json
from pathlib import Path

import pandas as pd
import torch

from .artifacts import sha256, write_json
from .data_setup import AMINO_ACIDS, load_evaluation_hlas, load_pseudosequences
from .devices import checked_device
from .gan import build_gan
from .legacy_priors import GaussianPrior
from .pepqgan.tokeniser import HLATokenizer, PeptideTokenizer
from .priors import FockPoolPrior


def generate_from_checkpoint(
    checkpoint: Path,
    data_root: Path,
    hla_file: Path,
    output_csv: Path,
    *,
    pool: Path | None = None,
    device: str = "cpu",
    attempts_per_hla: int = 1000,
    batch_size: int = 256,
) -> dict:
    """Decode every attempt while preserving invalid and duplicate sequences.

    Parameters
    ----------
    checkpoint : Path
        Public-format training checkpoint beside its ``config.json`` manifest.
    data_root : Path
        External training inputs matching the checkpoint's recorded hashes.
    hla_file : Path
        Ordered panel to generate; production uses the complete 131-HLA list.
    output_csv : Path
        New attempt CSV (optionally gzip) outside the curated results.
    pool : Path | None
        Matching completed photonic pool, or an explicit Gaussian pool. Default
        None uses fresh Gaussian draws only for a non-pool Gaussian checkpoint.
    device : str
        Explicit available CPU or CUDA device. Default is ``cpu``.
    attempts_per_hla : int
        Total attempts per HLA, including invalid outputs. Default is 1000.
    batch_size : int
        Decode batch size, recorded for replay. Default is 256.

    Returns
    -------
    dict
        Checkpoint/data/pool identities and attempt counts; scores are not computed.

    Notes
    -----
    A fresh prior starts once per checkpoint and continues across the ordered
    HLA panel. The advanced training cursor is not restored. This defines a new
    generation run, not byte-identical replay of archived GPU/CPU campaigns.
    """
    checkpoint, data_root, hla_file, output_csv = map(
        Path, (checkpoint, data_root, hla_file, output_csv)
    )
    if (
        min(attempts_per_hla, batch_size) < 1
        or type(attempts_per_hla) is not int
        or type(batch_size) is not int
    ):
        raise ValueError("Attempt and batch counts must be positive integers")
    if (
        output_csv.exists()
        or output_csv.with_suffix(output_csv.suffix + ".generation.json").exists()
    ):
        raise FileExistsError("Use a fresh generation output path")
    if output_csv.resolve().is_relative_to(
        Path(__file__).resolve().parents[1] / "results"
    ):
        raise ValueError("Raw generation outputs belong outside curated results")
    metadata = json.loads((checkpoint.parent / "config.json").read_text())
    for name, expected in metadata["data_files_sha256"].items():
        if Path(name).name != name or sha256(data_root / name) != expected:
            raise ValueError("Training data differs from checkpoint")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    saved_config = {
        key: value for key, value in state["config"].items() if key != "device"
    }
    recorded_config = {
        key: value for key, value in metadata["config"].items() if key != "device"
    }
    if saved_config != recorded_config or state["global_step"] < 1:
        raise ValueError("Checkpoint config/step does not match its training manifest")
    prior_name, seed = state["config"]["prior"], state["config"]["seed"]
    step = state["global_step"]
    if prior_name in ("dist", "indist"):
        if pool is None:
            raise ValueError("Photonic generation requires the matching completed pool")
        prior = FockPoolPrior(pool)
        if (
            prior.condition != prior_name
            or state["prior_state"]["metadata"] != prior.metadata
        ):
            raise ValueError("Generation pool differs from checkpoint")
    elif prior_name == "gaussian":
        if "prior_state" in state:
            from .classical_pools import ClassicalPoolPrior

            if pool is None:
                raise ValueError("This Gaussian checkpoint requires its explicit pool")
            prior = ClassicalPoolPrior(pool, "gaussian")
            if state["prior_state"]["metadata"] != prior.metadata:
                raise ValueError("Generation pool differs from checkpoint")
        else:
            if pool is not None:
                raise ValueError(
                    "Fresh-draw Gaussian checkpoint cannot switch to a pool"
                )
            prior = GaussianPrior(seed=0)
    else:
        raise ValueError("Unsupported checkpoint prior")
    hlas = load_evaluation_hlas(hla_file)
    mapping = load_pseudosequences(data_root / "MHC_pseudo_HLA_only.dat", set(hlas))
    if any(
        h not in mapping or len(mapping[h]) != 34 or not set(mapping[h]) <= AMINO_ACIDS
        for h in hlas
    ):
        raise ValueError("Required HLA conditioning input is missing or invalid")
    hla_tokens, peptide_tokens = (
        HLATokenizer(pseudo_mapping=mapping),
        PeptideTokenizer(),
    )
    selected_device = checked_device(device)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    generator, critic = build_gan(seed, "cpu")
    del critic
    generator.load_state_dict(state["generator_state_dict"])
    generator.to(selected_device).eval()
    del state
    rows = []
    with torch.inference_mode():
        for hla in hlas:
            for offset in range(0, attempts_per_hla, batch_size):
                size = min(batch_size, attempts_per_hla - offset)
                condition = (
                    hla_tokens.encode(hla, 34).repeat(size, 1).to(selected_device)
                )
                tokens = (
                    generator(prior.sample(size, device=selected_device), condition)
                    .argmax(-1)
                    .cpu()
                )
                rows.extend(
                    {
                        "prior": prior_name,
                        "seed": seed,
                        "hla": hla,
                        "attempt": offset + i + 1,
                        "peptide": peptide_tokens.decode(token),
                    }
                    for i, token in enumerate(tokens)
                )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    result = {
        "status": "complete",
        "prior": prior_name,
        "seed": seed,
        "training_step": step,
        "hlas": len(hlas),
        "attempts_per_hla": attempts_per_hla,
        "generated": len(rows),
        "batch_size": batch_size,
        "device": str(selected_device),
        "checkpoint_sha256": sha256(checkpoint),
        "data_sha256": metadata["data_files_sha256"],
        "hla_file_sha256": sha256(hla_file),
        "pool_metadata": prior.metadata if hasattr(prior, "metadata") else None,
        "prior_restart": "Fresh pool order/private CPU Gaussian seed 0; reset once, continue across ordered HLA panel",
        "invalid_and_duplicate_attempts_retained": True,
        "predictor_run": False,
        "output_sha256": sha256(output_csv),
    }
    write_json(output_csv.with_suffix(output_csv.suffix + ".generation.json"), result)
    return result
