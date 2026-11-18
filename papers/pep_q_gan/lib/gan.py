"""Build the author's conditional GAN and its training dataset locally.

Model defaults follow upstream conf/models/models.yaml and config.yaml.
No imports from the original checkout, no optimization or evaluation data.
"""

from pathlib import Path

import torch

from .data_setup import load_pseudosequences, load_training_data
from .pepqgan.data import PeptideDataset
from .pepqgan.model import Discriminator, Generator
from .pepqgan.tokeniser import HLATokenizer, PeptideTokenizer


def load_training_dataset(data_dir: str | Path) -> PeptideDataset:
    data_dir = Path(data_dir)
    records = load_training_data(data_dir / "HLA_pos9mers.txt")
    mapping = load_pseudosequences(
        data_dir / "MHC_pseudo_HLA_only.dat", {row.hla for row in records}
    )
    if any(len(row.peptide) != 9 or len(mapping[row.hla]) != 34 for row in records):
        raise ValueError(
            "The upstream GAN requires 9-residue peptides and 34-residue HLA pseudosequences."
        )
    return PeptideDataset(
        peptides=[row.peptide for row in records],
        labels=[row.label for row in records],
        hla_types=[row.hla for row in records],
        peptide_tokenizer=PeptideTokenizer(),
        hla_tokenizer=HLATokenizer(pseudo_mapping=mapping),
        pep_seq_len=9,
        hla_seq_len=34,
    )


def build_gan(
    seed: int = 46, device: str | torch.device = "cpu"
) -> tuple[Generator, Discriminator]:
    """Original 512-wide architecture; initialization does not reset global RNG."""
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        generator = Generator(
            z_dim=32,
            seq_len=9,
            hla_len=34,
            alphabet_size=22,
            hidden_size=512,
            embedding_dim=32,
            num_map_layers=3,
        )
        critic = Discriminator(
            seq_len=9, hla_len=34, alphabet_size=22, hidden_size=512, embedding_dim=32
        )
    return generator.to(device), critic.to(device)
