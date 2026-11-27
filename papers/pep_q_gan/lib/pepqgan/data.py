from torch import Tensor
from torch.utils.data import Dataset

from .tokeniser import HLATokenizer, PeptideTokenizer


class PeptideDataset(Dataset):
    """Dataset for peptide sequences with HLA context."""

    def __init__(
        self,
        peptides: list[str],
        labels: list[int],
        hla_types: list[str],
        peptide_tokenizer: PeptideTokenizer,
        hla_tokenizer: HLATokenizer,
        pep_seq_len: int,
        hla_seq_len: int,
    ):
        self.peptides = peptides
        self.labels = labels
        self.hla_types = hla_types
        self.peptide_tokenizer = peptide_tokenizer
        self.hla_tokenizer = hla_tokenizer
        self.pep_seq_len = pep_seq_len
        self.hla_seq_len = hla_seq_len

    def __len__(self) -> int:
        return len(self.peptides)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor, str]:
        pep_encoded = self.peptide_tokenizer.encode(
            self.peptides[idx], self.pep_seq_len
        )
        hla_encoded = self.hla_tokenizer.encode(
            self.hla_types[idx], self.hla_seq_len, use_pseudo=True
        )
        label_tensor = Tensor([self.labels[idx]]).float()

        return pep_encoded, label_tensor, hla_encoded, self.hla_types[idx]
