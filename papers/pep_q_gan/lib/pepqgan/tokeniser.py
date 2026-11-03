import torch


class PeptideTokenizer:
    """Tokenizer for peptide sequences."""

    def __init__(self, alphabet: str = "ACDEFGHIKLMNPQRSTVWY"):
        self.alphabet = alphabet
        # Create vocab with PAD=0, UNK=1, then amino acids
        self.char_to_idx = {"PAD": 0, "UNK": 1}
        for i, c in enumerate(alphabet):
            self.char_to_idx[c] = i + 2

        self.idx_to_char = {i: c for c, i in self.char_to_idx.items()}
        self.vocab_size = len(self.char_to_idx)
        self.pad_token = 0
        self.unk_token = 1

    def encode(self, sequence: str, max_length: int) -> torch.Tensor:
        """Convert sequence to indices and pad/truncate to max_length."""
        indices = []
        for char in sequence[:max_length]:
            if char in self.char_to_idx:
                indices.append(self.char_to_idx[char])
            else:
                # Unknown character
                indices.append(self.unk_token)

        # Pad to max_length
        indices = indices + [self.pad_token] * (max_length - len(indices))

        return torch.tensor(indices, dtype=torch.long)

    def decode(self, indices: torch.Tensor | list[int]) -> str:
        """Convert indices back to sequence."""
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()

        chars = []
        for idx in indices:
            if idx == self.pad_token:
                break  # Stop at first padding token
            elif idx in self.idx_to_char:
                char = self.idx_to_char[idx]
                if char != "PAD" and char != "UNK":
                    chars.append(char)
                elif char == "UNK":
                    chars.append("X")  # Use X for unknown
        return "".join(chars)

    def batch_encode(self, sequences: list[str], max_length: int) -> torch.Tensor:
        """Encode a batch of sequences."""
        return torch.stack([self.encode(seq, max_length) for seq in sequences])

    def __len__(self) -> int:
        """Return vocabulary size."""
        return self.vocab_size


class HLATokenizer:
    """Tokenizer for HLA sequences with pseudo-sequence mapping."""

    def __init__(
        self,
        alphabet: str = "ACDEFGHIKLMNPQRSTVWY",
        pseudo_mapping: dict[str, str] | None = None,
    ):
        self.alphabet = alphabet
        # Create vocab with PAD=0, UNK=1, then amino acids
        self.char_to_idx = {"PAD": 0, "UNK": 1}
        for i, c in enumerate(alphabet):
            self.char_to_idx[c] = i + 2

        self.idx_to_char = {i: c for c, i in self.char_to_idx.items()}
        self.vocab_size = len(self.char_to_idx)
        self.pad_token = 0
        self.unk_token = 1
        self.pseudo_mapping = pseudo_mapping or {}

    @classmethod
    def from_pseudo_file(cls, filepath: str, alphabet: str = "ACDEFGHIKLMNPQRSTVWY"):
        """Create HLA tokenizer from pseudo-sequence file."""
        pseudo_mapping = {}
        with open(filepath) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    allele, pseudo_seq = parts[0], parts[1]
                    pseudo_mapping[allele] = pseudo_seq
        return cls(alphabet=alphabet, pseudo_mapping=pseudo_mapping)

    def add_pseudo_mapping(self, allele: str, pseudo_seq: str):
        """Add a single HLA -> pseudo-sequence mapping."""
        self.pseudo_mapping[allele] = pseudo_seq

    def get_pseudo_sequence(self, hla_type: str) -> str:
        """Get pseudo-sequence for HLA type."""
        if hla_type not in self.pseudo_mapping:
            raise ValueError(
                f"HLA type '{hla_type}' not found in pseudo-sequence mapping"
            )
        return self.pseudo_mapping[hla_type]

    def encode(
        self, hla_type: str, max_length: int, use_pseudo: bool = True
    ) -> torch.Tensor:
        """Convert HLA type to indices using pseudo-sequence."""
        if use_pseudo:
            sequence = self.get_pseudo_sequence(hla_type)
        else:
            sequence = hla_type

        indices = []
        for char in sequence[:max_length]:
            if char in self.char_to_idx:
                indices.append(self.char_to_idx[char])
            else:
                # Unknown character
                indices.append(self.unk_token)

        # Pad to max_length
        indices = indices + [self.pad_token] * (max_length - len(indices))
        return torch.tensor(indices, dtype=torch.long)

    def decode(self, indices: torch.Tensor | list[int]) -> str:
        """Convert indices back to sequence."""
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()

        chars = []
        for idx in indices:
            if idx == self.pad_token:
                break  # Stop at first padding token
            elif idx in self.idx_to_char:
                char = self.idx_to_char[idx]
                if char != "PAD" and char != "UNK":
                    chars.append(char)
                elif char == "UNK":
                    chars.append("X")  # Use X for unknown
        return "".join(chars)

    def batch_encode(
        self, hla_types: list[str], max_length: int, use_pseudo: bool = True
    ) -> torch.Tensor:
        """Encode a batch of HLA types."""
        return torch.stack(
            [self.encode(hla_type, max_length, use_pseudo) for hla_type in hla_types]
        )

    def get_max_pseudo_length(self) -> int:
        """Get maximum length of all pseudo-sequences."""
        if not self.pseudo_mapping:
            return 0
        return max(len(seq) for seq in self.pseudo_mapping.values())

    def __len__(self) -> int:
        """Return vocabulary size."""
        return self.vocab_size


class DataProcessor:
    """High-level data processor that combines peptide and HLA tokenizers."""

    def __init__(
        self, peptide_tokenizer: PeptideTokenizer, hla_tokenizer: HLATokenizer
    ):
        self.peptide_tokenizer = peptide_tokenizer
        self.hla_tokenizer = hla_tokenizer

    @classmethod
    def from_files(cls, pseudo_file: str, alphabet: str = "ACDEFGHIKLMNPQRSTVWY"):
        """Create processor from pseudo-sequence file."""
        peptide_tokenizer = PeptideTokenizer(alphabet)
        hla_tokenizer = HLATokenizer.from_pseudo_file(pseudo_file, alphabet)
        return cls(peptide_tokenizer, hla_tokenizer)

    def read_data(self, file_path: str, skip_header: bool = True):
        """Read peptide data file."""
        seqs, y, hla = [], [], []
        with open(file_path) as f:
            if skip_header:
                f.readline()  # Skip header
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    seq, label, hla_type = parts[0], int(parts[1]), parts[2]
                    seqs.append(seq)
                    y.append(label)
                    hla.append(hla_type)
        return seqs, y, hla

    def filter_by_hla_types(
        self,
        seqs: list[str],
        y: list[int],
        hla: list[str],
        hla_types: list[str] | None = None,
    ):
        """Filter sequences by HLA types."""
        if not hla_types:
            return seqs, y, hla
        hla_set = set(hla_types)
        indices = [i for i, hla_type in enumerate(hla) if hla_type in hla_set]
        return (
            [seqs[i] for i in indices],
            [y[i] for i in indices],
            [hla[i] for i in indices],
        )

    def get_max_lengths(self, seqs: list[str], hla_types: list[str]):
        """Get maximum sequence lengths."""
        peptide_len = max(len(seq) for seq in seqs) if seqs else 0
        hla_len = self.hla_tokenizer.get_max_pseudo_length()
        return peptide_len, hla_len

    def process_dataset(
        self,
        seqs: list[str],
        y: list[int],
        hla_types: list[str],
        peptide_len: int,
        hla_len: int,
    ):
        """Process entire dataset into tensors."""
        # Encode peptides
        peptide_indices = self.peptide_tokenizer.batch_encode(seqs, peptide_len)
        # Encode HLA types
        hla_indices = self.hla_tokenizer.batch_encode(
            hla_types, hla_len, use_pseudo=True
        )
        # Convert labels
        labels = torch.tensor(y, dtype=torch.float32)
        return peptide_indices, labels, hla_indices
