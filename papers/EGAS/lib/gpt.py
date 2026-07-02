"""Minimal autoregressive GPT generator over the circuit-token vocabulary.

Follows the generative-quantum-eigensolver (GQE) scheme cited by the paper (ref [40]).
The GPT defines a distribution over length-D token sequences; ``p_theta(s) ~ exp(-gamma *
w_sum(s; theta))`` where ``w_sum`` is the cumulative logit score of the chosen tokens
(Eq. 10).  Sequences are sampled autoregressively with a temperature ``T`` and the model is
trained so that ``exp(-gamma * w_sum)`` matches ``exp(-gamma * E)`` (logit matching), biasing
sampling toward low-energy circuits.

A start token (id = vocab_size - 1) seeds generation; the first D positions emit circuit
tokens (ids 0..|C|-1).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TokenGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        d_model: int = 64,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size  # includes start token
        self.seq_len = seq_len
        self.start_token = vocab_size - 1
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len + 1, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size - 1)  # emit only circuit tokens

    def _causal_mask(self, length: int, device) -> torch.Tensor:
        return torch.triu(
            torch.full((length, length), float("-inf"), device=device), diagonal=1
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (B, L) input ids -> logits (B, L, |C|) over circuit tokens for next position."""
        B, L = tokens.shape
        pos = torch.arange(L, device=tokens.device).unsqueeze(0)
        h = self.tok_emb(tokens) + self.pos_emb(pos)
        h = self.blocks(h, mask=self._causal_mask(L, tokens.device))
        return self.head(self.ln(h))

    @torch.no_grad()
    def sample(self, n_samples: int, temperature: float, device="cpu"):
        """Autoregressively sample n_samples sequences. Returns LongTensor (n_samples, D)."""
        tokens = torch.full(
            (n_samples, 1), self.start_token, dtype=torch.long, device=device
        )
        for _ in range(self.seq_len):
            logits = self.forward(tokens)[:, -1, :]  # (n, |C|)
            probs = torch.softmax(logits / max(temperature, 1e-6), dim=-1)
            nxt = torch.multinomial(probs, 1)  # (n, 1)
            tokens = torch.cat([tokens, nxt], dim=1)
        return tokens[:, 1:]  # drop start token

    def w_sum(self, sequences: torch.Tensor) -> torch.Tensor:
        """Cumulative logit score sum_d logit(c_d | c_<d) for each sequence (grad-enabled).

        sequences: (B, D) circuit-token ids.
        """
        B, D = sequences.shape
        start = torch.full(
            (B, 1), self.start_token, dtype=torch.long, device=sequences.device
        )
        inp = torch.cat([start, sequences[:, :-1]], dim=1)  # teacher-forcing input
        logits = self.forward(inp)  # (B, D, |C|)
        chosen = torch.gather(logits, 2, sequences.unsqueeze(-1)).squeeze(-1)  # (B, D)
        return chosen.sum(dim=1)  # (B,)
