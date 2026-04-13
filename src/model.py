"""Mini GPT model assembly for chapter 04."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .attention import CausalSelfAttention


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.0


class MLP(nn.Module):
    def __init__(self, n_embd: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(
            n_embd=config.n_embd,
            n_head=config.n_head,
            block_size=config.block_size,
            dropout=config.dropout,
        )
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config.n_embd, config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        self.token_embed = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_embed = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        bsz, seq_len = idx.shape
        if seq_len > self.config.block_size:
            raise ValueError("Input sequence length exceeds block size")

        pos = torch.arange(0, seq_len, device=idx.device, dtype=torch.long)
        x = self.token_embed(idx) + self.pos_embed(pos)
        x = self.drop(x)

        for block in self.blocks:
            x = block(x)

        logits = self.lm_head(self.ln_f(x))

        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )

        return logits, loss
