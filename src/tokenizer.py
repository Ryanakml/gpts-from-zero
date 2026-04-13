"""Tokenizer utilities for chapter 02 experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import tiktoken


@dataclass
class CharTokenizer:
    """A tiny character-level tokenizer for educational purposes."""

    vocab: str

    def __post_init__(self) -> None:
        self.stoi = {ch: i for i, ch in enumerate(self.vocab)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

    def encode(self, text: str) -> List[int]:
        return [self.stoi[ch] for ch in text if ch in self.stoi]

    def decode(self, ids: List[int]) -> str:
        return "".join(self.itos[i] for i in ids if i in self.itos)


class GPT2Tokenizer:
    """Thin wrapper around tiktoken's GPT-2 encoding."""

    def __init__(self) -> None:
        self.enc = tiktoken.get_encoding("gpt2")

    def encode(self, text: str) -> List[int]:
        return self.enc.encode(text)

    def decode(self, ids: List[int]) -> str:
        return self.enc.decode(ids)
