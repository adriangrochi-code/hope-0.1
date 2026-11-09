"""Línea base convencional para comparar con HN-1 (ARQUITECTURA_HN1 §3.2 y §11 F1).

Transformer decoder estándar sobre tokens BPE (tokenizador propio entrenado con
el mismo corpus), con un tamaño parecido al de HN-1. Se compara en **bits por
byte (BPB)**, la única métrica justa entre modelos con distinta unidad de entrada.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hope.core.layers import Block, RMSNorm

LN2 = 0.6931471805599453


@dataclass
class BaselineConfig:
    vocab: int = 8192
    d: int = 384
    heads: int = 6
    mlp_hidden: int = 1408
    layers: int = 8


class TokenLM(nn.Module):
    def __init__(self, cfg: BaselineConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab, cfg.d)
        self.blocks = nn.ModuleList(Block(cfg.d, "attn", cfg.heads, cfg.mlp_hidden) for _ in range(cfg.layers))
        self.norm = RMSNorm(cfg.d)
        self.head = nn.Linear(cfg.d, cfg.vocab, bias=False)

    def forward(self, ids):
        h = self.embed(ids)
        for b in self.blocks:
            h = b(h)
        return self.head(self.norm(h))

    def loss(self, ids: torch.Tensor, token_bytes: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """``token_bytes[v]`` = bytes UTF-8 que representa el token v (para calcular BPB)."""
        logits = self.forward(ids[:, :-1])
        target = ids[:, 1:]
        ce_sum = F.cross_entropy(logits.float().reshape(-1, self.cfg.vocab), target.reshape(-1), reduction="sum")
        n_bytes = token_bytes[target].sum()
        ce = ce_sum / target.numel()
        return ce, {"ce": ce.item(), "bpb": (ce_sum / n_bytes).item() / LN2}


# --- tokenizador BPE ---------------------------------------------------------
def train_tokenizer(train_bin: str | Path, out_dir: str | Path, vocab: int = 8192,
                    max_bytes: int = 200_000_000):
    """Entrena un BPE a nivel de bytes con el corpus de entrenamiento (requiere ``tokenizers``)."""
    from tokenizers import ByteLevelBPETokenizer
    raw = Path(train_bin).read_bytes()[:max_bytes]
    docs = [d.decode("utf-8", errors="replace") for d in raw.split(b"\x00") if d]
    tok = ByteLevelBPETokenizer()
    tok.train_from_iterator(docs, vocab_size=vocab, min_frequency=2, special_tokens=["<eod>"])
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tok.save(str(out / "bpe.json"))
    return tok


def load_tokenizer(path: str | Path):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(path))


def encode_bin(tok, src_bin: str | Path, dst: str | Path) -> np.ndarray:
    """Tokeniza un ``.bin`` de bytes y guarda los ids como uint16."""
    raw = Path(src_bin).read_bytes()
    eod = tok.token_to_id("<eod>")
    ids: list[int] = []
    for doc in raw.split(b"\x00"):
        if doc:
            ids.extend(tok.encode(doc.decode("utf-8", errors="replace")).ids)
            ids.append(eod)
    arr = np.asarray(ids, dtype=np.uint16)
    arr.tofile(dst)
    return arr


def token_byte_lengths(tok) -> torch.Tensor:
    """Bytes UTF-8 que aporta cada token (el especial ``<eod>`` cuenta como 1, igual que el 0x00)."""
    vocab = tok.get_vocab_size()
    lengths = torch.zeros(vocab, dtype=torch.float32)
    for i in range(vocab):
        lengths[i] = max(len(tok.decode([i]).encode("utf-8")), 1)
    return lengths
