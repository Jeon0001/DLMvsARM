"""
noise_utils.py — Shared word-level corruption module for DLM vs ARM RQ2.

Used by both Gigaword (Jimin) and IWSLT (Yohan) preprocessing pipelines so the
two datasets receive *identical* noise treatment.

Policy (agreed 2026-05-07):
  - Source-only corruption; target stays clean.
  - Three noise types are NEVER mixed in one sentence; we generate separate
    files per type (swap-only / delete-only / replace-only).
  - Word-level (whitespace split). Punctuation stays attached to the word.
  - Sentences shorter than MIN_WORDS (5) are returned unchanged.
  - Delete keeps at least one word.
  - Replace samples real words from a provided in-domain train vocabulary
    (NOT random ASCII), excluding the lowest-frequency tail.
  - All randomness goes through a caller-supplied random.Random instance so
    runs are reproducible per (dataset, subset_size, noise_type, ratio, seed).
"""

from __future__ import annotations

import random
from collections import Counter
from typing import Iterable, Sequence

MIN_WORDS = 5
NOISE_TYPES = ("swap", "delete", "replace")


def build_vocab(texts: Iterable[str],
                min_count: int = 2,
                drop_top_frac: float = 0.0) -> list[str]:
    """Word-level vocab from a corpus, with optional frequency cuts.

    `min_count`     -> drop tail (typos, OOV) so Replace samples real words.
    `drop_top_frac` -> drop the top X fraction (stop words) to make Replace
                       more disruptive. Default 0 keeps everything above
                       `min_count`.
    """
    counter: Counter[str] = Counter()
    for t in texts:
        counter.update(t.split())
    items = [(w, c) for w, c in counter.items() if c >= min_count]
    items.sort(key=lambda x: -x[1])
    if drop_top_frac > 0:
        cut = int(len(items) * drop_top_frac)
        items = items[cut:]
    return [w for w, _ in items]


def _swap(words: list[str], ratio: float, rng: random.Random) -> list[str]:
    """Swap each eligible position with its right neighbor with prob `ratio`.
    A swapped position is skipped on the next step (no double-swap)."""
    out = words[:]
    i = 0
    while i < len(out) - 1:
        if rng.random() < ratio:
            out[i], out[i + 1] = out[i + 1], out[i]
            i += 2
        else:
            i += 1
    return out


def _delete(words: list[str], ratio: float, rng: random.Random) -> list[str]:
    kept = [w for w in words if rng.random() >= ratio]
    if not kept:
        kept = [rng.choice(words)]
    return kept


def _replace(words: list[str], ratio: float, rng: random.Random,
             vocab: Sequence[str]) -> list[str]:
    if not vocab:
        raise ValueError("Replace noise requires a non-empty vocab.")
    return [rng.choice(vocab) if rng.random() < ratio else w for w in words]


def corrupt(text: str,
            noise_type: str,
            ratio: float,
            rng: random.Random,
            vocab: Sequence[str] | None = None) -> str:
    """Apply ONE noise type at the given ratio to a single text.

    Returns the original text when ratio <= 0 or word count < MIN_WORDS.
    """
    if ratio <= 0:
        return text
    words = text.split()
    if len(words) < MIN_WORDS:
        return text
    if noise_type == "swap":
        out = _swap(words, ratio, rng)
    elif noise_type == "delete":
        out = _delete(words, ratio, rng)
    elif noise_type == "replace":
        out = _replace(words, ratio, rng, vocab or [])
    else:
        raise ValueError(
            f"Unknown noise_type={noise_type!r}; expected one of {NOISE_TYPES}"
        )
    return " ".join(out)


def make_rng(dataset: str, subset_size: int, noise_type: str,
             ratio: float, seed: int = 42) -> random.Random:
    """Deterministic per-cell RNG so each (dataset, size, type, ratio) cell is
    independently reproducible without coupling to a global random state."""
    key = f"{dataset}|{subset_size}|{noise_type}|{ratio:.3f}|{seed}"
    return random.Random(key)
