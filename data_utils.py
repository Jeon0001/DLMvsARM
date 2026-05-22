"""
data_utils.py — Fair DLM vs ARM Comparison
==========================================
Identical data pipeline for both models. No asymmetric treatment.
"""

import random
import string
from pathlib import Path

from datasets import load_dataset


def get_random_word(rng=None):
    rng = rng or random
    return ''.join(rng.choices(string.ascii_lowercase, k=rng.randint(3, 8)))


def apply_word_level_noise(text: str, noise_rate: float = 0.0, rng=None) -> str:
    """
    Applies word-level corruption (delete / replace / swap) at the given rate.
    Applied identically to both AR and DLM source sides.
    """
    rng = rng or random
    if noise_rate == 0.0:
        return text
    words = text.split()
    if not words:
        return text
    new_words = []
    i = 0
    while i < len(words):
        if rng.random() < noise_rate:
            action = rng.choice(['delete', 'replace', 'swap'])
            if action == 'delete':
                pass
            elif action == 'replace':
                new_words.append(get_random_word(rng))
            elif action == 'swap':
                if i < len(words) - 1:
                    new_words.append(words[i + 1])
                    new_words.append(words[i])
                    i += 1
                else:
                    new_words.append(words[i])
        else:
            new_words.append(words[i])
        i += 1
    return " ".join(new_words)


def local_gigaword_csv_path(data_dir: str, split: str, size: int) -> Path:
    """Return the CSV file for a local Gigaword split/subset."""
    base = Path(data_dir).expanduser()
    split = split.lower()

    if split == "train":
        filename = f"train_{size}.csv"
    elif split in {"validation", "val", "dev"}:
        filename = "validation_1000.csv"
    elif split in {"test", "test_full"}:
        filename = "test_full.csv"
    elif split.endswith(".csv"):
        candidate = Path(split).expanduser()
        path = candidate if candidate.is_absolute() else base / candidate
        if path.exists():
            return path
        raise FileNotFoundError(f"Local CSV split file not found: {path}")
    else:
        filename = f"{split}.csv"

    path = base / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Local Gigaword CSV not found: {path}. "
            "Use split=train, validation, test, or pass an explicit CSV filename."
        )
    return path


def prepare_dataset(dataset_name: str = "iwslt2017",
                    size: int = 1000,
                    noise_rate: float = 0.0,
                    seed: int = 42,
                    split: str = "train",
                    data_dir: str | None = None):
    """
    Loads a dataset split, selects `size` examples, and applies word-level
    noise to the source column. If `data_dir` is provided for gigaword, reads
    local CSV subsets instead of Hugging Face.

    Returns: (dataset, src_col, tgt_col)
    """
    rng = random.Random(seed)

    if data_dir:
        if dataset_name != "gigaword":
            raise ValueError("Local CSV data_dir is currently supported for dataset='gigaword' only.")
        csv_path = local_gigaword_csv_path(data_dir, split, size)
        dataset = load_dataset("csv", data_files=str(csv_path), split="train")
        text_col, target_col = "document", "summary"
    elif dataset_name == "gigaword":
        dataset = load_dataset("gigaword", split=split, trust_remote_code=True)
        text_col, target_col = "document", "summary"
    elif dataset_name == "iwslt2017":
        dataset = load_dataset("iwslt2017", "iwslt2017-en-de", split=split,
                               trust_remote_code=True)

        def extract_translation(example):
            return {
                "source": example["translation"]["en"],
                "target": example["translation"]["de"],
            }

        dataset = dataset.map(extract_translation, remove_columns=["translation"])
        text_col, target_col = "source", "target"
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # Fixed seed -> identical HF splits for AR and DLM. Local CSV files are
    # already pre-split/subsetted, so preserve their row order.
    if data_dir:
        dataset = dataset.select(range(min(size, len(dataset))))
    else:
        dataset = dataset.shuffle(seed=seed).select(range(size))

    def add_noise(example):
        example[f"noisy_{text_col}"] = apply_word_level_noise(
            example[text_col], noise_rate, rng
        )
        return example

    dataset = dataset.map(add_noise)
    return dataset, text_col, target_col


if __name__ == "__main__":
    ds, src, tgt = prepare_dataset(size=5, noise_rate=0.3)
    print("Sample (30% noise):")
    for i in range(2):
        print("  Original:", ds[i][src])
        print("  Noisy   :", ds[i][f"noisy_{src}"])
        print("  Target  :", ds[i][tgt])
        print("-" * 60)
