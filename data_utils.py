"""
data_utils.py — Fair DLM vs ARM Comparison
==========================================
Identical data pipeline for both models. No asymmetric treatment.
"""

import random
import string
from datasets import load_dataset


def get_random_word():
    return ''.join(random.choices(string.ascii_lowercase, k=random.randint(3, 8)))


def apply_word_level_noise(text: str, noise_rate: float = 0.0) -> str:
    """
    Applies word-level corruption (delete / replace / swap) at the given rate.
    Applied identically to both AR and DLM source sides.
    """
    if noise_rate == 0.0:
        return text
    words = text.split()
    if not words:
        return text
    new_words = []
    i = 0
    while i < len(words):
        if random.random() < noise_rate:
            action = random.choice(['delete', 'replace', 'swap'])
            if action == 'delete':
                pass
            elif action == 'replace':
                new_words.append(get_random_word())
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


def prepare_dataset(dataset_name: str = "iwslt2017",
                    size: int = 1000,
                    noise_rate: float = 0.0,
                    seed: int = 42):
    """
    Loads dataset, shuffles with a fixed seed, selects `size` examples,
    and applies word-level noise to the source column.

    Returns: (dataset, src_col, tgt_col)
    """
    if dataset_name == "gigaword":
        dataset = load_dataset("gigaword", split="train", trust_remote_code=True)
        text_col, target_col = "document", "summary"
    elif dataset_name == "iwslt2017":
        dataset = load_dataset("iwslt2017", "iwslt2017-en-de", split="train",
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

    # Fixed seed → identical splits for AR and DLM
    dataset = dataset.shuffle(seed=seed).select(range(size))

    def add_noise(example):
        example[f"noisy_{text_col}"] = apply_word_level_noise(
            example[text_col], noise_rate
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
