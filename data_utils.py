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


def prepare_formatted_dataset(dataset_name: str = "iwslt2017",
                              size: int = 1000,
                              noise_rate: float = 0.0,
                              seed: int = 42):
    """
    Loads an already-formatted JSONL dataset from formatted_data/.

    This function keeps the same call signature as prepare_dataset, so callers
    can replace `prepare_dataset(...)` with `prepare_formatted_dataset(...)`
    without changing the argument flow.

    Expected formatted rows:
      - Gigaword: {"noisy_document": "...", "summary": "..."}
      - IWSLT:    {"noisy_source": "...", "target": "..."}

    Important: formatted noisy data is split by explicit noise type. With
    noise_rate > 0, dataset_name must include a suffix such as
    "gigaword_swap", "gigaword_delete", or "gigaword_replace". A plain
    "gigaword" + noise_rate is intentionally rejected because there is no
    single mixed-noise formatted file.

    Returns: (dataset, src_col, tgt_col)
    """
    from pathlib import Path

    _ = seed
    noise_types = ("swap", "delete", "replace")
    dataset_aliases = {
        "gigaword": "gigaword",
        "iwslt": "iwslt",
        "iwslt2017": "iwslt",
    }
    formatted_columns = {
        "gigaword": ("document", "summary", "noisy_document"),
        "iwslt": ("source", "target", "noisy_source"),
    }

    explicit_path = Path(dataset_name)
    if explicit_path.suffix == ".jsonl":
        if not explicit_path.exists():
            raise FileNotFoundError(f"Preprocessed dataset file not found: {explicit_path}")
        data_file = explicit_path
    else:
        noise_type = None
        base_name = dataset_name
        for candidate in noise_types:
            suffix = f"_{candidate}"
            if dataset_name.endswith(suffix):
                noise_type = candidate
                base_name = dataset_name[:-len(suffix)]
                break

        if base_name not in dataset_aliases:
            supported = sorted(dataset_aliases)
            raise ValueError(
                f"Unsupported dataset: {dataset_name}. Expected one of {supported}, "
                f"optionally suffixed with _{'/_'.join(noise_types)}."
            )

        dataset_key = dataset_aliases[base_name]
        formatted_dir = Path(__file__).resolve().parent / "formatted_data" / dataset_key
        if not formatted_dir.exists():
            raise FileNotFoundError(f"Formatted dataset directory not found: {formatted_dir}")

        if noise_rate <= 0.0:
            data_file = formatted_dir / f"train_clean_{size}.jsonl"
        else:
            if noise_type is None:
                raise ValueError(
                    f"noise_rate={noise_rate} requires an explicit noise type suffix. "
                    f"Use one of: {base_name}_swap, {base_name}_delete, "
                    f"{base_name}_replace."
                )
            ratio_tag = f"{int(round(noise_rate * 100)):02d}"
            data_file = formatted_dir / f"train_noisy_{size}_{noise_type}_{ratio_tag}.jsonl"

        if not data_file.exists():
            raise FileNotFoundError(f"Preprocessed dataset file not found: {data_file}")

    dataset = load_dataset("json", data_files=str(data_file), split="train")
    column_names = set(dataset.column_names)
    for src_col, tgt_col, noisy_col in formatted_columns.values():
        if noisy_col in column_names and tgt_col in column_names:
            return dataset, src_col, tgt_col

    raise ValueError(
        "Formatted dataset has unsupported columns. Expected either "
        "{'noisy_document', 'summary'} for Gigaword or "
        "{'noisy_source', 'target'} for IWSLT. "
        f"Found: {dataset.column_names}"
    )


if __name__ == "__main__":
    ds, src, tgt = prepare_dataset(size=5, noise_rate=0.3)
    print("Sample (30% noise):")
    for i in range(2):
        print("  Original:", ds[i][src])
        print("  Noisy   :", ds[i][f"noisy_{src}"])
        print("  Target  :", ds[i][tgt])
        print("-" * 60)
