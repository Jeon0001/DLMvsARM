"""
04_format_preprocessed.py - Build formatted JSONL files for training/eval.

Reads existing preprocessed JSONL files:
  data/gigaword/*.jsonl  with {document, summary}
  data/iwslt/*.jsonl     with {source, target}

Writes formatted copies:
  formatted_data/gigaword/*.jsonl  with {noisy_document, summary}
  formatted_data/iwslt/*.jsonl     with {noisy_source, target}

Only files that already exist are converted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data"

DATASETS = {
    "gigaword": {
        "input_columns": ("document", "summary"),
        "output_columns": ("noisy_document", "summary"),
        "output_dir": "gigaword",
    },
    "iwslt": {
        "input_columns": ("source", "target"),
        "output_columns": ("noisy_source", "target"),
        "output_dir": "iwslt",
    },
}


def convert_file(src_path: Path, dst_path: Path,
                 input_columns: tuple[str, str],
                 output_columns: tuple[str, str]) -> int:
    source_col, target_col = input_columns
    noisy_col, formatted_target_col = output_columns
    count = 0

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with src_path.open(encoding="utf-8") as src, \
            dst_path.open("w", encoding="utf-8", newline="\n") as dst:
        for line_no, line in enumerate(src, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = [col for col in input_columns if col not in row]
            if missing:
                raise ValueError(
                    f"{src_path}:{line_no} is missing required columns: {missing}"
                )
            formatted = {
                noisy_col: row[source_col],
                formatted_target_col: row[target_col],
            }
            dst.write(json.dumps(formatted, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASETS),
        help="Format only one dataset. Defaults to all datasets with existing files.",
    )
    args = parser.parse_args()

    dataset_names = [args.dataset] if args.dataset else sorted(DATASETS)
    total_files = 0

    for dataset_name in dataset_names:
        spec = DATASETS[dataset_name]
        src_dir = DATA_ROOT / dataset_name
        dst_dir = REPO_ROOT / "formatted_data" / spec["output_dir"]

        if not src_dir.exists():
            print(f"skip {dataset_name}: source directory not found ({src_dir})")
            continue

        jsonl_files = sorted(src_dir.glob("*.jsonl"))
        if not jsonl_files:
            print(f"skip {dataset_name}: no JSONL files found in {src_dir}")
            continue

        for src_path in jsonl_files:
            dst_path = dst_dir / src_path.name
            n_rows = convert_file(
                src_path,
                dst_path,
                spec["input_columns"],
                spec["output_columns"],
            )
            total_files += 1
            print(f"wrote {dst_path.relative_to(REPO_ROOT)}  rows={n_rows}")

    print(f"formatted files written: {total_files}")


if __name__ == "__main__":
    main()
