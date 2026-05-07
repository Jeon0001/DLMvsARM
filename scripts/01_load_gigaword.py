"""
01_load_gigaword.py — Load Gigaword and run EDA.

The original `gigaword` dataset on HF Hub used a script and is no longer
loadable in datasets>=4.0 (`trust_remote_code` removed). We use the parquet
mirror `SalmanFaroz/gigaword`, which has IDENTICAL split sizes
(train=3,803,957 / val=189,651 / test=1,951) but renames the source column
`document` -> `article`. We rename it back to `document` on load so the rest
of the pipeline (data_utils.prepare_dataset) stays unchanged.

Outputs:
  - prints split sizes, column names, sample rows
  - prints word-length distribution (document/summary)
  - saves a small CSV of summary stats to data/gigaword/eda_stats.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from datasets import load_dataset


REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "data" / "gigaword"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def word_len(text: str) -> int:
    return len(text.split())


def describe_lengths(values, label: str) -> dict:
    s = pd.Series(values)
    desc = {
        "field": label,
        "count": int(s.count()),
        "mean": float(s.mean()),
        "std": float(s.std()),
        "min": int(s.min()),
        "p25": float(s.quantile(0.25)),
        "p50": float(s.quantile(0.50)),
        "p75": float(s.quantile(0.75)),
        "p95": float(s.quantile(0.95)),
        "max": int(s.max()),
    }
    print(f"[{label}] n={desc['count']}  "
          f"mean={desc['mean']:.1f}  std={desc['std']:.1f}  "
          f"min={desc['min']}  p25={desc['p25']:.0f}  p50={desc['p50']:.0f}  "
          f"p75={desc['p75']:.0f}  p95={desc['p95']:.0f}  max={desc['max']}")
    return desc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe_n", type=int, default=20000,
                        help="Number of train rows to scan for length stats "
                             "(streaming-style sample, not the full 4M).")
    parser.add_argument("--show_n", type=int, default=3,
                        help="Number of sample rows to print.")
    args = parser.parse_args()

    print("=" * 70)
    print("Loading SalmanFaroz/gigaword (parquet mirror of original Gigaword) ...")
    print("=" * 70)

    ds = load_dataset("SalmanFaroz/gigaword")
    ds = ds.rename_column("article", "document")

    print("\nSplits & sizes:")
    for split, sub in ds.items():
        print(f"  {split:10s}  n={len(sub):>10,}  cols={sub.column_names}")

    train = ds["train"]
    print(f"\nFirst {args.show_n} train rows (raw):")
    for i in range(args.show_n):
        row = train[i]
        print(f"--- row {i} ---")
        print(f"  document: {row['document']}")
        print(f"  summary : {row['summary']}")

    n_probe = min(args.probe_n, len(train))
    print(f"\nScanning first {n_probe:,} rows for length distribution ...")
    probe = train.select(range(n_probe))
    doc_lens = [word_len(x) for x in probe["document"]]
    sum_lens = [word_len(x) for x in probe["summary"]]

    print()
    rows = [
        describe_lengths(doc_lens, "document.words"),
        describe_lengths(sum_lens, "summary.words"),
    ]
    for split in ("validation", "test"):
        if split in ds:
            sub = ds[split]
            describe_lengths([word_len(x) for x in sub["document"]],
                             f"{split}.document.words")
            describe_lengths([word_len(x) for x in sub["summary"]],
                             f"{split}.summary.words")

    out_csv = OUT_DIR / "eda_stats.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nSaved length stats -> {out_csv}")


if __name__ == "__main__":
    main()
