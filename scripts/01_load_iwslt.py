"""
01_load_iwslt.py — Load IWSLT 2017 EN->DE and run EDA.  [TEMPLATE for Yohan]

Mirrors data_utils.prepare_dataset() loading conventions:
    load_dataset("iwslt2017", "iwslt2017-en-de", split="train",
                 trust_remote_code=True)

NOTE on `trust_remote_code`: datasets >= 4.0 removed it. If the call below
fails the same way `gigaword` did, search the Hub for a parquet mirror, e.g.

    from huggingface_hub import HfApi
    [r.id for r in HfApi().list_datasets(search="iwslt2017", limit=20)]

and switch to that id (Jimin used `SalmanFaroz/gigaword` for Gigaword for the
same reason).

Outputs:
  - prints split sizes, sample rows
  - prints word-length distribution for source(en) and target(de)
  - saves data/iwslt/eda_stats.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from datasets import load_dataset


REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "data" / "iwslt"
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


def extract_translation(example):
    return {
        "source": example["translation"]["en"],
        "target": example["translation"]["de"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show_n", type=int, default=3)
    args = parser.parse_args()

    print("=" * 70)
    print("Loading iwslt2017 (en-de) ...")
    print("=" * 70)

    # TODO(Yohan): if `trust_remote_code` removal blocks loading, swap this
    # for a parquet mirror found on the Hub.
    ds = load_dataset("iwslt2017", "iwslt2017-en-de")
    ds = ds.map(extract_translation, remove_columns=["translation"])

    print("\nSplits & sizes:")
    for split, sub in ds.items():
        print(f"  {split:10s}  n={len(sub):>10,}  cols={sub.column_names}")

    train = ds["train"]
    print(f"\nFirst {args.show_n} train rows (raw):")
    for i in range(args.show_n):
        row = train[i]
        print(f"--- row {i} ---")
        print(f"  source(en): {row['source']}")
        print(f"  target(de): {row['target']}")

    print("\nLength distributions:")
    rows = []
    for split in ds:
        sub = ds[split]
        rows.append(describe_lengths(
            [word_len(x) for x in sub["source"]], f"{split}.source.en.words"))
        rows.append(describe_lengths(
            [word_len(x) for x in sub["target"]], f"{split}.target.de.words"))

    out_csv = OUT_DIR / "eda_stats.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nSaved length stats -> {out_csv}")


if __name__ == "__main__":
    main()
