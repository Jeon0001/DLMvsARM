"""
02_make_iwslt_subsets.py — Build clean IWSLT 2017 subsets.  [TEMPLATE for Yohan]

Mirrors scripts/02_make_subsets.py (the Gigaword version) so the two datasets
have identical subset semantics: NESTED with seed=42.

Outputs (in data/iwslt/):
    train_clean_{N}.jsonl       — N in {100, 500, 1000, 5000, 10000}
    valid_clean_500.jsonl       — eval pool
    test_clean_<n_test>.jsonl   — full test split
    summary.json                — counts + sha256 prefix per file
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from datasets import load_dataset


REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "data" / "iwslt"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASET_ID = "iwslt2017"
DATASET_CONFIG = "iwslt2017-en-de"
SUBSET_SIZES = (100, 500, 1000, 5000, 10000)
VALID_POOL = 500
SEED = 42


def extract_translation(example):
    return {
        "source": example["translation"]["en"],
        "target": example["translation"]["de"],
    }


def write_jsonl(rows, path: Path) -> str:
    h = hashlib.sha256()
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            line = json.dumps(
                {"source": r["source"], "target": r["target"]},
                ensure_ascii=False,
            ) + "\n"
            f.write(line)
            h.update(line.encode("utf-8"))
    return h.hexdigest()[:16]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    print(f"Loading {DATASET_ID} / {DATASET_CONFIG} ...")
    ds = load_dataset(DATASET_ID, DATASET_CONFIG).map(
        extract_translation, remove_columns=["translation"]
    )

    base = ds["train"].shuffle(seed=args.seed).select(range(max(SUBSET_SIZES)))
    print(f"Base shuffled subset: n={len(base):,}  (seed={args.seed})")

    summary = {
        "dataset_id": f"{DATASET_ID}/{DATASET_CONFIG}",
        "seed": args.seed,
        "splits": {f"{s}_full": len(ds[s]) for s in ds},
        "files": {},
    }

    for n in SUBSET_SIZES:
        sub = base.select(range(n))
        out = OUT_DIR / f"train_clean_{n}.jsonl"
        sha = write_jsonl(sub, out)
        summary["files"][out.name] = {
            "n": n, "sha256_prefix": sha,
            "first_source_preview": sub[0]["source"][:120],
        }
        print(f"  wrote {out.name:28s}  n={n:>5}  sha={sha}")

    n_valid = min(VALID_POOL, len(ds["validation"]))
    valid = ds["validation"].shuffle(seed=args.seed).select(range(n_valid))
    out = OUT_DIR / f"valid_clean_{n_valid}.jsonl"
    sha = write_jsonl(valid, out)
    summary["files"][out.name] = {"n": n_valid, "sha256_prefix": sha}
    print(f"  wrote {out.name:28s}  n={n_valid:>5}  sha={sha}")

    n_test = len(ds["test"])
    out = OUT_DIR / f"test_clean_{n_test}.jsonl"
    sha = write_jsonl(ds["test"], out)
    summary["files"][out.name] = {"n": n_test, "sha256_prefix": sha}
    print(f"  wrote {out.name:28s}  n={n_test:>5}  sha={sha}")

    summary_path = OUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    print(f"\nSaved manifest -> {summary_path}")


if __name__ == "__main__":
    main()
