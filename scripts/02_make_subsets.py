"""
02_make_subsets.py — Build clean Gigaword subsets and write JSONL files.

Subsets are NESTED (the original data_utils.py convention):
    base = train.shuffle(seed=42).select(range(10_000))
    sub_N = base.select(range(N))   for N in {100, 500, 1000, 5000, 10000}

This means a 100-sample run is a strict subset of a 500-sample run, etc.,
which keeps the noise-rate / data-size ablations comparable.

Outputs (in data/gigaword/):
    train_clean_{N}.jsonl       — N in {100, 500, 1000, 5000, 10000}
    valid_clean_500.jsonl       — eval pool (clean)
    test_clean_1951.jsonl       — full test split (clean)
    summary.json                — counts + first-row hashes for reproducibility
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from datasets import load_dataset


REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "data" / "gigaword"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASET_ID = "SalmanFaroz/gigaword"
SUBSET_SIZES = (100, 500, 1000, 5000, 10000)
VALID_POOL = 500     # how many valid rows to dump (eval uses 100; keep 5x headroom)
SEED = 42


def write_jsonl(rows, path: Path) -> str:
    """Write rows as JSONL. Returns sha256 of the file content."""
    h = hashlib.sha256()
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            line = json.dumps(
                {"document": r["document"], "summary": r["summary"]},
                ensure_ascii=False,
            ) + "\n"
            f.write(line)
            h.update(line.encode("utf-8"))
    return h.hexdigest()[:16]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    print(f"Loading {DATASET_ID} ...")
    ds = load_dataset(DATASET_ID).rename_column("article", "document")

    base = ds["train"].shuffle(seed=args.seed).select(range(max(SUBSET_SIZES)))
    print(f"Base shuffled subset: n={len(base):,}  (seed={args.seed})")

    summary = {
        "dataset_id": DATASET_ID,
        "seed": args.seed,
        "splits": {
            "train_full": len(ds["train"]),
            "validation_full": len(ds["validation"]),
            "test_full": len(ds["test"]),
        },
        "files": {},
    }

    # Train subsets (nested)
    for n in SUBSET_SIZES:
        sub = base.select(range(n))
        out = OUT_DIR / f"train_clean_{n}.jsonl"
        sha = write_jsonl(sub, out)
        first_doc = sub[0]["document"][:120]
        summary["files"][out.name] = {
            "n": n, "sha256_prefix": sha, "first_doc_preview": first_doc,
        }
        print(f"  wrote {out.name:28s}  n={n:>5}  sha={sha}")

    # Validation pool (clean)
    n_valid = min(VALID_POOL, len(ds["validation"]))
    valid = ds["validation"].shuffle(seed=args.seed).select(range(n_valid))
    out = OUT_DIR / f"valid_clean_{n_valid}.jsonl"
    sha = write_jsonl(valid, out)
    summary["files"][out.name] = {"n": n_valid, "sha256_prefix": sha}
    print(f"  wrote {out.name:28s}  n={n_valid:>5}  sha={sha}")

    # Test (clean, full)
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
