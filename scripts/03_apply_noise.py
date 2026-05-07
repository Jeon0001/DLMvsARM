"""
03_apply_noise.py — Build train vocab + apply noise to all train subsets.

Reads:  data/gigaword/train_clean_{N}.jsonl
Writes: data/gigaword/train_noisy_{N}_{type}_{ratio}.jsonl
        data/gigaword/vocab.txt   (one word per line, sorted by freq desc)
        data/gigaword/noise_manifest.json

Policy (see noise_utils.py docstring): source-only word-level corruption,
one noise type per file, deterministic per (size, type, ratio) RNG.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))   # so `import noise_utils` works

from noise_utils import (   # noqa: E402
    NOISE_TYPES, build_vocab, corrupt, make_rng, MIN_WORDS,
)

OUT_DIR = REPO_ROOT / "data" / "gigaword"
SUBSET_SIZES = (100, 500, 1000, 5000, 10000)
NOISE_RATIOS = (0.1, 0.2, 0.3)
DATASET_NAME = "gigaword"
SEED = 42


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write_jsonl(rows: list[dict], path: Path) -> tuple[str, dict]:
    """Write JSONL; return (sha256_prefix, audit_dict)."""
    h = hashlib.sha256()
    n_changed = 0
    n_skipped_short = 0
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            line = json.dumps(
                {"document": r["document"], "summary": r["summary"]},
                ensure_ascii=False,
            ) + "\n"
            f.write(line)
            h.update(line.encode("utf-8"))
            if r.get("_changed"):
                n_changed += 1
            if r.get("_skipped_short"):
                n_skipped_short += 1
    return h.hexdigest()[:16], {
        "n_changed": n_changed,
        "n_skipped_short": n_skipped_short,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--vocab_min_count", type=int, default=2)
    args = parser.parse_args()

    # ---- 1. Vocab from the largest train subset (10K base) ------------------
    base_path = OUT_DIR / "train_clean_10000.jsonl"
    if not base_path.exists():
        sys.exit(f"Missing {base_path}. Run scripts/02_make_subsets.py first.")
    base_rows = read_jsonl(base_path)
    print(f"Base for vocab: {base_path.name}  n={len(base_rows):,}")

    vocab = build_vocab((r["document"] for r in base_rows),
                        min_count=args.vocab_min_count)
    print(f"Vocab size (min_count={args.vocab_min_count}): {len(vocab):,} words")
    vocab_path = OUT_DIR / "vocab.txt"
    vocab_path.write_text("\n".join(vocab) + "\n", encoding="utf-8")

    # ---- 2. Apply noise -----------------------------------------------------
    manifest = {
        "seed": args.seed,
        "min_words": MIN_WORDS,
        "noise_types": list(NOISE_TYPES),
        "noise_ratios": list(NOISE_RATIOS),
        "subset_sizes": list(SUBSET_SIZES),
        "vocab_size": len(vocab),
        "files": {},
    }

    for n in SUBSET_SIZES:
        clean_path = OUT_DIR / f"train_clean_{n}.jsonl"
        rows_clean = read_jsonl(clean_path)
        for ntype in NOISE_TYPES:
            for ratio in NOISE_RATIOS:
                rng = make_rng(DATASET_NAME, n, ntype, ratio, seed=args.seed)
                rows_out = []
                for r in rows_clean:
                    src = r["document"]
                    if len(src.split()) < MIN_WORDS:
                        rows_out.append({
                            "document": src, "summary": r["summary"],
                            "_changed": False, "_skipped_short": True,
                        })
                        continue
                    noisy = corrupt(src, ntype, ratio, rng, vocab=vocab)
                    rows_out.append({
                        "document": noisy, "summary": r["summary"],
                        "_changed": noisy != src, "_skipped_short": False,
                    })
                ratio_tag = f"{int(ratio * 100):02d}"
                out_path = OUT_DIR / f"train_noisy_{n}_{ntype}_{ratio_tag}.jsonl"
                sha, audit = write_jsonl(rows_out, out_path)
                manifest["files"][out_path.name] = {
                    "n": n, "noise_type": ntype, "noise_ratio": ratio,
                    "sha256_prefix": sha, **audit,
                }
                print(f"  {out_path.name:46s}  changed={audit['n_changed']:>5}/{n}"
                      f"  skipped<{MIN_WORDS}w={audit['n_skipped_short']:>3}"
                      f"  sha={sha}")

    manifest_path = OUT_DIR / "noise_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                             encoding="utf-8")
    print(f"\nSaved -> {manifest_path}")
    print(f"Vocab -> {vocab_path}  ({len(vocab):,} words)")

    # ---- 3. Sanity preview --------------------------------------------------
    print("\n=== sanity preview (n=1000, ratio=0.3) ===")
    n_preview = 1000
    rows_clean = read_jsonl(OUT_DIR / f"train_clean_{n_preview}.jsonl")
    sample_doc = rows_clean[0]["document"]
    print(f"\noriginal:\n  {sample_doc}")
    for ntype in NOISE_TYPES:
        rng = make_rng(DATASET_NAME, n_preview, ntype, 0.3, seed=args.seed)
        out = corrupt(sample_doc, ntype, 0.3, rng, vocab=vocab)
        print(f"\n[{ntype:7s} 30%]:\n  {out}")


if __name__ == "__main__":
    main()
