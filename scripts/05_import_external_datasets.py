"""
05_import_external_datasets.py - Import sibling dataset/ artifacts.

Converts externally prepared datasets into the same formatted layout consumed by
data_utils.prepare_formatted_dataset:

  formatted_data/gigaword/*.jsonl
    {"noisy_document": "...", "summary": "..."}

  formatted_data/iwslt/*.jsonl
    {"noisy_source": "...", "target": "..."}

The input directory defaults to ../dataset relative to this repository.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_from_disk


REPO_ROOT = Path(__file__).resolve().parent.parent
FORMATTED_ROOT = REPO_ROOT / "formatted_data"
DEFAULT_EXTERNAL_ROOT = REPO_ROOT.parent / "dataset"


def write_jsonl(rows, path: Path) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def import_gigaword(external_root: Path) -> dict:
    src_root = external_root / "gigaword_10k_size_subsets"
    manifest_path = src_root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing Gigaword manifest: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dst_root = FORMATTED_ROOT / "gigaword"
    imported = {
        "source": str(src_root),
        "source_dataset": manifest.get("source_dataset"),
        "notes": [
            "External Gigaword uses columns document/summary.",
            "document was mapped from the source article column.",
            "Clean subset files are copied into the canonical formatted_data/gigaword directory.",
        ],
        "files": {},
    }

    for meta in manifest["files"]:
        src_path = src_root / meta["jsonl"]
        if not src_path.exists():
            raise FileNotFoundError(f"Missing Gigaword JSONL: {src_path}")

        split = meta["split"]
        size = meta["size"]
        if split == "train":
            out_name = f"train_clean_{size}.jsonl"
        elif split == "validation":
            out_name = f"valid_clean_{size}.jsonl"
        elif split == "test":
            out_name = f"test_clean_{size}.jsonl"
        else:
            raise ValueError(f"Unsupported Gigaword split: {split}")

        def rows():
            with src_path.open(encoding="utf-8") as f:
                for line_no, line in enumerate(f, start=1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    missing = [col for col in ("document", "summary") if col not in row]
                    if missing:
                        raise ValueError(
                            f"{src_path}:{line_no} missing required columns: {missing}"
                        )
                    yield {
                        "noisy_document": row["document"],
                        "summary": row["summary"],
                    }

        dst_path = dst_root / out_name
        n_rows = write_jsonl(rows(), dst_path)
        imported["files"][out_name] = {
            "source_file": str(src_path),
            "source_name": Path(meta["jsonl"]).name,
            "split": split,
            "n": n_rows,
        }
        print(f"gigaword -> {dst_path.relative_to(REPO_ROOT)} rows={n_rows}")

    return imported


def import_iwslt(external_root: Path) -> dict:
    src_root = external_root / "iwslt2017_multiple_size"
    if not src_root.exists():
        raise FileNotFoundError(f"Missing IWSLT directory: {src_root}")

    subset_dirs = sorted(
        [p for p in src_root.glob("iwslt_*") if p.is_dir()],
        key=lambda p: int(p.name.rsplit("_", 1)[1]),
    )
    if not subset_dirs:
        raise FileNotFoundError(f"No IWSLT subset directories found in {src_root}")

    dst_root = FORMATTED_ROOT / "iwslt"
    imported = {
        "source": str(src_root),
        "notes": [
            "External IWSLT is loaded with datasets.load_from_disk.",
            "Translation direction is src=en to tgt=de.",
            "Files are clean only; noisy_source equals the clean src.",
        ],
        "files": {},
    }

    for subset_dir in subset_dirs:
        size = int(subset_dir.name.rsplit("_", 1)[1])
        ds = load_from_disk(str(subset_dir))
        if not {"src", "tgt"}.issubset(ds.column_names):
            raise ValueError(
                f"{subset_dir} must contain src/tgt columns. Found: {ds.column_names}"
            )

        dst_path = dst_root / f"train_clean_{size}.jsonl"
        n_rows = write_jsonl(
            (
                {"noisy_source": row["src"], "target": row["tgt"]}
                for row in ds
            ),
            dst_path,
        )
        imported["files"][dst_path.name] = {
            "source_dir": str(subset_dir),
            "split": "train",
            "n": n_rows,
        }
        print(f"iwslt -> {dst_path.relative_to(REPO_ROOT)} rows={n_rows}")

    return imported


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--external_root", type=Path, default=DEFAULT_EXTERNAL_ROOT)
    parser.add_argument(
        "--dataset",
        choices=("all", "gigaword", "iwslt"),
        default="all",
    )
    args = parser.parse_args()

    external_root = args.external_root.resolve()
    if not external_root.exists():
        raise FileNotFoundError(f"External dataset root not found: {external_root}")

    if args.dataset in ("all", "gigaword"):
        manifest = import_gigaword(external_root)
        out = FORMATTED_ROOT / "gigaword" / "import_manifest.json"
        out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {out.relative_to(REPO_ROOT)}")

    if args.dataset in ("all", "iwslt"):
        manifest = import_iwslt(external_root)
        out = FORMATTED_ROOT / "iwslt" / "import_manifest.json"
        out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
