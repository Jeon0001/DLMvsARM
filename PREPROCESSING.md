# Preprocessing Pipeline — Shared by Gigaword (Jimin) and IWSLT (Yohan)

This document defines the **single noise policy** both datasets must follow so
that RQ2 (robustness to training-time noise) compares like-for-like.

> **One rule above all:** never apply noise inline at training time again. Both
> datasets are now pre-corrupted into JSONL files, regenerated deterministically
> from `noise_utils.py` with a per-cell RNG seed.

---

## 1. Policy (the 14-question table)

| # | Decision | Value |
|---|---|---|
| 1  | Where is noise applied? | **Source side only.** Target is always clean. |
| 2  | IWSLT direction | **EN → DE** (source = en, target = de) |
| 3  | Noise ratios | **0.0 / 0.1 / 0.2 / 0.3** (clean is one file, the others are 3) |
| 4  | Mixing types in one file? | **No.** One noise type per file: `swap` / `delete` / `replace`. |
| 5  | Order if mixed | N/A (we never mix) |
| 6  | Replace vocabulary | Real words from the **source-side train vocab** of that dataset, with `min_count=2` (no random ASCII). |
| 7  | Tokenization | **Whitespace split** (word level) |
| 8  | Punctuation / special tokens | Punctuation stays attached to its word; no special tokens at this stage. |
| 9  | Edge cases | Sentences with `< 5` source words → returned unchanged. `delete` keeps at least one word. |
| 10 | File format / naming | JSONL, two fields `{document, summary}` (or `{source, target}` for IWSLT). Filenames: `train_{clean,noisy}_{N}_{type}_{ratio}.jsonl` |
| 11 | Noisy version per subset? | **Yes.** All five subsets (100/500/1K/5K/10K) × 3 noise types × 3 noise ratios = 45 noisy + 5 clean train files per dataset. |
| 12 | Subset before or after noise? | **Subset first, then noise.** (Each subset gets its own deterministic noise.) |
| 13 | Random seed | **42**, plus a per-cell key `f"{dataset}|{N}|{type}|{ratio}|{seed}"` so each cell is independently reproducible. |
| 14 | Train / valid / test | **Noise only on train.** Valid and test stay clean. |

---

## 2. Directory layout

```
DLMvsARM/
├── noise_utils.py                  # SHARED noise module (both datasets import this)
├── data_utils.py                   # (legacy in-memory pipeline — to be retired)
├── train.py / eval.py              # to be updated to read JSONL files
├── requirements.txt                # full env (training)
├── PREPROCESSING.md                # this file
├── scripts/
│   ├── 01_load_gigaword.py         # Jimin
│   ├── 02_make_subsets.py          # Jimin (Gigaword subsets)
│   ├── 03_apply_noise.py           # Jimin (Gigaword noise)
│   ├── 01_load_iwslt.py            # Yohan
│   ├── 02_make_iwslt_subsets.py    # Yohan
│   └── 03_apply_iwslt_noise.py     # Yohan
└── data/
    ├── gigaword/
    │   ├── summary.json            # subset manifest (sha256 per file)        ← tracked
    │   ├── noise_manifest.json     # noise manifest (sha256 + audit)          ← tracked
    │   ├── vocab.txt               # 11,829 words used for replace            ← tracked
    │   ├── eda_stats.csv           # length distribution                      ← tracked
    │   ├── train_clean_*.jsonl     # 5 files                                  ← gitignored
    │   ├── train_noisy_*.jsonl     # 45 files                                 ← gitignored
    │   ├── valid_clean_500.jsonl                                              ← gitignored
    │   └── test_clean_1951.jsonl                                              ← gitignored
    └── iwslt/                      # Yohan creates the same shape here
        ├── summary.json
        ├── noise_manifest.json
        ├── vocab.txt
        └── ...
```

**Why JSONL is gitignored:** total ~10 MB per dataset, regenerable in <1 min
from the scripts. **Why manifests are tracked:** they let the other person verify
their files match yours by sha256 without uploading the data itself.

---

## 3. Workflow

### 3.1 Setup (once, locally)

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows; use source .venv/bin/activate elsewhere
pip install datasets pandas tqdm
```

(The training environment installs the full `requirements.txt`; for
preprocessing only the three above are needed.)

### 3.2 Gigaword (Jimin) — already done

```bash
python scripts/01_load_gigaword.py        # download + EDA
python scripts/02_make_subsets.py         # writes train_clean_{N}.jsonl + valid + test
python scripts/03_apply_noise.py          # builds vocab + writes 45 noisy files
```

### 3.3 IWSLT (Yohan)

```bash
python scripts/01_load_iwslt.py           # download iwslt2017-en-de + EDA
python scripts/02_make_iwslt_subsets.py   # writes train_clean_{N}.jsonl
python scripts/03_apply_iwslt_noise.py    # builds en-vocab + writes 45 noisy files
```

The three IWSLT scripts are mirror copies of the Gigaword ones with the
dataset id and column names swapped — see the inline `# TODO(Yohan)` markers.

---

## 4. Public API (`noise_utils.py`)

```python
from noise_utils import (
    NOISE_TYPES, MIN_WORDS,
    build_vocab, corrupt, make_rng,
)

# 1. Build the source-side vocab from the *largest* train subset
vocab = build_vocab(
    (row["source"] for row in train_10k),   # iterable of strings
    min_count=2,                            # drop hapax legomena
)

# 2. For every (subset_size, noise_type, ratio) cell, get a deterministic RNG
rng = make_rng("iwslt", subset_size=5000, noise_type="replace",
               ratio=0.2, seed=42)

# 3. Corrupt one source string at a time
noisy = corrupt(text="The cat sat on the mat",
                noise_type="replace",
                ratio=0.2,
                rng=rng,
                vocab=vocab)
```

`corrupt` returns the input unchanged when `ratio <= 0` or
`len(text.split()) < MIN_WORDS` — counted as `_skipped_short` in the audit.

---

## 5. Cross-dataset reproducibility check

Both datasets write a `noise_manifest.json` like:

```json
{
  "files": {
    "train_noisy_5000_replace_20.jsonl": {
      "n": 5000,
      "noise_type": "replace",
      "noise_ratio": 0.2,
      "sha256_prefix": "bccf60ae67a5f923",
      "n_changed": 4988,
      "n_skipped_short": 0
    },
    ...
  }
}
```

The `sha256_prefix` lets you verify *byte-for-byte* that you generated the
right file. To verify your local copy:

```python
import hashlib, json
from pathlib import Path

manifest = json.loads(Path("data/iwslt/noise_manifest.json").read_text())
for name, meta in manifest["files"].items():
    h = hashlib.sha256(Path("data/iwslt", name).read_bytes()).hexdigest()[:16]
    status = "OK" if h == meta["sha256_prefix"] else "MISMATCH"
    print(f"{status:10s} {name}  expect={meta['sha256_prefix']}  got={h}")
```

If we both check in the same `noise_manifest.json` and the sha256s agree, we
have proven that our noise pipelines produced identical bytes.

---

## 6. Re-running / changing policy

If the policy ever changes (new ratio, new vocab cut), update **only**
`noise_utils.py` (or the constants at the top of each `03_*.py`) and rerun the
two `03_*.py` scripts. Subsets do not need to be regenerated unless the seed
or sizes change.
