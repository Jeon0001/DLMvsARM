# DLMvsARM

This branch keeps the original inline dataset loader as `prepare_dataset()` and
adds a separate formatted-data loader, `prepare_formatted_dataset()`.

The two functions intentionally have different semantics:

- `prepare_dataset()` loads from Hugging Face, shuffles/selects examples, and
  applies inline source noise.
- `prepare_formatted_dataset()` reads already split/noised JSONL files from
  `formatted_data/`.

Because `prepare_formatted_dataset()` keeps the same call signature as
`prepare_dataset()`, a call site can replace `prepare_dataset(...)` with
`prepare_formatted_dataset(...)` without changing the surrounding argument flow.
`train.py` and `eval.py` are not switched by default in this branch.

## Formatted Data Layout

Formatted training inputs live in the repository root under:

```text
formatted_data/
  gigaword/
  iwslt/
```

This directory is intentionally not ignored by git. The old `data/**/*.jsonl`
rule still applies only to generated raw preprocessing artifacts under `data/`.

### Gigaword

Expected row format:

```json
{"noisy_document": "...", "summary": "..."}
```

Current files:

```text
formatted_data/gigaword/train_clean_{100,500,1000,5000,10000}.jsonl
formatted_data/gigaword/train_noisy_{N}_{swap,delete,replace}_{10,20,30}.jsonl
formatted_data/gigaword/valid_clean_*.jsonl
formatted_data/gigaword/test_clean_1951.jsonl
```

Clean Gigaword subset files were copied from the external
`../dataset/gigaword_10k_size_subsets/jsonl/` source. Those external train
subsets are nested prefixes.

The noisy Gigaword files are the existing preprocessing outputs with explicit
noise types (`swap`, `delete`, `replace`). No new mixed-noise dataset is created.

### IWSLT

Expected row format:

```json
{"noisy_source": "...", "target": "..."}
```

Current files:

```text
formatted_data/iwslt/train_clean_{100,500,1000,5000,10000}.jsonl
```

The IWSLT files were imported from the external Hugging Face `load_from_disk`
directories under `../dataset/iwslt2017_multiple_size/`. The task direction is
`src=en -> tgt=de`, converted to `noisy_source -> target`.

## Loader Behavior

`data_utils.prepare_dataset()` preserves the original inline loader behavior:

- loads from Hugging Face
- shuffles with `seed`
- selects `size`
- applies inline word-level noise

`data_utils.prepare_formatted_dataset()` is the formatted loader. Replace a
call to `prepare_dataset(...)` with `prepare_formatted_dataset(...)` when the
training/evaluation run should use committed JSONL files from `formatted_data/`.

Clean data:

```bash
python train.py --dataset gigaword --subset_size 1000 --noise_rate 0.0
python train.py --dataset iwslt2017 --subset_size 1000 --noise_rate 0.0
```

Noisy Gigaword data:

```bash
python train.py --dataset gigaword_swap --subset_size 1000 --noise_rate 0.2
python train.py --dataset gigaword_delete --subset_size 1000 --noise_rate 0.2
python train.py --dataset gigaword_replace --subset_size 1000 --noise_rate 0.2
```

For `noise_rate > 0`, the dataset name must include the explicit noise suffix.
This maps directly to the existing files:

```text
train_noisy_{N}_{type}_{ratio}.jsonl
```

For example, use `gigaword_swap`, `gigaword_delete`, or `gigaword_replace`.
Do not use plain `gigaword` with `noise_rate > 0` for formatted data; there is
no single mixed-noise formatted file.

## Current Limitations

- IWSLT currently has clean formatted files only. Calls such as
  `--dataset iwslt2017 --noise_rate 0.2` or `--dataset iwslt2017_swap` will fail
  until matching `train_noisy_*` files are produced.
- Gigaword clean train subsets now come from the external
  `gigaword_10k_size_subsets` source, while the existing noisy files come from
  the preprocessing pipeline outputs already present in this branch. They are
  treated as separate formatted inputs; this change does not regenerate noisy
  Gigaword files from the external clean subsets.
- `prepare_formatted_dataset()` keeps the same call signature as
  `prepare_dataset()` so call sites can replace one function with the other
  without changing their argument flow, but `seed` is intentionally unused
  because the data is already pre-split.

## Import Scripts

Build formatted files from raw preprocessing outputs:

```bash
python scripts/04_format_preprocessed.py
```

Import the external sibling `../dataset/` artifacts:

```bash
python scripts/05_import_external_datasets.py
```

If both scripts are used, run `05_import_external_datasets.py` after
`04_format_preprocessed.py`. The current canonical Gigaword clean subsets are
the external `gigaword_10k_size_subsets` copies, while noisy Gigaword files are
the existing explicit-noise preprocessing outputs.
