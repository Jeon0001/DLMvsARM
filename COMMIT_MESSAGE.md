Use formatted JSONL datasets for training

- Preserve the original `prepare_dataset()` implementation for the legacy
  Hugging Face + inline-noise path.
- Add `prepare_formatted_dataset()` for preformatted JSONL files under the
  tracked `formatted_data/` directory.
- Keep `prepare_formatted_dataset()` signature-compatible with
  `prepare_dataset()` so call sites can replace one with the other without
  changing argument flow.
- Leave `train.py` and `eval.py` on `prepare_dataset()` by default; formatted
  data is enabled by explicitly replacing that function at the call site.
- Add formatted Gigaword and IWSLT inputs:
  - `formatted_data/gigaword` uses `noisy_document` / `summary`.
  - `formatted_data/iwslt` uses `noisy_source` / `target`.
- Keep Gigaword noisy data explicit by requiring suffixes such as
  `gigaword_swap`, `gigaword_delete`, or `gigaword_replace`.
- Do not create synthetic mixed-noise files. Only existing preprocessed noisy
  files are used.
- Document that plain `gigaword` with `noise_rate > 0` is invalid for formatted
  data; callers must specify the explicit noise type suffix.
- Document limitations:
  - IWSLT currently has clean formatted files only.
  - Gigaword clean train subsets were copied from the external
    `gigaword_10k_size_subsets` source, while existing noisy files are reused
    from the branch preprocessing artifacts.
