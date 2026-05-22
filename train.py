"""
train.py — Fair DLM vs ARM Comparison
======================================
Fixes vs. DLMvsARM_Prelim/train.py:

  [FIX-1] LoRA task_type is FEATURE_EXTRACTION for DLM (bidirectional model).
          CAUSAL_LM injects causal attention biases that break LLaDA's
          bidirectional attention, systematically harming its fine-tuning.

  [FIX-2] DiffusionDataCollator uses a fixed masking rate (default 0.15) instead
          of t ~ U(0,1). A uniform random rate creates highly variable gradient
          signal — some batches mask almost nothing, others mask everything.
          A fixed rate gives a stable, dense signal comparable to AR's loss.

  [FIX-3] DiffusionDataCollator masks TARGET tokens ONLY, not the full sequence.
          The source side is kept visible (as AR sees it). Previously, DLM was
          trained with random masking over source+target, which (combined with
          word-level noise) doubly-corrupted the source at noise_rate > 0.
"""

import argparse
import os

import torch
from accelerate import Accelerator
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    get_scheduler,
)
from peft import get_peft_model, LoraConfig, TaskType
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm

from data_utils import prepare_dataset


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Fair DLM vs ARM fine-tuning on seq2seq tasks."
    )
    p.add_argument("--model_type", choices=["ar", "dlm"], required=True)
    p.add_argument("--dataset", default="iwslt2017")
    p.add_argument("--data_dir", default=None,
                   help="Optional local CSV directory for dataset subsets.")
    p.add_argument("--subset_size", type=int, default=1000)
    p.add_argument("--noise_rate", type=float, default=0.0)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--output_dir", default="checkpoints")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Collators
# ---------------------------------------------------------------------------

class DiffusionDataCollator:
    """
    [FIX-3] Target-only masking collator for LLaDA.

    Masking uses per-sample t ~ U(0, 1) — LLaDA's correct training objective.
    Using a fixed low rate (e.g. 0.15) would mean the model never sees heavy
    masking during training, breaking generation which always starts from 100%
    masked tokens.

    Only TARGET tokens are masked (source is kept visible), preventing the
    double-corruption of the source side that the Prelim had. [FIX-3]

    'source_len' must be present in each feature dict (set by tokenize_function).
    """

    def __init__(self, tokenizer, mask_token_id: int):
        self.tokenizer = tokenizer
        self.mask_token_id = mask_token_id

    def __call__(self, features):
        source_lens = [f["source_len"] for f in features]

        input_ids = [torch.tensor(f["input_ids"]) for f in features]
        attention_mask = [torch.tensor(f["attention_mask"]) for f in features]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask, batch_first=True, padding_value=0
        )

        labels = input_ids.clone()
        masked_input_ids = input_ids.clone()
        batch_size, _ = input_ids.shape

        # [FIX-3] Target-only region: positions after source prefix, within padding
        target_region = torch.zeros_like(attention_mask, dtype=torch.bool)
        for i, src_len in enumerate(source_lens):
            target_region[i, src_len:] = attention_mask[i, src_len:].bool()

        # Per-sample masking ratio t ~ U(0, 1) — LLaDA's training objective.
        # This ensures the model trains on all masking levels (0% → 100%),
        # which is required for generation to work (starts at 100% masked).
        mask_ratios = torch.rand(batch_size)
        rand_tensor = torch.rand(input_ids.shape)
        mask_prob = mask_ratios.unsqueeze(1).expand_as(input_ids)
        masked_indices = (rand_tensor < mask_prob) & target_region

        masked_input_ids[masked_indices] = self.mask_token_id

        # Loss only on masked target positions
        labels[~masked_indices] = -100

        return {
            "input_ids": masked_input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    accelerator = Accelerator(log_with="wandb")

    run_name = f"{args.model_type}_{args.dataset}_{args.subset_size}_noise{args.noise_rate}"
    if accelerator.is_main_process:
        wandb.init(project="diffusion_vs_ar_llm_fair", name=run_name, config=vars(args))

    # ---- Model ----
    model_name = (
        "meta-llama/Meta-Llama-3-8B-Instruct"
        if args.model_type == "ar"
        else "GSAI-ML/LLaDA-8B-Instruct"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.model_type == "dlm" and getattr(tokenizer, "mask_token", None) is None:
        tokenizer.add_special_tokens({"mask_token": "<|mask|>"})

    bnb_config = BitsAndBytesConfig(load_in_4bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map={"": accelerator.local_process_index},
        trust_remote_code=True,
    )

    if args.model_type == "dlm" and len(tokenizer) > model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))

    # [FIX-1] CAUSAL_LM for AR (correct), FEATURE_EXTRACTION for DLM (bidirectional).
    lora_task = (
        TaskType.CAUSAL_LM if args.model_type == "ar" else TaskType.FEATURE_EXTRACTION
    )
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=lora_task,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ---- Data ----
    dataset, src_col, tgt_col = prepare_dataset(
        args.dataset, args.subset_size, args.noise_rate, data_dir=args.data_dir
    )

    def tokenize_function(examples):
        texts = [
            f"Source: {s}\nTarget: {t}"
            for s, t in zip(examples[f"noisy_{src_col}"], examples[tgt_col])
        ]
        result = tokenizer(texts, truncation=True, max_length=512)

        if args.model_type == "dlm":
            # [FIX-3] Compute source prefix lengths so collator can skip them.
            prefixes = [
                f"Source: {s}\nTarget: " for s in examples[f"noisy_{src_col}"]
            ]
            result["source_len"] = [
                len(tokenizer(p, add_special_tokens=False)["input_ids"])
                for p in prefixes
            ]
        return result

    tokenized = dataset.map(
        tokenize_function, batched=True, remove_columns=dataset.column_names
    )

    if args.model_type == "ar":
        collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    else:
        collator = DiffusionDataCollator(tokenizer, tokenizer.mask_token_id)

    dataloader = DataLoader(
        tokenized, batch_size=args.batch_size, shuffle=True, collate_fn=collator
    )

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Prepare BEFORE computing num_steps so len(dataloader) reflects the
    # DistributedSampler-sharded length (total / num_processes), not the
    # full dataset length. With 4 GPUs and 5000 examples this is the
    # difference between 1875 real steps and the wrong value of 7500.
    optimizer, dataloader = accelerator.prepare(optimizer, dataloader)

    num_steps = args.epochs * len(dataloader)   # correct sharded count
    lr_scheduler = get_scheduler(
        "linear", optimizer=optimizer, num_warmup_steps=max(1, num_steps // 10),
        num_training_steps=num_steps
    )
    lr_scheduler = accelerator.prepare(lr_scheduler)

    # ---- Training Loop ----
    model.train()
    if accelerator.is_main_process:
        print(f"[train] {args.epochs} epochs × {len(dataloader)} steps/epoch "
              f"= {num_steps} total steps across {accelerator.num_processes} GPU(s).")
    pbar = tqdm(range(num_steps), disable=not accelerator.is_local_main_process)

    for epoch in range(args.epochs):
        for step, batch in enumerate(dataloader):
            outputs = model(**batch)
            loss = outputs.loss

            # LLaDA may return dict or None; recompute manually if needed.
            if loss is None or isinstance(loss, dict):
                logits = outputs.logits
                labels = batch["labels"]
                # Non-causal: do NOT shift logits/labels.
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), labels.view(-1)
                )

            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            pbar.update(1)

            if accelerator.is_main_process and step % 10 == 0:
                wandb.log({
                    "loss": loss.item(),
                    "epoch": epoch,
                    "lr": lr_scheduler.get_last_lr()[0],
                })

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        save_path = os.path.join(args.output_dir, run_name)
        unwrapped.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        wandb.finish()
        print(f"Saved to {save_path}")


if __name__ == "__main__":
    main()
