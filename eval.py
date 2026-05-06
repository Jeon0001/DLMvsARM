"""
eval.py — Fair DLM vs ARM Comparison
======================================
Fixes vs. DLMvsARM_Prelim/eval.py:

  [FIX-4] Both models use the same max_new_tokens (default 128).
          Prelim hard-coded DLM target_len=64 while AR got max_new_tokens=128.

  [FIX-5] DLM decoding steps are configurable (--dlm_steps, default 64)
          and labelled clearly. Prelim used only 16 steps with no
          justification, giving DLM far fewer refinement passes than
          the effective 128 AR forward passes.

  [FIX-6] 'tracker' metric is now a UNIFIED scoring NLL for both models:
          - AR : teacher-forced causal NLL on the reference target.
          - DLM: masked reconstruction NLL (all target tokens masked,
                 one forward pass). These are the natural log-likelihoods
                 under each model's paradigm, both in the same units
                 (nats/token, lower = better). Prelim mixed AR NLL (lower=better)
                 with DLM confidence (higher=better) under the same JSON key.

  [FIX-7] --subset_size is an explicit CLI argument; no fragile path-parsing.

  [FIX-8] DLM generation uses target_len = max_new_tokens (same as AR).
"""

import argparse
import json
import os

import torch
import evaluate as hf_evaluate
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from data_utils import prepare_dataset


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Fair DLM vs ARM evaluation.")
    p.add_argument("--model_type", choices=["ar", "dlm"], required=True)
    p.add_argument("--checkpoint_dir", required=True)
    p.add_argument("--dataset", default="iwslt2017")
    p.add_argument("--train_subset_size", type=int, required=True,
                   help="Training subset size used (100/500/1000/5000). Used as x-axis in plots.")
    p.add_argument("--eval_samples", type=int, default=100,
                   help="Number of examples to evaluate on.")
    p.add_argument("--noise_rate", type=float, default=0.0)
    p.add_argument("--max_new_tokens", type=int, default=128,
                   help="Max generation tokens — SAME for AR and DLM.")
    p.add_argument("--dlm_steps", type=int, default=64,
                   help="Refinement steps for DLM decoding (default 64).")
    p.add_argument("--output_file", default="results.json")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def distinct_n(texts, n=2):
    """Distinct-N diversity metric."""
    if not texts:
        return 0.0
    ngrams, total = set(), 0
    for text in texts:
        words = text.split()
        for i in range(len(words) - n + 1):
            ngrams.add(tuple(words[i:i + n]))
            total += 1
    return len(ngrams) / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# [FIX-6] Unified scoring NLL
# ---------------------------------------------------------------------------

@torch.no_grad()
def score_nll_ar(model, tokenizer, dataset, src_col, tgt_col):
    """
    Teacher-forced causal NLL on the reference target (nats/token).
    Source tokens are excluded from the loss via labels=-100.
    """
    model.eval()
    total_nll, total_tokens = 0.0, 0
    for item in tqdm(dataset, desc="Scoring AR"):
        src_prefix = f"Source: {item[f'noisy_{src_col}']}\nTarget: "
        full_text   = src_prefix + item[tgt_col]

        src_len  = len(tokenizer(src_prefix, add_special_tokens=False)["input_ids"])
        full_ids = tokenizer(full_text, return_tensors="pt").input_ids.to(model.device)

        labels = full_ids.clone()
        labels[:, :src_len] = -100          # mask source from loss

        out = model(input_ids=full_ids, labels=labels)
        n_tgt = (labels != -100).sum().item()
        if n_tgt > 0:
            total_nll    += out.loss.item() * n_tgt
            total_tokens += n_tgt

    return total_nll / total_tokens if total_tokens > 0 else float("inf")


@torch.no_grad()
def score_nll_dlm(model, tokenizer, dataset, src_col, tgt_col):
    """
    Masked reconstruction NLL on the reference target (nats/token).
    All target tokens are masked at once (t=1, maximum corruption), and the
    model predicts them in one forward pass. This is the DLM's natural
    scoring mode and is directly comparable to AR teacher-forced NLL.
    """
    model.eval()
    mask_id = tokenizer.mask_token_id
    total_nll, total_tokens = 0.0, 0

    for item in tqdm(dataset, desc="Scoring DLM"):
        src_prefix = f"Source: {item[f'noisy_{src_col}']}\nTarget: "
        full_text   = src_prefix + item[tgt_col]

        src_len  = len(tokenizer(src_prefix, add_special_tokens=False)["input_ids"])
        full_ids = tokenizer(full_text, return_tensors="pt").input_ids.to(model.device)

        # Mask ALL target tokens
        masked_ids = full_ids.clone()
        masked_ids[:, src_len:] = mask_id

        labels = torch.full_like(full_ids, -100)
        labels[:, src_len:] = full_ids[:, src_len:]   # reference target as labels

        out = model(input_ids=masked_ids)
        n_tgt = (labels != -100).sum().item()
        if n_tgt > 0:
            loss = torch.nn.functional.cross_entropy(
                out.logits.view(-1, out.logits.size(-1)), labels.view(-1)
            )
            total_nll    += loss.item() * n_tgt
            total_tokens += n_tgt

    return total_nll / total_tokens if total_tokens > 0 else float("inf")


# ---------------------------------------------------------------------------
# Generation evaluation — AR
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_ar(model, tokenizer, dataset, src_col, tgt_col,
                max_new_tokens: int = 128):
    bleu  = hf_evaluate.load("sacrebleu")
    rouge = hf_evaluate.load("rouge")
    predictions, references = [], []

    model.eval()
    for item in tqdm(dataset, desc="Generating AR"):
        src_text = f"Source: {item[f'noisy_{src_col}']}\nTarget: "
        inputs   = tokenizer(src_text, return_tensors="pt").to(model.device)
        in_len   = inputs.input_ids.shape[1]

        # [FIX-4] max_new_tokens passed in (default 128)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        pred = tokenizer.decode(outputs[0, in_len:], skip_special_tokens=True)
        predictions.append(pred)
        references.append([item[tgt_col]])

    return {
        "bleu":       bleu.compute(predictions=predictions, references=references)["score"],
        "rougeL":     rouge.compute(predictions=predictions, references=references)["rougeL"],
        "distinct_2": distinct_n(predictions, 2),
    }


# ---------------------------------------------------------------------------
# Generation evaluation — DLM
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_dlm(model, tokenizer, dataset, src_col, tgt_col,
                 max_new_tokens: int = 128, steps: int = 64):
    """
    Iterative confidence-based unmasking loop.

    [FIX-4] target_len = max_new_tokens (128), matching AR output capacity.
    [FIX-5] steps = 64 (configurable). Each step is one full forward pass
             over the sequence. Comparable to AR's sequential per-token passes.
    """
    bleu  = hf_evaluate.load("sacrebleu")
    rouge = hf_evaluate.load("rouge")
    predictions, references = [], []

    model.eval()
    mask_id = tokenizer.mask_token_id

    for item in tqdm(dataset, desc="Generating DLM"):
        src_text = f"Source: {item[f'noisy_{src_col}']}\nTarget: "
        inputs   = tokenizer(src_text, return_tensors="pt").to(model.device)
        src_len  = inputs.input_ids.shape[1]

        # [FIX-4] target_len now equals max_new_tokens
        target_len = max_new_tokens
        target_masks = torch.full((1, target_len), mask_id, device=model.device)
        current_seq  = torch.cat([inputs.input_ids, target_masks], dim=1)

        # [FIX-5] configurable steps
        for step in range(steps):
            out    = model(input_ids=current_seq)
            logits = out.logits[0, src_len:]              # target logits only
            probs  = torch.softmax(logits, dim=-1)
            max_probs, max_tokens = torch.max(probs, dim=-1)

            is_masked = (current_seq[0, src_len:] == mask_id)
            if not is_masked.any():
                break

            # Unmask the most confident masked positions this step
            num_to_unmask = max(1, int(is_masked.sum().item() / (steps - step)))
            masked_probs  = max_probs.clone()
            masked_probs[~is_masked] = -1.0
            _, top_indices = torch.topk(masked_probs, num_to_unmask)

            for idx in top_indices:
                current_seq[0, src_len + idx] = max_tokens[idx]

        pred = tokenizer.decode(current_seq[0, src_len:], skip_special_tokens=True)
        pred = pred.replace(tokenizer.mask_token, "").strip()
        predictions.append(pred)
        references.append([item[tgt_col]])

    return {
        "bleu":       bleu.compute(predictions=predictions, references=references)["score"],
        "rougeL":     rouge.compute(predictions=predictions, references=references)["rougeL"],
        "distinct_2": distinct_n(predictions, 2),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    model_name = (
        "meta-llama/Meta-Llama-3-8B-Instruct"
        if args.model_type == "ar"
        else "GSAI-ML/LLaDA-8B-Instruct"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir, trust_remote_code=True)

    bnb_config = BitsAndBytesConfig(load_in_4bit=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    if args.model_type == "dlm" and len(tokenizer) > base_model.config.vocab_size:
        base_model.resize_token_embeddings(len(tokenizer))

    model = PeftModel.from_pretrained(base_model, args.checkpoint_dir)

    # [FIX-7] subset_size is explicit
    dataset, src_col, tgt_col = prepare_dataset(
        args.dataset, args.eval_samples, args.noise_rate
    )

    # Generation metrics
    if args.model_type == "ar":
        gen_results = evaluate_ar(
            model, tokenizer, dataset, src_col, tgt_col,
            max_new_tokens=args.max_new_tokens
        )
    else:
        gen_results = evaluate_dlm(
            model, tokenizer, dataset, src_col, tgt_col,
            max_new_tokens=args.max_new_tokens,
            steps=args.dlm_steps,
        )

    # [FIX-6] Unified scoring NLL (same units, same direction for both models)
    if args.model_type == "ar":
        scoring_nll = score_nll_ar(model, tokenizer, dataset, src_col, tgt_col)
    else:
        scoring_nll = score_nll_dlm(model, tokenizer, dataset, src_col, tgt_col)

    results = {
        **gen_results,
        "scoring_nll":       scoring_nll,
        "model_type":        args.model_type,
        "noise_rate":        args.noise_rate,
        "train_subset_size": args.train_subset_size,  # x-axis: training size
        "eval_samples":      args.eval_samples,        # how many examples evaluated
        "max_new_tokens":    args.max_new_tokens,
        "dlm_steps":         args.dlm_steps if args.model_type == "dlm" else None,
    }

    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {args.output_file}")
    print(results)


if __name__ == "__main__":
    main()
