"""
plot_results.py — Fair DLM vs ARM Comparison
=============================================
Reads all JSON result files from results/ and produces:
  1. BLEU vs. Subset Size (per noise level)
  2. ROUGE-L vs. Subset Size (per noise level)
  3. Scoring NLL vs. Subset Size (unified metric, same direction for both models)
  4. Distinct-2 vs. Subset Size (diversity)
  5. BLEU vs. Noise Rate (robustness heatmap)
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


MODELS = {"ar": "LLaMA-3 (AR)", "dlm": "LLaDA (DLM)"}
COLORS = {"ar": "#4C8BF5", "dlm": "#F5754C"}
LINESTYLES = {"ar": "-o", "dlm": "--s"}
NOISE_LABELS = {0.0: "Clean (0%)", 0.1: "10% noise", 0.3: "30% noise"}


def load_results(results_dir: str) -> list[dict]:
    records = []
    for path in Path(results_dir).glob("*.json"):
        if path.name.startswith("eval_"):
            continue
        with open(path) as f:
            records.append(json.load(f))
    return records


def subset_size_plot(records, metric, ylabel, title, out_path, noise_rates):
    fig, axes = plt.subplots(1, len(noise_rates), figsize=(5 * len(noise_rates), 5),
                             sharey=True)
    if len(noise_rates) == 1:
        axes = [axes]

    for ax, noise in zip(axes, noise_rates):
        for model_key, model_label in MODELS.items():
            pts = sorted(
                [r for r in records
                 if r["model_type"] == model_key
                 and abs(r["noise_rate"] - noise) < 1e-6],
                key=lambda r: r["train_subset_size"]
            )
            if not pts:
                continue
            xs = [r["train_subset_size"] for r in pts]
            ys = [r[metric] for r in pts]
            ax.plot(xs, ys, LINESTYLES[model_key],
                    color=COLORS[model_key], label=model_label, linewidth=2)

        ax.set_title(NOISE_LABELS.get(noise, f"noise={noise}"), fontsize=12)
        ax.set_xlabel("Training examples")
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()

    axes[0].set_ylabel(ylabel)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


def noise_robustness_table(records, metric, ylabel, out_path, subset_sizes):
    """Bar chart: metric vs noise rate for each subset size."""
    noise_vals = sorted({r["noise_rate"] for r in records})
    x = np.arange(len(noise_vals))
    width = 0.35

    fig, axes = plt.subplots(1, len(subset_sizes), figsize=(5 * len(subset_sizes), 5),
                             sharey=True)
    if len(subset_sizes) == 1:
        axes = [axes]

    for ax, size in zip(axes, subset_sizes):
        ar_vals  = [next((r[metric] for r in records
                         if r["model_type"] == "ar"
                         and r["train_subset_size"] == size
                         and abs(r["noise_rate"] - n) < 1e-6), 0.0)
                    for n in noise_vals]
        dlm_vals = [next((r[metric] for r in records
                          if r["model_type"] == "dlm"
                          and r["train_subset_size"] == size
                          and abs(r["noise_rate"] - n) < 1e-6), 0.0)
                    for n in noise_vals]

        ax.bar(x - width/2, ar_vals,  width, label="LLaMA-3 (AR)",  color=COLORS["ar"])
        ax.bar(x + width/2, dlm_vals, width, label="LLaDA (DLM)", color=COLORS["dlm"])
        ax.set_xticks(x)
        ax.set_xticklabels([f"{int(n*100)}%" for n in noise_vals])
        ax.set_xlabel("Noise rate")
        ax.set_title(f"n={size}", fontsize=12)
        ax.legend()
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    axes[0].set_ylabel(ylabel)
    fig.suptitle(f"{ylabel} vs Noise Rate", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results")
    p.add_argument("--plots_dir", default="plots")
    args = p.parse_args()

    os.makedirs(args.plots_dir, exist_ok=True)
    records = load_results(args.results_dir)
    if not records:
        print("No result JSON files found. Run eval.py first.")
        return

    noise_rates   = sorted({r["noise_rate"]        for r in records})
    subset_sizes  = sorted({r["train_subset_size"] for r in records})

    # 1. BLEU vs subset size
    subset_size_plot(records, "bleu", "BLEU Score",
                     "BLEU vs. Training Size",
                     f"{args.plots_dir}/bleu_vs_size.png", noise_rates)

    # 2. ROUGE-L vs subset size
    subset_size_plot(records, "rougeL", "ROUGE-L",
                     "ROUGE-L vs. Training Size",
                     f"{args.plots_dir}/rouge_vs_size.png", noise_rates)

    # 3. Scoring NLL vs subset size  [FIX-6: unified metric]
    subset_size_plot(records, "scoring_nll", "Scoring NLL (nats/token, ↓ better)",
                     "Teacher-Forced / Masked NLL vs. Training Size",
                     f"{args.plots_dir}/nll_vs_size.png", noise_rates)

    # 4. Distinct-2 vs subset size
    subset_size_plot(records, "distinct_2", "Distinct-2",
                     "Output Diversity vs. Training Size",
                     f"{args.plots_dir}/distinct2_vs_size.png", noise_rates)

    # 5. Noise robustness
    noise_robustness_table(records, "bleu", "BLEU",
                           f"{args.plots_dir}/noise_robustness_bleu.png",
                           subset_sizes)
    noise_robustness_table(records, "scoring_nll", "Scoring NLL (↓ better)",
                           f"{args.plots_dir}/noise_robustness_nll.png",
                           subset_sizes)

    print("\nAll plots saved to", args.plots_dir)


if __name__ == "__main__":
    main()
