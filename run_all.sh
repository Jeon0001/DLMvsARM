#!/bin/bash
# run_all.sh — Fair DLM vs ARM Comparison
# =========================================
# All 4 subset sizes run in parallel, one GPU each (simple, quality-preserving).
# Each GPU runs its own single-process job so batch_size=2 is never inflated.
#
# Fixes vs. DLMvsARM_Prelim/run_all.sh:
#   [FIX] MODELS=("ar" "dlm") — both models trained in the same run.
#   [FIX] --subset_size passed explicitly to eval.py (not parsed from path).
#   [FIX] --max_new_tokens 128 and --dlm_steps 64 for fair eval budget.

set -e

# ---- Configuration ----
MODELS=("ar" "dlm")
NOISE_RATES=(0.0 0.1 0.3)
SUBSET_SIZES=(100 500 1000 5000)
GPUS=(0 1 2 3)

DATASET="${DATASET:-gigaword}"
DATA_DIR="${DATA_DIR:-/root/ssd/dataset/gigaword_10k_size_subsets/csv}"
EVAL_SPLIT="${EVAL_SPLIT:-validation}"

DATA_ARGS=(--dataset "$DATASET")
if [ -n "$DATA_DIR" ]; then
    DATA_ARGS+=(--data_dir "$DATA_DIR")
fi

export WANDB_MODE=offline

mkdir -p checkpoints results plots

echo "============================================================"
echo " Fair DLM vs ARM — Experiment Runner"
echo " Models     : ${MODELS[*]}"
echo " Dataset    : $DATASET"
echo " Data dir   : ${DATA_DIR:-<huggingface>}"
echo " Eval split : $EVAL_SPLIT"
echo " Noise rates: ${NOISE_RATES[*]}"
echo " Sizes      : ${SUBSET_SIZES[*]}  (all parallel, 1 GPU each)"
echo "============================================================"

echo "[Step 0] Pre-downloading models to cache..."
python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
print('[1/2] Caching LLaMA-3-8B-Instruct...')
AutoModelForCausalLM.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct')
AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct')
print('[2/2] Caching LLaDA-8B-Instruct...')
AutoModelForCausalLM.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)
AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)
print('All models cached.')
"

for MODEL in "${MODELS[@]}"; do
    for NOISE in "${NOISE_RATES[@]}"; do

        echo ""
        echo "============================================================"
        echo " Training | model=$MODEL | noise=$NOISE"
        echo "============================================================"

        # All 4 sizes in parallel, one GPU each, --num_processes=1 (no DDP)
        PIDS=()
        for i in "${!SUBSET_SIZES[@]}"; do
            SIZE="${SUBSET_SIZES[$i]}"
            GPU="${GPUS[$i]}"

            CUDA_VISIBLE_DEVICES=$GPU accelerate launch \
                --num_processes=1 \
                train.py \
                    --model_type "$MODEL" \
                    --subset_size "$SIZE" \
                    --noise_rate "$NOISE" \
                    "${DATA_ARGS[@]}" \
                    --batch_size 2 \
                    --epochs 3 \
                    --output_dir checkpoints \
                > checkpoints/${MODEL}_${SIZE}_noise${NOISE}.log 2>&1 &

            PIDS+=($!)
            echo "  size=$SIZE → GPU $GPU (PID ${PIDS[-1]})"
        done

        echo "Waiting for all sizes to finish..."
        wait "${PIDS[@]}"
        echo "Training done."

        # ---- Evaluation (all sizes in parallel, one GPU each) ----
        echo "Running evaluation..."
        EVAL_PIDS=()
        for i in "${!SUBSET_SIZES[@]}"; do
            SIZE="${SUBSET_SIZES[$i]}"
            GPU="${GPUS[$i]}"

            CUDA_VISIBLE_DEVICES=$GPU python3 eval.py \
                --model_type "$MODEL" \
                --checkpoint_dir "checkpoints/${MODEL}_${DATASET}_${SIZE}_noise${NOISE}" \
                "${DATA_ARGS[@]}" \
                --eval_split "$EVAL_SPLIT" \
                --train_subset_size "$SIZE" \
                --eval_samples 100 \
                --noise_rate "$NOISE" \
                --max_new_tokens 128 \
                $( [ "$MODEL" = "dlm" ] && echo "--dlm_steps 64" ) \
                --output_file "results/${MODEL}_${DATASET}_${SIZE}_noise${NOISE}.json" \
                > results/eval_${MODEL}_${DATASET}_${SIZE}_noise${NOISE}.log 2>&1 &

            EVAL_PIDS+=($!)
        done

        echo "Waiting for evaluation to finish..."
        wait "${EVAL_PIDS[@]}"
        echo "Evaluation done."

    done
done

echo ""
echo "============================================================"
echo " All done. Generating plots..."
echo "============================================================"
python3 plot_results.py --results_dir results
echo "Plots saved to plots/. Experiment complete."


