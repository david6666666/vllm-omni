#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# 2x2 timing matrix for SAGE_ATTN_3 vs TORCH_SDPA, compiled vs eager,
# on HunyuanVideo-1.5 480p T2V. Run from the repo root:
#
#     bash benchmarks/diffusion/bench_sage_attn3_matrix.sh
#
# Requires: sageattn3 installed (for SAGE_ATTN_3 rows) and a Blackwell GPU.

set -u

MODEL=${MODEL:-hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v}
PROMPT=${PROMPT:-"A dog running across a field of golden wheat."}
STEPS=${STEPS:-30}
HEIGHT=${HEIGHT:-480}
WIDTH=${WIDTH:-832}
FRAMES=${FRAMES:-33}
SEED=${SEED:-42}
GUIDANCE=${GUIDANCE:-6.0}

OUT_DIR=${OUT_DIR:-/tmp/bench_sage3}
mkdir -p "$OUT_DIR"

declare -A TOTAL PER_STEP TEXT_ENC TRANSFORMER VAE

run() {
    local name="$1"; shift
    local log="$OUT_DIR/${name}.log"
    echo "[$(date +%T)] Running $name ..."
    env "$@" python examples/offline_inference/text_to_video/text_to_video.py \
        --model "$MODEL" \
        --prompt "$PROMPT" \
        --height "$HEIGHT" --width "$WIDTH" --num-frames "$FRAMES" \
        --num-inference-steps "$STEPS" --seed "$SEED" --guidance-scale "$GUIDANCE" \
        --enable-diffusion-pipeline-profiler \
        --output "$OUT_DIR/${name}.mp4" > "$log" 2>&1 || {
            echo "  FAILED — see $log"
            TOTAL[$name]="FAIL"; PER_STEP[$name]="FAIL"
            TEXT_ENC[$name]="-"; TRANSFORMER[$name]="-"; VAE[$name]="-"
            return
        }

    local wait_ms
    wait_ms=$(grep "add_req_and_wait=" "$log" | tail -1 | sed -nE 's/.*add_req_and_wait=([0-9.]+) ms.*/\1/p')
    local total_s
    total_s=$(grep -oE "Total generation time: [0-9.]+ seconds" "$log" | tail -1 | sed -nE 's/.*: ([0-9.]+) seconds.*/\1/p')
    local text_enc_s
    text_enc_s=$(grep -oE "text_encoder.forward took [0-9.]+s" "$log" | awk '{sub(/s$/,"",$NF); sum+=$NF} END {printf "%.3f", sum+0}')
    local vae_s
    vae_s=$(grep -oE "vae.decode took [0-9.]+s" "$log" | awk '{sub(/s$/,"",$NF); sum+=$NF} END {printf "%.3f", sum+0}')
    local pipeline_s
    pipeline_s=$(grep -oE "HunyuanVideo15Pipeline\.forward took [0-9.]+s" "$log" | tail -1 | sed -nE 's/.*took ([0-9.]+)s.*/\1/p')
    local transformer_s
    transformer_s=$(awk "BEGIN {printf \"%.3f\", ${pipeline_s:-0} - ${vae_s:-0} - ${text_enc_s:-0}}")

    TOTAL[$name]="${total_s:-0}"
    PER_STEP[$name]=$(awk "BEGIN {printf \"%.3f\", ${wait_ms:-0} / 1000 / $STEPS}")
    TEXT_ENC[$name]="$text_enc_s"
    TRANSFORMER[$name]="$transformer_s"
    VAE[$name]="$vae_s"

    echo "  total=${TOTAL[$name]}s  per-step=${PER_STEP[$name]}s/it  transformer=${TRANSFORMER[$name]}s  vae=${VAE[$name]}s"
}

run sdpa_eager     TORCH_COMPILE_DISABLE=1
run sdpa_compiled
run sage3_eager    TORCH_COMPILE_DISABLE=1 DIFFUSION_ATTENTION_BACKEND=SAGE_ATTN_3
run sage3_compiled DIFFUSION_ATTENTION_BACKEND=SAGE_ATTN_3

echo
echo "HunyuanVideo-1.5 ${WIDTH}x${HEIGHT}x${FRAMES}, ${STEPS} steps, seed=${SEED}"
echo
printf "| %-23s | %-9s | %-15s | %-12s | %-15s | %-14s |\n" \
    "Config" "Total (s)" "Per-step (s/it)" "Text Enc (s)" "Transformer (s)" "VAE Decode (s)"
printf "| %-23s | %-9s | %-15s | %-12s | %-15s | %-14s |\n" \
    "-----------------------" "---------" "---------------" "------------" "---------------" "--------------"
for row in sdpa_eager sdpa_compiled sage3_eager sage3_compiled; do
    case "$row" in
        sdpa_eager)     label="SDPA + Eager" ;;
        sdpa_compiled)  label="SDPA + Compiled" ;;
        sage3_eager)    label="SAGE_ATTN_3 + Eager" ;;
        sage3_compiled) label="SAGE_ATTN_3 + Compiled" ;;
    esac
    printf "| %-23s | %-9s | %-15s | %-12s | %-15s | %-14s |\n" \
        "$label" "${TOTAL[$row]}" "${PER_STEP[$row]}" "${TEXT_ENC[$row]}" "${TRANSFORMER[$row]}" "${VAE[$row]}"
done

echo
echo "Logs: $OUT_DIR/*.log"
