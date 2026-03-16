#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Paths
DATA_ROOT="${DATA_ROOT:-/data/bench_data}"
GEBENCH_ROOT="${GEBENCH_ROOT:-${DATA_ROOT}/GEBench}"
GEDIT_ROOT="${GEDIT_ROOT:-${DATA_ROOT}/GEdit-Bench}"

GEBENCH_OUT="${GEBENCH_OUT:-${REPO_ROOT}/benchmarks/accuracy/text_to_image/outputs}"
GEDIT_OUT="${GEDIT_OUT:-${REPO_ROOT}/benchmarks/accuracy/image_to_image/results}"
GEDIT_SCORE="${GEDIT_SCORE:-${REPO_ROOT}/benchmarks/accuracy/image_to_image/scores}"

# Service endpoints
GEN_T2I_URL="${GEN_T2I_URL:-http://127.0.0.1:8000}"
GEN_EDIT_URL="${GEN_EDIT_URL:-http://127.0.0.1:8001}"
JUDGE_URL="${JUDGE_URL:-http://127.0.0.1:8002}"
API_KEY="${API_KEY:-EMPTY}"

# Models
QWEN_IMAGE_MODEL="${QWEN_IMAGE_MODEL:-Qwen/Qwen-Image}"
QWEN_IMAGE_EDIT_MODEL="${QWEN_IMAGE_EDIT_MODEL:-Qwen/Qwen-Image-Edit}"
JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
GEDIT_MODEL_NAME="${GEDIT_MODEL_NAME:-qwen_image_edit}"

# Benchmark execution knobs
SMOKE_SAMPLES="${SMOKE_SAMPLES:-5}"
WORKERS="${WORKERS:-4}"

# Serve template. Override if your deployment uses a different entrypoint.
SERVE_CMD_PREFIX="${SERVE_CMD_PREFIX:-python -m vllm_omni.entrypoints.openai.api_server}"

usage() {
  cat <<'EOF'
Usage:
  benchmarks/accuracy/run_accuracy_bench.sh <command>

Commands:
  init-dirs               Create local dataset and output directories
  download-gebench        Clone the GEBench dataset from Hugging Face
  download-gedit          Download GEdit-Bench with datasets.save_to_disk
  serve-qwen-image        Start local Qwen-Image service on port 8000
  serve-qwen-image-edit   Start local Qwen-Image-Edit service on port 8001
  serve-judge             Start local judge service on port 8002
  gebench-smoke           Run a small GEBench smoke workflow
  gebench-full            Run all GEBench types with model routing
  gedit-smoke             Run a small GEdit-Bench smoke workflow
  gedit-full              Run full GEdit-Bench generation/evaluation
  summarize-gebench       Recompute GEBench summary
  summarize-gedit         Recompute GEdit-Bench summary from all/all CSV
  all-smoke               Run GEBench smoke + GEdit-Bench smoke

Environment overrides:
  DATA_ROOT, GEBENCH_ROOT, GEDIT_ROOT
  GEBENCH_OUT, GEDIT_OUT, GEDIT_SCORE
  GEN_T2I_URL, GEN_EDIT_URL, JUDGE_URL, API_KEY
  QWEN_IMAGE_MODEL, QWEN_IMAGE_EDIT_MODEL, JUDGE_MODEL
  GEDIT_MODEL_NAME, SMOKE_SAMPLES, WORKERS
  SERVE_CMD_PREFIX

Notes:
  - Service commands are long-running foreground processes.
  - Start `serve-judge`, `serve-qwen-image`, and `serve-qwen-image-edit`
    in separate shells before running benchmark commands.
  - GEBench types 3/4 use Qwen-Image. Types 1/2/5 use Qwen-Image-Edit.
EOF
}

log() {
  printf '[accuracy-bench] %s\n' "$*"
}

init_dirs() {
  mkdir -p "${DATA_ROOT}" "${GEBENCH_OUT}" "${GEDIT_OUT}" "${GEDIT_SCORE}"
}

download_gebench() {
  init_dirs
  if [[ -d "${GEBENCH_ROOT}/.git" || -f "${GEBENCH_ROOT}/README.md" ]]; then
    log "GEBench dataset already exists at ${GEBENCH_ROOT}"
    return 0
  fi
  git clone https://huggingface.co/datasets/stepfun-ai/GEBench "${GEBENCH_ROOT}"
}

download_gedit() {
  init_dirs
  if [[ -d "${GEDIT_ROOT}" && -f "${GEDIT_ROOT}/dataset_dict.json" ]]; then
    log "GEdit-Bench dataset already exists at ${GEDIT_ROOT}"
    return 0
  fi
  GEDIT_ROOT_ESCAPED="${GEDIT_ROOT}" python - <<'PY'
import os
from datasets import load_dataset

target = os.environ["GEDIT_ROOT_ESCAPED"]
ds = load_dataset("stepfun-ai/GEdit-Bench")
ds.save_to_disk(target)
PY
}

serve_model() {
  local model="$1"
  local port="$2"
  cd "${REPO_ROOT}"
  eval "${SERVE_CMD_PREFIX} --model ${model@Q} --port ${port@Q}"
}

serve_qwen_image() {
  serve_model "${QWEN_IMAGE_MODEL}" 8000
}

serve_qwen_image_edit() {
  serve_model "${QWEN_IMAGE_EDIT_MODEL}" 8001
}

serve_judge() {
  serve_model "${JUDGE_MODEL}" 8002
}

run_gebench_generate() {
  local data_type="$1"
  local base_url="$2"
  local model="$3"
  local maybe_max_samples=()
  if [[ $# -ge 4 && -n "$4" ]]; then
    maybe_max_samples=(--max-samples "$4")
  fi

  cd "${REPO_ROOT}"
  python benchmarks/accuracy/text_to_image/run_gebench.py generate \
    --dataset-root "${GEBENCH_ROOT}" \
    --output-root "${GEBENCH_OUT}" \
    --base-url "${base_url}" \
    --model "${model}" \
    --api-key "${API_KEY}" \
    --data-type "${data_type}" \
    --workers "${WORKERS}" \
    "${maybe_max_samples[@]}"
}

run_gebench_evaluate() {
  local data_type="$1"
  local maybe_max_samples=()
  if [[ $# -ge 2 && -n "$2" ]]; then
    maybe_max_samples=(--max-samples "$2")
  fi

  cd "${REPO_ROOT}"
  python benchmarks/accuracy/text_to_image/run_gebench.py evaluate \
    --dataset-root "${GEBENCH_ROOT}" \
    --output-root "${GEBENCH_OUT}" \
    --data-type "${data_type}" \
    --judge-base-url "${JUDGE_URL}" \
    --judge-model "${JUDGE_MODEL}" \
    --judge-api-key "${API_KEY}" \
    --workers "${WORKERS}" \
    "${maybe_max_samples[@]}"
}

summarize_gebench() {
  cd "${REPO_ROOT}"
  python benchmarks/accuracy/text_to_image/run_gebench.py summarize \
    --output-root "${GEBENCH_OUT}"
}

gebench_smoke() {
  run_gebench_generate type3 "${GEN_T2I_URL}" "${QWEN_IMAGE_MODEL}" "${SMOKE_SAMPLES}"
  run_gebench_evaluate type3 "${SMOKE_SAMPLES}"

  run_gebench_generate type1 "${GEN_EDIT_URL}" "${QWEN_IMAGE_EDIT_MODEL}" "${SMOKE_SAMPLES}"
  run_gebench_evaluate type1 "${SMOKE_SAMPLES}"

  summarize_gebench
}

gebench_full() {
  run_gebench_generate type3 "${GEN_T2I_URL}" "${QWEN_IMAGE_MODEL}"
  run_gebench_evaluate type3

  run_gebench_generate type4 "${GEN_T2I_URL}" "${QWEN_IMAGE_MODEL}"
  run_gebench_evaluate type4

  run_gebench_generate type1 "${GEN_EDIT_URL}" "${QWEN_IMAGE_EDIT_MODEL}"
  run_gebench_evaluate type1

  run_gebench_generate type2 "${GEN_EDIT_URL}" "${QWEN_IMAGE_EDIT_MODEL}"
  run_gebench_evaluate type2

  run_gebench_generate type5 "${GEN_EDIT_URL}" "${QWEN_IMAGE_EDIT_MODEL}"
  run_gebench_evaluate type5

  summarize_gebench
}

run_gedit_generate() {
  local task_type="$1"
  local instruction_language="$2"
  local maybe_max_samples=()
  if [[ $# -ge 3 && -n "$3" ]]; then
    maybe_max_samples=(--max-samples "$3")
  fi

  cd "${REPO_ROOT}"
  python benchmarks/accuracy/image_to_image/run_gedit_bench.py generate \
    --dataset-ref "${GEDIT_ROOT}" \
    --output-root "${GEDIT_OUT}" \
    --base-url "${GEN_EDIT_URL}" \
    --model "${QWEN_IMAGE_EDIT_MODEL}" \
    --model-name "${GEDIT_MODEL_NAME}" \
    --api-key "${API_KEY}" \
    --task-type "${task_type}" \
    --instruction-language "${instruction_language}" \
    --workers "${WORKERS}" \
    "${maybe_max_samples[@]}"
}

run_gedit_evaluate() {
  local task_type="$1"
  local instruction_language="$2"
  local maybe_max_samples=()
  if [[ $# -ge 3 && -n "$3" ]]; then
    maybe_max_samples=(--max-samples "$3")
  fi

  cd "${REPO_ROOT}"
  python benchmarks/accuracy/image_to_image/run_gedit_bench.py evaluate \
    --dataset-ref "${GEDIT_ROOT}" \
    --output-root "${GEDIT_OUT}" \
    --model-name "${GEDIT_MODEL_NAME}" \
    --save-dir "${GEDIT_SCORE}" \
    --task-type "${task_type}" \
    --instruction-language "${instruction_language}" \
    --judge-base-url "${JUDGE_URL}" \
    --judge-model "${JUDGE_MODEL}" \
    --judge-api-key "${API_KEY}" \
    --workers "${WORKERS}" \
    "${maybe_max_samples[@]}"
}

summarize_gedit() {
  cd "${REPO_ROOT}"
  python benchmarks/accuracy/image_to_image/run_gedit_bench.py summarize \
    --csv-path "${GEDIT_SCORE}/${GEDIT_MODEL_NAME}_all_all_vie_score.csv" \
    --language all
}

gedit_smoke() {
  run_gedit_generate background_change en "${SMOKE_SAMPLES}"
  run_gedit_evaluate background_change en "${SMOKE_SAMPLES}"

  cd "${REPO_ROOT}"
  python benchmarks/accuracy/image_to_image/run_gedit_bench.py summarize \
    --csv-path "${GEDIT_SCORE}/${GEDIT_MODEL_NAME}_background_change_en_vie_score.csv" \
    --language en
}

gedit_full() {
  run_gedit_generate all all
  run_gedit_evaluate all all
  summarize_gedit
}

all_smoke() {
  gebench_smoke
  gedit_smoke
}

main() {
  local command="${1:-}"
  case "${command}" in
    init-dirs) init_dirs ;;
    download-gebench) download_gebench ;;
    download-gedit) download_gedit ;;
    serve-qwen-image) serve_qwen_image ;;
    serve-qwen-image-edit) serve_qwen_image_edit ;;
    serve-judge) serve_judge ;;
    gebench-smoke) gebench_smoke ;;
    gebench-full) gebench_full ;;
    gedit-smoke) gedit_smoke ;;
    gedit-full) gedit_full ;;
    summarize-gebench) summarize_gebench ;;
    summarize-gedit) summarize_gedit ;;
    all-smoke) all_smoke ;;
    ""|-h|--help|help) usage ;;
    *)
      printf 'Unknown command: %s\n\n' "${command}" >&2
      usage
      return 1
      ;;
  esac
}

main "$@"
