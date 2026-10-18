#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash yj/run_eval.sh RUN_NAME MODEL_PATH [--dry-run]

Evaluate a local model with the existing math_eval_a100.py and the shared
yj/artifacts/seed42/eval_data dataset. Results go to yj/runs/RUN_NAME/eval.
Set CUDA_VISIBLE_DEVICES to select GPUs (default: 0).
--dry-run prints the command without loading data or a model.
EOF
}

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
  usage
  exit 0
fi
if [[ $# -lt 2 || $# -gt 3 ]]; then
  usage >&2
  exit 2
fi
if [[ $# -eq 3 && $3 != --dry-run ]]; then
  usage >&2
  exit 2
fi

run_name=$1
if [[ ! $run_name =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]]; then
  echo 'RUN_NAME must use letters, digits, underscores or hyphens.' >&2
  exit 2
fi

# Resolve the model relative to the caller before changing to the evaluator root.
model_path=$2
if [[ $model_path != /* ]]; then
  model_path="$PWD/$model_path"
fi
yj_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
eval_root="$(cd -- "$yj_dir/.." && pwd)"
data_dir="$yj_dir/artifacts/seed42/eval_data"
output_dir="$yj_dir/runs/$run_name/eval"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}"

command=(
  python "$eval_root/math_eval_a100.py"
  --model_name_or_path "$model_path"
  --data_names math --data_dir "$data_dir"
  --split test --num_test_sample -1 --prompt_type cot
  --seed 42 --temperature 0 --n_sampling 1
  --use_vllm --save_outputs --output_dir "$output_dir"
)

if [[ ${3:-} == --dry-run ]]; then
  printf 'cd %q\n' "$eval_root"
  printf 'CUDA_VISIBLE_DEVICES=%q ' "$CUDA_VISIBLE_DEVICES"
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi

if [[ ! -f "$data_dir/math/test.jsonl" ]]; then
  echo "Evaluation data missing. Run: python \"$yj_dir/prepare_data.py\"" >&2
  exit 1
fi
if [[ ! -d "$model_path" ]]; then
  echo "Local model directory not found: $model_path" >&2
  exit 1
fi
if [[ -z $CUDA_VISIBLE_DEVICES ]]; then
  echo 'Set CUDA_VISIBLE_DEVICES to the A100 GPU index, for example 0.' >&2
  exit 1
fi

cd -- "$eval_root"
exec "${command[@]}"
