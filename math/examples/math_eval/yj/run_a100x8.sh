#!/usr/bin/env bash
# Default is a dry plan. GPU work requires the explicit --execute flag.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${TRAIN_PY:?Set TRAIN_PY to the training environment Python executable}"
: "${EVAL_PY:?Set EVAL_PY to the existing math_eval_a100.py environment Python executable}"
YJ_WORKSPACE="${YJ_WORKSPACE:-yj/studies/a100x8_v1}"
read -r -a yj_gpu_slots <<< "${YJ_GPU_IDS:-0 1 2 3 4 5 6 7}"
yj_phase="${1:-plan}"
if (( $# > 0 )); then shift; fi
yj_execute=false
yj_resume=false
for yj_arg in "$@"; do
  case "$yj_arg" in
    --execute) yj_execute=true ;;
    --resume-completed) yj_resume=true ;;
    *) echo "Unsupported argument: $yj_arg" >&2; exit 2 ;;
  esac
done
yj_entry=("$TRAIN_PY" yj/train.py research --workspace "$YJ_WORKSPACE")
run_phase() {
  local yj_command=("${yj_entry[@]}" launch --phase "$1" --gpus "${yj_gpu_slots[@]}" --python "$EVAL_PY")
  if "$yj_execute"; then yj_command+=(--execute); fi
  if "$yj_resume"; then yj_command+=(--resume-completed); fi
  "${yj_command[@]}"
}
case "$yj_phase" in
  plan)
    if "$yj_execute"; then echo 'plan does not accept --execute' >&2; exit 2; fi
    run_phase main
    ;;
  pretrain|diagnose|main|contamination) run_phase "$yj_phase" ;;
  all)
    run_phase pretrain
    run_phase diagnose
    run_phase main
    if "$yj_execute"; then "${yj_entry[@]}" statistics --step 300; fi
    ;;
  *) echo 'Usage: bash yj/run_a100x8.sh {plan|pretrain|diagnose|main|contamination|all} [--execute] [--resume-completed]' >&2; exit 2 ;;
esac
