#!/usr/bin/env bash
# Pass a fresh absolute output root for an independent experiment.
set -euo pipefail
cd "$(dirname "$0")/../.."
root="${1:-$PWD/runs/single-gpu-b200-2026-11-08}"
findings="${2:-$PWD/findings/single-gpu-b200-2026-11-08}"
mkdir -p "$root"
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
.venv/bin/python data.py > "$root/data-preparation.log"
.venv/bin/python experiments/single_gpu/checks.py > "$root/correctness.json"
for stage in performance memory pipeline numerical compile; do
  .venv/bin/python experiments/single_gpu/suite.py --root "$root" --stage "$stage"
done
if [ ! -f "$root/resume-after-fix/result.json" ]; then
  .venv/bin/python experiments/single_gpu/resume_check.py --output "$root/resume-after-fix"
fi
checkpoint="$root/fp32-baseline-r1/checkpoint.pt"
if [ ! -f "$root/generation-check/result.json" ]; then
  .venv/bin/python experiments/single_gpu/generation_check.py \
    --checkpoint "$checkpoint" --output "$root/generation-check" > "$root/generation-check.log" 2>&1
fi
for backend in auto no-cudnn; do
  extra=()
  if [ "$backend" = no-cudnn ]; then extra=(--without-cudnn); fi
  if [ ! -f "$root/generation-backend-$backend/result.json" ]; then
    .venv/bin/python experiments/single_gpu/generation_backend.py \
      --checkpoint "$checkpoint" --output "$root/generation-backend-$backend" "${extra[@]}" \
      > "$root/generation-backend-$backend.log" 2>&1
  fi
done
.venv/bin/python experiments/single_gpu/verify.py --root "$root" --output "$findings"
.venv/bin/python experiments/single_gpu/analyze.py --root "$root" --output "$findings"
echo "Completed lab. Compact findings: $findings"
