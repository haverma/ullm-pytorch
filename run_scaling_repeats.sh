#!/usr/bin/env bash
# Run only after the initial single/ddp2/ddp4 measurements have completed.
set -euo pipefail
cd "$(dirname "$0")"
root="${1:-runs/comparison-1300}"
for repeat in 2 3; do
  for gpus in 1 2 4; do
    if [ "$gpus" = 1 ]; then
      label="single-repeat$repeat"
      command=(.venv/bin/python benchmark.py --device cuda:0)
      visible=0
    else
      label="ddp$gpus-repeat$repeat"
      command=(.venv/bin/torchrun --standalone --nproc_per_node="$gpus" train_ddp.py --device cuda)
      if [ "$gpus" = 2 ]; then visible=0,1; else visible=0,1,2,3; fi
    fi
    if [ -e "$root/$label" ]; then
      echo "Refusing to overwrite $root/$label" >&2
      exit 1
    fi
    echo "Starting $label"
    CUDA_VISIBLE_DEVICES="$visible" OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
      "${command[@]}" --seed 42 --batch-size 32 --seq-length 128 \
      --max-steps 1300 --no-profile --run-dir "$root/$label" \
      --checkpoint-dir "checkpoints/comparison-1300/$label" > "$root/$label.log" 2>&1
    echo "Completed $label"
  done
done
