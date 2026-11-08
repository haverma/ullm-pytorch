# Comparing the original trainer with NCCL DDP

`train.py` remains the original single-device trainer. `train_ddp.py` runs the
same model with one process per NVIDIA GPU and NCCL gradient synchronization.
`benchmark.py` measures the original trainer without changing its optimization
loop. Both comparison entry points set one CPU intra-op thread per process.

## Workload and fairness

All measured runs use FP32 eager execution, seed 42, sequence length 128,
global batch 32, 1,300 optimizer steps, and the original model defaults
(192 embedding dimensions, six heads, six layers). Each run processes
5,324,800 training token positions. Validation uses 20 batches of 32 every
100 steps; generation uses 80 new tokens; checkpoints are saved at steps
500, 1,000, and 1,300. The cosine schedule ends at step 1,300.

For DDP, `--batch-size` means **global** batch size. Two GPUs each process 16
examples; four GPUs each process eight. Increasing GPU count does not increase
the global batch, learning rate, token count, or number of optimizer updates.

`ShardBatches` slices the original seeded global batch sampler. Every rank
generates the same index permutation but loads only its own contiguous slice.
This deliberately retains the single-device trainer's exact batch membership
and order. It avoids padding/epoch-order changes from switching samplers during
this controlled experiment. Index-permutation work is repeated on each rank.

All ranks train through DDP. Only rank 0 evaluates and generates through the
underlying module; the other ranks wait at a barrier. Checkpoints and weight
logs are written only by rank 0. This preserves the original validation workload
but leaves serial work that limits scaling. Each rank writes its own profile.

## Run on the B200 node

```bash
ssh ubuntu@151.106.165.224
cd /home/ubuntu/ullm-pytorch

# Original trainer, measured without profiler overhead.
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 .venv/bin/python benchmark.py \
  --device cuda:0 --seed 42 --batch-size 32 --seq-length 128 --max-steps 1300 \
  --no-profile --run-dir runs/my-comparison/single \
  --checkpoint-dir checkpoints/my-comparison/single

# Verify sharding and gradient equivalence first.
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 \
  .venv/bin/torchrun --standalone --nproc_per_node=2 check_ddp.py

# Same training workload distributed over two GPUs.
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 \
  .venv/bin/torchrun --standalone --nproc_per_node=2 train_ddp.py \
  --device cuda --seed 42 --batch-size 32 --seq-length 128 --max-steps 1300 \
  --no-profile --run-dir runs/my-comparison/ddp2 \
  --checkpoint-dir checkpoints/my-comparison/ddp2
```

For four GPUs, expose `0,1,2,3`, set `--nproc_per_node=4`, and use fresh output
directories. For a separate diagnostic run, replace `--no-profile` with
`--profile` and use another fresh directory. Profiling skips 20 steps, warms up
for two, and captures five (training updates 23–27). Open the exported
`trace.json` in a Chrome-trace-compatible viewer. `operators.txt` lists operation
timings. Diagnostic runs continue through all 1,300 updates; their elapsed times
are excluded from performance comparisons.

After initial runs named `single`, `ddp2`, and `ddp4`,
`bash run_scaling_repeats.sh` performs two additional unprofiled runs of each
configuration, sequentially. It refuses to overwrite repeat directories.
`python analyze_scaling.py` aggregates their median runtimes and profile data
into `runs/comparison-1300/comparison.json`.

## How to interpret measurements

- `wall_seconds` measures the trainer function through its final checkpoint,
  including model/data setup, DDP initialization where applicable, evaluation,
  generation, logging, and checkpoint writing. Python interpreter/import and
  torchrun parent-process startup are excluded. DDP's final replica checksum
  audit and memory-gather are excluded. Rank-0 initial checksum verification is
  included in DDP setup.
- End-to-end tokens/sec is 5,324,800 divided by that elapsed time. Report the
  median of three runs and their range; these are short, single-node experiments.
- Legacy `metrics.jsonl` throughput ends before the current validation and
  checkpoint. It includes previous overhead and should not be mistaken for
  pure training-step throughput.
- Peak memory refers to PyTorch allocated/reserved memory, not all device usage.
  DDP reports every rank, and rank 0 also handles full-batch validation. NCCL
  allocations outside the PyTorch allocator may not be represented.
- GPU kernel times in profiles are sums of device events. They can overlap,
  especially communication and backward compute, and are not additive wall time.
  NCCL kernels may wait for peers. Profiled CPU step durations include profiler
  overhead and are not used to claim speedups.
- Small loss/parameter differences can result from different floating-point
  reduction orders. Matching seed and global batch does not promise bit-exact
  trajectories across GPU counts.

## Correctness and scope

`check_ddp.py` checks eight actual input/target batches and compares all model
gradients against a single-device full-batch reference. Each full DDP run checks
that every rank's final parameter checksum is identical.

This comparison trainer supports fresh runs only and explicitly rejects
`--resume`. Rank-0 checkpoints preserve the existing format; they are not a
complete per-rank recovery implementation. No mixed precision, compilation,
FSDP, elastic recovery, or multi-node networking is introduced in this experiment.
