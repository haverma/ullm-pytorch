# Single-GPU performance and debugging lab

All jobs run sequentially on GPU 0 of the B200 node. This directory contains
experiment code; [findings](../../findings/single-gpu-b200-2026-11-08/README.md)
contain the analysis and compact evidence. Raw logs, checkpoint tensors,
allocator snapshots, compiler caches, and traces live under the ignored
`runs/single-gpu-b200-2026-11-08/` directory.

## Components

- `worker.py`: one isolated job with CPU wall timing, CUDA-event timing,
  FP32/BF16, accumulation, block activation checkpointing, loader controls,
  failure injection, snapshots, and full-graph compilation.
- `checks.py`: compare all gradients against the original TinyLM, including
  accumulation and activation checkpointing.
- `suite.py`: sequential experiment groups; every subprocess gets an isolated
  output directory and a timeout. Expected OOM and non-finite outcomes are
  persisted rather than discarded. Existing complete results are reused;
  incomplete output directories require inspection before another attempt.
- `resume_check.py`: exercise original `train.py` checkpoint/resume under
  deterministic settings; compare full model, optimizer, scheduler, and RNG state.
- `generation_check.py` and `generation_backend.py`: investigate cold versus
  warm generation and isolate automatic versus cuDNN-disabled SDPA selection.
- `analyze.py`: create compact evidence and a manifest of raw artifacts.
- `plot.py`: optional matplotlib visualization of the extracted findings.
- `verify.py`: audit expected outcomes, workloads, initialization, gradients,
  deterministic resume, and compiled/eager parameter and loss differences.
- `run_all.sh`: run the full suite sequentially; completed results are reused.

## Run on the node

```bash
ssh ubuntu@151.106.165.224
cd /home/ubuntu/ullm-pytorch
# Full suite (pass another absolute output root for independent trials):
bash experiments/single_gpu/run_all.sh

# Individual stages:
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  .venv/bin/python experiments/single_gpu/checks.py
.venv/bin/python experiments/single_gpu/suite.py --stage performance
.venv/bin/python experiments/single_gpu/suite.py --stage memory
.venv/bin/python experiments/single_gpu/suite.py --stage pipeline
.venv/bin/python experiments/single_gpu/suite.py --stage numerical
.venv/bin/python experiments/single_gpu/suite.py --stage compile

CUDA_VISIBLE_DEVICES=0 .venv/bin/python experiments/single_gpu/resume_check.py \
  --output /home/ubuntu/ullm-pytorch/runs/single-gpu-b200-2026-11-08/resume-check
```

For new independent trials, pass `--root` to the suite. Each worker can also be
run directly with `--output`, `--steps`, and its experiment switches; use
`worker.py --help` for the complete argument list.

## Measurement boundaries

The baseline uses the same default model and seeded batch stream as the original
trainer. `LabLM` adds optional activation checkpointing without changing the
parameter layout. The worker releases an unused logits reference promptly;
the harness also adds explicit timing instrumentation. This establishes a new
controlled baseline and is not a bit-for-bit copy of the old timing harness.

Twenty real training updates warm up the GPU by default; their learning counts,
but their time/tokens are excluded from steady-state throughput. Training wall
time includes batch loading, transfers, forward/backward, and updates. CUDA is
synchronized at warmup and evaluation boundaries, not after every measured step.
Evaluation, generation, and final checkpoint write time are measured separately.
Only the first training step is synchronized for cold-start latency; it normally
falls in the excluded warmup.

CUDA-event step durations start after CPU batch loading and include stream
execution plus host launch gaps between events. They do not measure input wait.
Phase event sums need not equal wall time. Whole-job time includes initialization
and warmup but excludes final parameter hashing, result serialization, and
snapshot export. Checkpoint writes are buffered filesystem writes, not an fsync
durability guarantee. Baseline/BF16 and batch sweeps use three unprofiled repeats;
failure drills and mitigation demonstrations are individual runs.

The batch sweep uses batches 32–1,024 and 32,768 sequence examples
(4,194,304 token positions) per
run. Changing batch size changes update count and optimization noise even at a
matched token budget; the cosine schedule spans each run's full update count.
Validation always uses the same full batches of 32, independently of training
microbatch size. BF16 uses autocast with FP32 parameter/optimizer storage.

Profiles are separate diagnostic jobs, capturing updates 23–27 after warmup.
The synthetic input stall sleeps for 30 ms per collated batch. Worker/prefetch
tuning is evaluated both on the real in-memory workload and on the injected
stall; improvement for one should not be generalized to the other.

## Failure drills and limitations

OOM probes increase batch size with an explicit allocator limit of 85% of the
GPU's visible capacity. This is a real PyTorch allocator OOM on B200, but the
observed boundary is **not** the unrestricted physical capacity boundary. Each
probe gets a fresh process. The next job tests recovery using accumulation with
the same failed effective batch size. Snapshot history is capped at 2,000 entries.
Internal allocator-history APIs are version-dependent and used only for this lab.

A separate batch-2,048 probe removes that process budget and verifies physical
allocation failure, followed by a fresh-process health check. These are isolated
application OOMs; the experiments do not reset GPUs or modify other jobs.

The graph-retention drill saves graph-bearing losses and uses `retain_graph=True`
for 12 updates; the healthy control does not retain graphs. It uses batch 8 and
a 25% process memory ceiling. Non-finite drills inject NaN at update 5 and check
before an invalid optimizer update. These controlled failures are not claims
that the original trainer naturally suffers those faults.

Accumulation and block checkpointing keep effective batch 32, initial weights,
data membership, and update count fixed. Their gradients are verified within
tolerance, not promised bit-exact across different arithmetic paths. Loss is
normalized by accumulation count; clipping and optimizer/scheduler advance once
per effective batch. This lab implements single-GPU accumulation only.

Compilation uses default Inductor, a dedicated cache, four compile threads, and
`fullgraph=True`; a successful capture cannot silently accept graph breaks in
that forward. First-step latency is reported separately from warmed throughput.
Compilation and cache outcomes depend on the installed software and machine.
Two additional compiled jobs reuse the populated cache so cached startup and
warmed-throughput variation can be distinguished from the initial cold run.

## Plotting and evidence

`run_all.sh` finishes with the numerical/outcome audit and compact analysis.
Historical before-fix and cleanup-debug outcomes are retained in the original
results directory but are not required in a fresh independent run.

`plot.py` requires matplotlib and does not require PyTorch. The delivered figures
were rendered locally in a temporary plotting environment; no plotting packages
were added to the node's pinned training environment. To render from extracted
evidence in a plotting environment with matplotlib installed:

```bash
python experiments/single_gpu/plot.py findings/single-gpu-b200-2026-11-08
```

Large tensor files/caches remain on the node. Snapshots and traces are also
available locally; compact JSON, figures, and narrative RCAs live in `findings/`.
