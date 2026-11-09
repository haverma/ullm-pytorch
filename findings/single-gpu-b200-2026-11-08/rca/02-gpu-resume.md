# GPU resume failure from RNG tensor placement

## Trigger and symptom

The original trainer runs 20 planned optimizer steps, saving a checkpoint at
step 10. A second process attempts to resume that checkpoint with the same
planned 20-step schedule and configuration. It fails before the next update:

```text
restore_checkpoint → restore_rng_state → torch.set_rng_state
TypeError: RNG state must be a torch.ByteTensor
```

## Cause

`torch.load(..., map_location=device)` remapped the saved CPU RNG byte tensor
onto CUDA along with the model state. `torch.set_rng_state()` requires its CPU
state tensor. Saving RNG state was not sufficient; loading also had to preserve
the correct tensor placement.

## Fix

`train.py:restore_checkpoint` now loads the checkpoint with `map_location="cpu"`.
The model and optimizer loaders place their state on the appropriate devices;
RNG restoration receives CPU byte tensors. The model, loss, and ordinary
fresh-training path are unchanged.

## Verification

The rerun uses deterministic algorithms, one CPU intra-op thread, a fixed seed,
and `CUBLAS_WORKSPACE_CONFIG=:4096:8`. It compares an uninterrupted 20-step run
with another process resumed from step 10:

| Check | Result |
|---|---|
| Model parameters | Bit-exact; maximum absolute difference 0 |
| Optimizer state | Bit-exact |
| Scheduler state | Equal |
| RNG state | Equal |
| Completed optimizer steps | Both 20 |

This validates the controlled single-GPU configuration. It does not promise
identical replay across platforms, nondeterministic settings, worker-prefetched
loaders, or distributed topology changes. DDP resume remains unsupported by
`train_ddp.py`.

## Artifacts

`experiments/single_gpu/resume_check.py` reproduces the experiment. Raw
`resume-before-fix/` and `resume-after-fix/` directories under
`runs/single-gpu-b200-2026-11-08/` retain result JSON, logs, and checkpoints.
Source hashes in other lab run summaries distinguish runs before and after the
resume-only source change.
