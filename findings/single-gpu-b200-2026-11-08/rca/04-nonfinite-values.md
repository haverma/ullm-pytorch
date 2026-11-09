# Detecting non-finite values before an optimizer update

## Drills

Two isolated FP32 jobs inject a NaN at update 5:

| Injection | Detection point | Recorded evidence |
|---|---|---|
| Multiply loss by NaN | Before backward | `step=5`, `phase=loss_before_backward` |
| Set one gradient entry to NaN | Before clipping/update | `step=5`, `phase=gradient_before_clipping`, `parameter=tok_emb.weight` |

Both jobs stop with `status=nonfinite_detected`. They do not apply the invalid
optimizer update. A separate healthy control completes 30 updates and validation
without injecting either fault. These outcomes are deliberate detections, not
unexpected test failures.

## Why the location matters

Checking only a later validation loss or saving the corrupted checkpoint would
lose the first causal signal. A loss check distinguishes a bad forward/loss
path from a gradient that becomes invalid later. Checking named gradients before
clipping/update identifies the affected parameter and avoids propagating NaN
through optimizer state and model weights.

The checks synchronize GPU work and scan tensors, so the lab enables them for
these diagnostic jobs rather than silently adding them to every benchmark.
This does not identify a naturally occurring numerical-stability bug in the
original model or establish all possible NaN causes. Real incidents would also
require input, precision, operation, and gradient-history evidence.

Raw artifacts are `nan-loss/result.json`, `nan-gradient/result.json`, and
`finite-control/result.json` under `runs/single-gpu-b200-2026-11-08/`.
