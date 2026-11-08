# Evidence: 1,300-step B200 scaling experiment

These small artifacts are versioned with the code and
[findings](../../b200-scaling-results.md). They were extracted from completed
outputs under `runs/comparison-1300/`; no training was rerun to prepare them.

| File | Contents |
|---|---|
| `comparison.json` | Three-run aggregates for 1/2/4 GPUs and seven rank-profile summaries |
| `runs.json` | All 12 full runs: configuration, timing, peaks, source hashes, quality metrics, and validation curves |
| `correctness.json` | Batch-partition and full-model gradient-equivalence check |
| `trace-manifest.json` | Paths, byte sizes, and SHA-256 hashes of the seven raw traces |

Twelve full runs means nine unprofiled timing runs plus three diagnostic jobs.
The three diagnostic jobs produced 1+2+4 rank traces. Timings from diagnostic
jobs are excluded from the medians. `runs.json` omits generated text samples;
the original metrics and samples remain in the raw outputs.

The traces are retained locally under `runs/comparison-1300/` and on the B200
node at `/home/ubuntu/ullm-pytorch/runs/comparison-1300/`. Checkpoints remain on
that node at `/home/ubuntu/ullm-pytorch/checkpoints/comparison-1300/`.
Traces, datasets, checkpoints, environments, and generated logs remain ignored
by Git. Their retention depends on the local checkout and node lifetime; the
manifest verifies available files but does not reconstruct a missing trace.

The source hashes identify the exact measured versions of the original trainer,
model, data loader, and relevant comparison entry point. The accompanying
[implementation guide](../../distributed-training.md) describes how to run new
experiments and interpret timing boundaries.
