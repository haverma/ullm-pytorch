# B200 single-GPU performance and debugging findings

The [single-GPU lab](../../experiments/single_gpu/README.md) completed on
`ubuntu@151.106.165.224`, GPU 0. The audit verified 75 results: 70 completed
normally, two intentionally OOMed, two detected injected NaNs, and one recorded
the original resume failure before its fix. GPU workloads are finished.

- [Report and coverage of all 14 steps](report.md)
- [Summary figure](summary.png), also available as [SVG](summary.svg)
- [Measurements and profiles](results.json)
- [Correctness audit](verification.json)
- [Runtime/data/source metadata](metadata.json)
- [Raw artifact hashes](artifact-manifest.json)
- [Measured source revision history](source-revisions/README.md)

Failure analyses:

1. [Allocator and physical-capacity OOM](rca/01-allocator-oom.md)
2. [GPU resume bug and fix](rca/02-gpu-resume.md)
3. [Input-pipeline stall](rca/03-input-stall.md)
4. [Non-finite loss and gradients](rca/04-nonfinite-values.md)
5. [Retained computation graphs](rca/05-retained-graphs.md)
6. [BF16 generation cold start](rca/06-bf16-generation-cold-start.md)

Raw logs, profiler traces, and allocator snapshots are synchronized locally and
on the node under `runs/single-gpu-b200-2026-11-08/`. Large checkpoint tensors and
the compiler cache remain on the node. Raw outputs are ignored by Git; this
directory keeps compact evidence, plots, and analysis beside the experiment code.
Artifact retention depends on keeping the local files/node; hashes do not
reconstruct a missing trace or checkpoint.

The earlier [single-versus-DDP comparison](../../docs/b200-scaling-results.md)
is a separate experiment with its own timing harness. Do not combine the two
sets of runtimes without accounting for their measurement boundaries.
