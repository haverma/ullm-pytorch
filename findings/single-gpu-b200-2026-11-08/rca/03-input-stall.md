# Diagnosing an input-pipeline stall

## Trigger

The collate function sleeps for 30 ms per training batch. This is a controlled
simulation of slow input preparation, not a claim that Tiny Shakespeare has
that delay naturally. The same batch size, model, seed, and 100-update workload
are used for each loader configuration; 20 updates warm up before timing.

## Evidence

Three-run median training throughput:

| Configuration | Tokens/sec |
|---|---:|
| Normal input, zero workers | 297,691 |
| Normal input, two workers | 293,339 |
| Normal input, two workers + pinned/nonblocking transfers | 342,341 |
| Normal input, same + prefetch factor 4 | 344,015 |
| Injected delay, zero workers | 97,121 |
| Injected delay, two workers + pinned/nonblocking transfers | 264,166 |

Moving batch preparation into worker processes and prefetching lets CPU work
overlap GPU execution. The stalled case improves approximately 2.72x, but remains
below the corresponding unstalled pinned configuration. Two workers are not
a general cure: workers alone did not improve this normal in-memory workload.
Pinned/nonblocking transfers materially helped this particular setup; the
prefetch-factor increase added comparatively little.

## Trace interpretation

Compare `baseline-profile`, `stall-profile`, and `stall-fixed-profile` under
`runs/single-gpu-b200-2026-11-08/`. The named `train/load_batch` region localizes
the injected CPU wait. Check it against the CUDA timeline rather than calling
every idle gap a slow GPU kernel. CUDA-event step timing starts after batch
loading, whereas training wall time includes that wait; the discrepancy is
expected and is itself useful evidence.

The profiler adds overhead, so the throughput table uses unprofiled repeats.
Its five captured updates are illustrative of this injected bottleneck, not a
complete production I/O characterization.
