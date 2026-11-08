# B200 single-GPU versus DDP: 1,300-step comparison

## Result

Two B200s provided the lowest median time for this workload: **17.40 seconds**,
versus **19.23 seconds** on one B200 and **18.66 seconds** on four. This is a
**1.11x throughput increase** over one GPU, not a 2x increase. Four GPUs did not
improve on two. Final validation losses differed by less than 0.00005.

Measurements were collected on `ubuntu@151.106.165.224`, hostname `2507XLV0CU`,
with NVIDIA driver 580.126.20, PyTorch 2.11.0+cu128, CUDA runtime 12.8, and
NCCL 2.28.9. GPU topology reports NV18 connections between the GPUs used.
These are single-node NVLink experiments; they do not test cross-node RDMA.

## Controlled workload

- Original TinyLM: embedding dimension 192, six heads, six layers; FP32 eager.
- Seed 42, Tiny Shakespeare/GPT-2 tokenization, sequence length 128.
- Global batch **32** in every configuration: per-rank batches 32, 16, and 8.
- **1,300 optimizer steps / 5,324,800 training token positions** in every run.
- Same learning rate, cosine schedule, initialization, and global batch stream.
- Same validation batches every 100 steps, 80-token text generation, weight
  logging, and checkpoints at steps 500, 1,000, and 1,300.
- Three sequential unprofiled runs per GPU count. Separate full-length
  diagnostic runs capture only five warmed-up steps, updates 23–27.
- One CPU intra-op thread per worker in all measured configurations.

## Timing and quality

| GPUs | Batch per GPU | Median trainer runtime | Runtime range | Tokens/sec | Speedup | Final validation loss |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 32 | 19.23 s | 18.80–19.30 s | 276,959 | 1.00x | 3.609478 |
| 2 | 16 | 17.40 s | 17.25–17.61 s | 306,094 | 1.11x | 3.609433 |
| 4 | 8 | 18.66 s | 18.03–18.81 s | 285,284 | 1.03x | 3.609430 |

Trainer runtime includes setup, distributed initialization where applicable,
validation, generation, logging, and checkpoint writes. Interpreter/import and
torchrun parent startup are excluded. Timings exclude the final DDP replica
checksum audit. Profiled run times are not included in this table.

The original loop's elapsed metric at step 1,300, before the final validation
and checkpoint, had medians of **17.87 / 13.79 / 13.13 seconds** for 1/2/4 GPUs.
That suggests more improvement inside the loop than the whole-job result, but
still includes earlier validation, generation, logging, and checkpoints: it is
not an isolated training-step benchmark.

At the same per-GPU pricing, GPU-seconds were approximately **19.23 / 34.79 /
74.66** for 1/2/4 GPUs. Two GPUs minimize elapsed time among the tested choices;
one minimizes GPU resource use. No provider pricing was assumed.

In the first single-GPU run, the lowest sampled validation loss was **3.609129
at step 1,200**, followed by **3.609478 at step 1,300**. The experiment retained
1,300 steps as requested. This does not establish a universal optimal stopping
point, particularly because the schedule depends on the configured total steps.

## Memory

| GPUs | Maximum per-rank peak allocated | Maximum per-rank peak reserved |
|---:|---:|---:|
| 1 | 3.51 GiB | 4.26 GiB |
| 2 | 2.94 GiB | 4.52 GiB |
| 4 | 2.74 GiB | 3.48 GiB |

These are PyTorch allocator peaks, not total physical GPU consumption or
aggregate memory across all ranks. Rank 0 evaluates the full validation batch,
so its memory peak is higher than the training-only workers. For example, the
first two-GPU run's rank-1 peak allocation was approximately 1.88 GiB. DDP
replicates parameters, gradients, and optimizer state; only the batch/activation
work is split. NCCL memory outside the allocator may be absent from these peaks.

## What the profiles show

| Configuration/rank | GPU kernel time per captured step | NCCL kernel time per captured step |
|---|---:|---:|
| Single GPU | 11.290 ms | 0 |
| Two GPUs, rank 0 | 6.517 ms | 0.126 ms |
| Two GPUs, rank 1 | 8.499 ms | 2.053 ms |
| Four GPUs, rank 0 | 11.313 ms | 6.975 ms |
| Four GPUs, rank 1 | 4.426 ms | 0.184 ms |
| Four GPUs, rank 2 | 10.086 ms | 5.802 ms |
| Four GPUs, rank 3 | 9.596 ms | 5.312 ms |

The single-GPU trace is dominated by matrix multiplication, with additional
softmax and attention kernels. DDP traces contain explicit
`ncclDevKernel_AllReduce_Sum_f32_RING_LL` kernels, confirming gradient
communication on the GPUs.

Excluding NCCL events, kernel time falls from approximately **11.29 ms** on
one GPU to **6.4 ms per rank** on two and **4.3 ms per rank** on four. The
distributed work split is doing useful compute reduction. However, NCCL event
durations vary strongly between ranks and grow on several ranks in the
four-GPU capture. They can include waiting for peers, not just data movement.

Interpretation: this small FP32 model/global batch offers limited work per GPU.
Communication, rank coordination, initialization, and serial validation and
generation constrain the time-to-solution gain. Four GPUs are an inefficient
choice for this exact experiment. The profile does **not** establish a network
fault or measure pure NVLink transfer latency.

Kernel durations above are sums over a five-step diagnostic capture. Kernels
can overlap with each other, especially NCCL and backward compute; these sums
are not additive wall-clock step times. Profiling adds overhead, and this
short early-training capture does not characterize every stage of the run.

## Correctness evidence

- Eight actual input/target batches matched the original trainer's batches,
  partitioned into the expected rank slices.
- Full-model gradient comparison against a single-GPU full-batch computation
  passed; maximum absolute gradient error was **6.7055e-7**. Both losses were
  **10.8803920746** for that check batch.
- Every completed DDP run verified identical final parameter checksums across
  ranks. Each GPU-count configuration reproduced its final validation loss
  across the three unprofiled runs in this environment.
- FP32 reduction order differs across GPU counts, so trajectories are close
  rather than bit-exact. This experiment validates the current model/workload;
  it is not a general recovery or numerical-stability test.

## Reproduce and inspect

Compact [experiment evidence](experiments/b200-1300/README.md) is committed
alongside this report, including per-run summaries, validation curves, profile
aggregates, correctness results, and raw-trace hashes. General explanations live
separately in [concepts/](../concepts/README.md).

See [distributed-training.md](distributed-training.md) for the implementation,
commands, timing definitions, and limitations. The original `train.py`,
`model.py`, and `data.py` were unchanged. `train_ddp.py` is a separate entry
point; distributed resume is explicitly unsupported in this version.

Raw outputs are retained locally and on the node under `runs/comparison-1300/`:

- `comparison.json`: all timing aggregates and per-rank profile summaries.
- `single*/`, `ddp2*/`, `ddp4*/`: metrics, sampled weights, and run summaries.
- `single-profile/profiler/trace.json`: single-GPU trace.
- `ddp2-profile/profiler/rank-*/trace.json`: two-GPU traces.
- `ddp4-profile/profiler/rank-*/trace.json`: four-GPU traces.
- `correctness.log`, `topology.txt`, and `requirements.txt`: validation/environment evidence.

Checkpoints remain on the node under `checkpoints/comparison-1300/`. Large run
outputs and checkpoints remain ignored by Git. Source hashes are recorded in
each run summary.

Next performance experiments should establish a new controlled baseline before
trying BF16, larger global batches, accumulation, or compilation. Those changes
may change both hardware efficiency and training behavior; their benefits should
not be credited to DDP in this fixed-workload comparison.
