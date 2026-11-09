# Controlled allocator OOM during loss backward

## Trigger and symptom

An isolated FP32 TinyLM process increases effective batch size at sequence length
128, with a PyTorch allocator budget of 85% of visible GPU capacity. Batches
128, 512, and 1,024 complete the two-update probe. Batch 2,048 raises
`torch.OutOfMemoryError` in backward on the first update.

This tests a process allocation ceiling on B200, not the unrestricted physical
capacity boundary. The exception explicitly reports 178.35 GiB total capacity,
60.91 GiB free, and a 151.60 GiB allowed process budget.

## Evidence and diagnosis

At failure, PyTorch had 116.68 GiB allocated and requested another 49.08 GiB.
Reserved-but-unallocated memory was only 21.90 MiB, so fragmentation is not
supported as the main cause by this evidence.

The Python traceback reaches `worker.py`'s backward phase. The allocator history
adds the C++ context: `structured_nll_loss_backward` / `nll_loss_backward` /
`NllLossBackward0`. The failed allocation was 52,698,284,032 bytes. Its scale
matches a dense gradient over token positions and the full vocabulary:

```text
2,048 sequences × 128 positions × 50,257 vocabulary entries × 4 bytes
    = 52,698,284,032 bytes ≈ 49.08 GiB
```

The existing live allocations plus this request exceed the configured budget.
The snapshot's largest existing segment was approximately another tensor of
that size. This is memory pressure from the dense loss/output path; the evidence
does not establish a GPU hardware fault or a fragmented allocator.

## Mitigation and verification

A fresh process uses the same effective batch of 2,048 but microbatch 32 and
64 accumulation passes. Losses are normalized by 64; gradients are cleared and
the optimizer/scheduler advance once per effective batch. It completes three
updates with peak allocation approximately **3.61 GiB**.

Smaller microbatches avoid simultaneously materializing the full effective
batch's dense loss tensors. This is not evidence that block activation
checkpointing alone would rescue that same loss allocation. Accumulation changes
execution time and floating-point reduction order; the short recovery probe is
a successful memory/recovery demonstration, not a full quality-equivalence run.

## Physical-capacity confirmation

A separate batch-2,048 process removes the 85% ceiling. It gets further into
backward but still OOMs: PyTorch has **165.76 GiB** allocated and requests another
**49.08 GiB** with only **11.83 GiB** physically free. The allocator history now
identifies `structured_log_softmax_backward` / `_log_softmax_backward_data`.
Thus the budgeted and physical failures occur at different allocations in the
same dense cross-entropy backward path. Both are preserved separately.

Batch 1,024 succeeds, and batch 2,048 fails; this is a coarse tested bracket,
not an exact maximum batch size. A fresh 30-step training/validation process
after the physical OOM completes successfully. No GPU reset was needed.

## Artifacts

Raw directory: `runs/single-gpu-b200-2026-11-08/`.

- `oom-batch-2048/result.json`: configuration, traceback, allocator counters.
- `oom-batch-2048/oom.pickle`: trusted allocator history/snapshot.
- `oom-batch-2048/oom-allocation.json`: extracted allocation event and C++ frames.
- `oom-boundary.json`: observed successful and failing candidates.
- `oom-recovery-batch-2048-micro32/result.json`: successful recovery evidence.
- `physical-oom-batch-2048/`: unrestricted-capacity failure and allocator history.
- `post-physical-oom-health/result.json`: successful post-failure workload.

The snapshot is for local inspection; no upload to an external memory viewer
was performed. Private allocator APIs make the snapshot tooling version-dependent.
