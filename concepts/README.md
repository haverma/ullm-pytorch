# Concepts and theory

These notes collect the explanations from the project discussions. They describe
the mechanisms separately from the measured B200 experiment.

1. [PyTorch training and GPU execution](pytorch-training-and-gpu-execution.md):
   tensors, attention, MLPs, weight updates, precision, checkpoints, and profiling.
2. [Distributed training](distributed-training.md): ranks, DDP, NCCL, batch
   partitioning, microbatches, gradient accumulation, microsteps versus optimizer
   steps, scaling, synchronization, and failure diagnosis.

For executable commands and current implementation details, use
[the DDP implementation guide](../docs/distributed-training.md). For measured
results, use [the B200 comparison](../docs/b200-scaling-results.md) and its
[committed evidence](../docs/experiments/b200-1300/README.md).

Discussion sources: this project's “Assess PyTorch skills for DGX Cloud” chat,
“Plan learning after Stage 1,” and “Explain ullm-pytorch.” These are edited
explanations, not verbatim transcripts. The distributed notes also incorporate
the user-supplied side-chat excerpt on microbatches, accumulation, synchronization,
and iteration counting, including its corrected 32-versus-128 batch example.
Interviewer-specific speculation and
unrelated personal details are intentionally omitted.
