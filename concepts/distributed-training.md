# Distributed training: from a single process to DDP

## What is being distributed?

The original job has one model, one optimizer, and a batch on one GPU. Its step
is data loading → transfer → forward → loss → backward → optimizer update.
Selecting another device moves this job; it does not split the work.

With data parallelism, multiple workers each hold a complete model and optimizer
and process different examples. Their gradients are synchronized so each replica
applies the same update. The parameters remain replicated; DDP does not combine
GPU memory into one large pool or shard the optimizer state.

This differs from tensor parallelism (splitting individual operations/weight
matrices), pipeline parallelism (splitting model layers into stages), and FSDP
(sharding model state with communication to assemble what computation needs).
This project implements DDP only.

## Process, rank, device, and launcher

A typical GPU DDP deployment uses one process per GPU. Each process runs the
Python training program independently.

| Term | Meaning | Two-GPU single-node example |
|---|---|---|
| World size | Number of workers in the process group | 2 |
| Rank | Worker identity across the group | 0 or 1 |
| Local rank | Worker identity within its node | 0 or 1 |
| Device | GPU used by this process | `cuda:0` or `cuda:1` |
| Process group | Workers participating in communication | Both workers |

`torchrun --standalone --nproc_per_node=2 train_ddp.py` launches two workers and
sets their rank information. The script still has to select the appropriate GPU,
initialize the process group, partition data, and wrap the model in DDP. Merely
launching the original `train.py` twice would create independent training jobs,
with possible GPU contention and conflicting outputs.

On multiple nodes, ranks remain globally unique, while local ranks repeat on
each node. Rendezvous configuration coordinates worker discovery. Our
`--standalone` experiment is single-node and does not validate that setup.

`CUDA_VISIBLE_DEVICES` limits what the process sees and renumbers those devices:

```bash
# Physical GPU 3 is the only visible device and becomes cuda:0.
CUDA_VISIBLE_DEVICES=3 python train.py --device cuda:0

# Physical GPUs 2 and 5 become local devices cuda:0 and cuda:1.
CUDA_VISIBLE_DEVICES=2,5 torchrun --standalone --nproc_per_node=2 train_ddp.py
```

Visibility is not reservation: another process may still access the same GPU.
A B200 is one GPU; the node used in this experiment contains eight B200s.

## What NCCL does

NCCL provides GPU communication collectives. PyTorch's distributed process
group can use it as its backend:

```python
torch.cuda.set_device(local_rank)
dist.init_process_group(backend="nccl")
model = DistributedDataParallel(model, device_ids=[local_rank])
```

The application normally calls PyTorch APIs, which use NCCL underneath. NCCL
is useful within one node as well as across nodes. Intra-node communication can
use NVLink; cross-node paths depend on the hardware/software configuration and
can involve InfiniBand or Ethernet/RDMA. An NCCL timeout alone does not prove a
network fault. The B200 experiment's traces show intra-node all-reduce kernels;
they do not test cross-node networking.

| Collective | Conceptual operation |
|---|---|
| All-reduce | Combine values from all ranks and give the result to every rank |
| All-gather | Collect every rank's contribution on every rank |
| Reduce-scatter | Combine contributions and give each rank a portion of the result |
| Broadcast | Send a source rank's values to the other ranks |
| Barrier | Wait until every participating rank has reached that point |

DDP uses synchronization to align model replicas initially and gradient
reduction during training. This is not a consensus protocol: there is no
replicated log or automatic continuation after arbitrary worker loss.

## What happens during backward?

Each rank computes a local forward pass and loss, then calls `loss.backward()`.
Autograd computes local gradients. DDP groups gradients into buckets and launches
reductions as those buckets become ready, allowing communication to overlap
with remaining backward computation. It does not necessarily wait for the whole
backward pass and issue one enormous transfer.

For equally sized local batches and mean-reduced losses, the desired gradient is:

```text
global_gradient = (rank_0_gradient + ... + rank_N-1_gradient) / N
```

The default DDP reduction yields the average gradient. All workers then clip
those synchronized gradients and call their optimizers. Matching initial state,
gradients, and optimizer behavior keeps the replicas aligned. In this experiment
we verified final parameter checksums across ranks.

Unequal local batch sizes or different loss reductions require more careful
weighting; simply averaging each rank's mean loss/gradient may be incorrect.

## Global batch and fair comparisons

Without accumulation:

```text
global_batch = batch_per_rank × world_size
tokens_per_update = global_batch × sequence_length
```

With equally sized accumulation microbatches:

```text
effective_global_batch = microbatch_per_rank × world_size × accumulation_steps
```

Our comparison fixes global batch 32: one GPU handles 32 examples, two handle
16 each, and four handle eight each. All configurations process 5,324,800 token
positions over 1,300 optimizer updates at sequence length 128. The learning-rate
schedule and validation workload also stay fixed.

Keeping batch 32 **per rank** would instead increase global batch to 64 or 128.
That can improve GPU utilization but changes the optimization workload. Compare
learning at matched token counts and report that change separately.

`ShardBatches` slices the legacy seeded global batches so we can check the same
examples participate in each update. A conventional `DistributedSampler` is
also useful, but changes to its shuffle/padding behavior must be accounted for;
epoch-based shuffling normally requires `set_epoch(epoch)`. DDP itself does
not partition the input data.

## Why extra GPUs may not help much

For fixed work, speedup is `single_runtime / distributed_runtime`, and scaling
efficiency is `speedup / GPU_count`. Perfect two-GPU efficiency would mean a
2x speedup. GPU-seconds (`GPU_count × elapsed_seconds`) express resource use,
not a provider-specific monetary cost.

Splitting a tiny batch reduces compute per GPU, while initialization,
communication, kernel launches, Python coordination, and serial work remain.
The slowest worker limits synchronous progress. Larger GPU counts can therefore
be slower than a smaller distributed group.

Our rank-0 validation, text generation, logging, and checkpoint writing are
serial. Other ranks wait. The measured two-GPU gain was 1.11x end to end; this is
evidence about this workload and node, not a general upper bound for DDP or B200.

## Reading distributed profiles

Capture CPU and CUDA events on every rank and label loading, transfer, forward,
backward, and optimizer work. Look at the relationship between compute and NCCL
events rather than just summing operation durations.

NCCL kernel time can contain waiting for a peer to produce its gradients.
Communication can overlap compute, so summed durations are not wall-clock step
time. A long event on one rank does not by itself identify the slow rank or prove
poor network bandwidth. Profiling adds overhead; use separate unprofiled runs
for performance claims and record the warmup/capture window.

## Failure diagnosis

| Symptom | Questions and evidence |
|---|---|
| NCCL timeout | Did another rank OOM, exit, or stop before this collective? |
| All ranks hang | Did ranks issue different collective sequences/shapes? Is one stuck in input loading? |
| Low utilization | Are ranks waiting on input, CPU launches, communication, or serial validation? |
| Poor scaling | How much useful compute remains per rank? What work is serial? |
| Divergent results | Are global batch, data order, initialization, loss weighting, and schedule comparable? |

Example: rank 1 runs out of memory and exits; rank 0 later times out while
waiting in all-reduce. The timeout is a secondary symptom. Correlate timestamps,
host, rank, GPU, step, and phase to find the first failure. Logs from only rank 0
can hide the cause. `NCCL_DEBUG=INFO` and PyTorch distributed debug logging are
diagnostic aids, not fixes.

Rank-0 checkpoint writing avoids concurrent writers, but is not a complete
distributed recovery design. A robust restart also considers per-rank random
state, data position, optimizer/scheduler state, compatible topology, and
checkpoint publication. Our DDP comparison entry point rejects `--resume` and
does not implement elastic recovery or failure injection yet.

## Code connections and references

| Concept | Project implementation/evidence |
|---|---|
| Process group, NCCL, DDP, rank barriers | `train_ddp.py:main` |
| Fixed global batches | `train_ddp.py:ShardBatches` and `sharded_loader` |
| Gradient/data equivalence | `check_ddp.py` |
| Separate profiling and timing | `benchmark.py`, `train_ddp.py:make_profiler` |
| Medians, speedup, GPU-seconds | `analyze_scaling.py` |
| Measured scaling and limitations | [B200 findings](../docs/b200-scaling-results.md) |

Official references: [DDP design](https://docs.pytorch.org/docs/stable/notes/ddp.html),
[DDP tutorial](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html),
[distributed communication](https://docs.pytorch.org/docs/stable/distributed.html).
