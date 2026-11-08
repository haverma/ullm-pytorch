# PyTorch training and GPU execution

## What this model learns

TinyLM trains from randomly initialized weights on Tiny Shakespeare using GPT-2
tokenization. It uses the tokenizer, not pretrained GPT-2 weights. Next-token
training supplies its own labels: an input window `[A, B, C, D]` has targets
`[B, C, D, E]`. The corpus is split into training and validation data so progress
can be checked on examples not used for gradient updates.

The output of training is learned parameter tensors plus metrics and generated
samples. `model.py` defines the architecture; checkpoints store its learned
numbers and training state. A checkpoint is not an inference server.

## One training step

1. The CPU DataLoader creates integer token-ID tensors of shape `[B, T]`.
2. Inputs and targets move to the selected device. Model parameters are already
   there; CPU input loading and file logging do not automatically move to GPU.
3. The model returns logits of shape `[B, T, V]` and cross-entropy loss.
4. `loss.backward()` computes gradients. This does not update model weights.
5. Gradient clipping limits the gradient norm. `optimizer.step()` changes the
   learned parameters using gradients and optimizer state.
6. The scheduler advances the learning rate. Gradients are cleared before the
   next update because PyTorch otherwise accumulates them.

`B` is batch size, `T` sequence length, `V` vocabulary size (50,257 here), and
`C` hidden dimension (192 by default). Longer sequences also change this
project's positional-embedding dimensions, so a sequence-length change is not
just a data-pipeline setting.

## Attention: exchange information between token positions

An attention block receives hidden vectors `[B, T, C]`, not raw token IDs.
Its learned `c_attn` projection produces queries, keys, and values by mapping
`C → 3C`. Splitting gives three `[B, T, C]` tensors.

Queries express what a position is looking for; keys support matching; values
provide the information to combine. Each is reshaped and transposed into
`[B, H, T, C/H]`. With C=192 and six heads, each head uses 32 dimensions.
Heads are parts of the computation, not separate GPUs.

Conceptually, a head computes `softmax(QKᵀ / sqrt(head_dim) + mask) V`.
`is_causal=True` means position t can attend only to positions ≤t, preventing
future-token leakage. Training can still compute all positions in parallel.

After attention, transposing and joining the heads returns `[B, T, C]`.
`contiguous()` provides the layout required by the subsequent `view`. The output
projection mixes the heads' results. Batch and head dimensions remain distinct;
the earlier comment about moving the head into the batch dimension is imprecise.

## MLP: transform each token's features

The block's MLP is a multilayer perceptron with two linear layers and GELU:

```text
[B, T, 192] → Linear(192, 768) → GELU → Linear(768, 192) → [B, T, 192]
```

A linear layer computes `x @ weight.T + bias`. Its weights and biases are
learned. GELU provides nonlinearity: without it, the two affine transformations
could be represented by one affine transformation.

The MLP applies the same weights independently to each token position. It does
not exchange information between tokens; attention does that. The block uses
pre-normalization and residual additions:

```python
x = x + self.attn(self.ln_1(x))
x = x + self.mlp(self.ln_2(x))
```

The residual addition retains the prior representation while adding the learned
transformation. LayerNorm normalizes features within each token vector.

## Weights, seeds, and checkpoints

Weights are tensors visible through `model.named_parameters()`. The project
initializes linear/embedding weights from a normal distribution with standard
deviation 0.02; linear biases start at zero, and LayerNorm uses its default
unit scale and zero bias.

The first weight log samples five values from each of three tensors. Later
records show changes relative to those initial values. Matching 15 samples is
useful but does not prove every parameter matches; a full-state checksum is a
stronger check. Logging millions of values would obscure the signal and slow
the job. Even small GPU-to-CPU reads can synchronize execution.

Keep seed, architecture, initialization code, and software environment fixed
when comparing trials. Identical initialization is different from identical
training: batch membership, reduction order, and nondeterministic operations
can affect results. Seeds alone do not guarantee bit-exact GPU replay.

A `.pt` checkpoint is a binary file created with `torch.save`. The existing
trainer stores model, optimizer, scheduler, completed steps, configuration, and
RNG states. AdamW's moment estimates explain why a training checkpoint and its
memory requirements exceed the parameter size alone. Resuming also restores
training progress; it is not equivalent to starting a fresh trial from the
same initial weights. Changing `--max-steps` on resume does not reconstruct the
saved scheduler's original plan.

## FP32, BF16, and memory

FP32 uses four bytes per scalar; BF16 uses two and retains roughly FP32's
exponent range with fewer significant digits. BF16 mixed precision can run
suitable operations more efficiently on supported GPUs while retaining higher
precision where needed. Parameters and optimizer state often remain FP32, so
mixed precision does not automatically halve total training memory. The
reported DDP experiment used FP32 throughout; BF16 support was smoke-tested,
not benchmarked as part of that comparison.

Training memory includes parameters, gradients, optimizer state, activations,
temporary workspaces, and runtime/communication allocations. PyTorch's
allocated memory tracks live tensor allocations; reserved memory includes
allocator-managed capacity that may be cached for reuse. External allocations
may not appear in PyTorch memory snapshots. Releasing unused cached memory
does not free tensors that remain live.

Increasing batch size or sequence length can trigger OOM. Gradient accumulation
reduces the microbatch size while retaining an effective batch size, but needs
appropriate loss scaling and update scheduling. Activation checkpointing saves
activation memory by recomputing selected forward work during backward. This
is different from saving a training checkpoint to disk.

## Profiling and debugging

CUDA execution is asynchronous with respect to the CPU. Timing only a Python
call may measure enqueue time rather than completed GPU work. CUDA events or
properly placed synchronization are needed for GPU-aware timing. Synchronizing
every operation also changes execution behavior and should not be added
indiscriminately to a performance run.

PyTorch Profiler records CPU/CUDA events, shapes, and optional allocation data.
Named regions connect traces to loading, transfers, forward, backward, and
updates. `nvidia-smi` gives a coarse device view; it does not explain individual
operators, prove high arithmetic efficiency, or replace a timeline.

Use a warmup, capture a limited representative window, and keep unprofiled
timing runs separate. Distinguish pure compute time from time-to-solution,
including evaluation, generation, logging, and storage. Compare learning after
equal token counts when changing batch sizes. A lower loss at one sampled
step does not establish a universally optimal training duration.

Useful debugging exercises include OOM, retained autograd graphs, input-loader
stalls, non-finite loss/gradients, and recovery from an interrupted run. Start
with the first abnormal event, reproduce it, gather targeted evidence, and
verify the fix. A profiler, a memory snapshot, and a stack trace answer
different questions. The completed scaling experiment is not evidence that
all these failure drills have already been implemented.

## Code and references

- `data.py`: tokenization, shifted targets, Dataset, and DataLoader.
- `model.py`: embeddings, attention, MLP, residual blocks, and loss.
- `train.py`: explicit training loop, metrics, sampled weights, checkpointing.
- `benchmark.py`: end-to-end baseline measurement and warmed-up profiling.
- [Distributed concepts](distributed-training.md): the additional DDP layer.

Official references: [CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html),
[AMP](https://docs.pytorch.org/docs/stable/amp.html),
[reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html),
[profiler recipe](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html).
