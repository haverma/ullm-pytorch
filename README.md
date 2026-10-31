# ullm-pytorch

## Stage 1 — Build a small language-model training workload
Duration: two Saturdays. Goal: own a complete, reproducible PyTorch training job that later stages can profile, distribute, containerize, break, and serve.
Use one project only: a small decoder-only Transformer trained on a compact text corpus.

## Target outcome
A repository containing:
`tiny-lm/ train.py model.py data.py evaluate.py configs/ baseline.yaml checkpoints/ runs/ tests/ README.md`
It must train, evaluate, checkpoint, resume, and emit usable metrics.

See [the Stage 1 class and data-flow diagram](docs/stage1-architecture.md) for
how the three current Python modules interact.

## Install on Ubuntu with an NVIDIA GPU

The remote environment was prepared at `/home/ubuntu/ullm-pytorch`, with Python
3.12.3, PyTorch 2.11.0+cu128, CUDA runtime 12.8, and NVIDIA driver 580.126.20.
All eight B200 GPUs were detected. The virtual environment is `.venv/` inside
the project directory.

On another Linux x86_64 NVIDIA machine, copy the project (including
`requirements.lock.txt`), change into its directory, and run:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  -r requirements.lock.txt
python -m pip check
```

If Ubuntu reports that `venv`/`ensurepip` is missing, install the matching
`python3.12-venv` system package first. A compatible NVIDIA driver must already
be installed. The pip packages provide CUDA runtime libraries; these commands
do not install Python or a GPU driver.

`requirements.lock.txt` uses the same requirements-file syntax as
`requirements.txt`: pip reads it with `-r`; it is not an executable. It is a
`pip freeze` snapshot with exact versions, not a hash-verified lockfile. It is
specific to this Linux/Python/CUDA environment and should not be used unchanged
on macOS. To use a CPU or Mac instead, create a separate environment and install
the appropriate PyTorch build plus `tiktoken requests numpy`.

On the already-prepared node, just activate the existing environment:

```bash
ssh ubuntu@151.106.167.19
cd /home/ubuntu/ullm-pytorch
source .venv/bin/activate
```

Activation applies only to the current shell; it is not permanent or shared
between SSH sessions. You can also use `.venv/bin/python` directly without
activating the environment, as in the diagnostic command below.

## Check readiness without starting training

```bash
nvidia-smi
python -c 'import torch; print("PyTorch:", torch.__version__); print("CUDA:", torch.version.cuda); print("Available:", torch.cuda.is_available()); print("GPU count:", torch.cuda.device_count()); print("Architectures:", torch.cuda.get_arch_list())'
python train.py --help
```

For a B200, the compiled architectures should include `sm_100`. These commands
do not start training. Environment preparation verified source checksums,
dependency consistency, tokenization, initial weight sampling, and CPU profiler
trace export. It did not execute a model forward pass, backward pass, optimizer
step, or CUDA kernel execution test. GPU workload execution is a separate step.
The prepared node also has `preflight/report.json` with the check results.

## Training data

`train.py` automatically downloads [Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)
on its first run, tokenizes it with the GPT-2 tokenizer, and caches a deterministic
90% training / 10% validation split in `data/train.bin` and `data/val.bin`.
Targets are the input tokens shifted by one position; no manual labels are needed.
Prepare data without training by running `python data.py`.

## Start a single-GPU training run

These commands start training. On the prepared remote node, wait for explicit
confirmation before executing them.

For a first 100-step diagnostic run with profiling, connect and use the
environment's Python directly (no activation required):

```bash
ssh ubuntu@151.106.167.19
cd /home/ubuntu/ullm-pytorch
.venv/bin/python train.py \
  --device cuda:0 --seed 42 \
  --batch-size 32 --seq-length 128 --max-steps 100 \
  --profile \
  --run-dir runs/diagnostic --checkpoint-dir checkpoints/diagnostic
```

For a 1,000-step baseline without profiling overhead:

```bash
python train.py \
  --device cuda:0 --seed 42 \
  --batch-size 32 --seq-length 128 --max-steps 1000 \
  --no-profile \
  --run-dir runs/baseline --checkpoint-dir checkpoints/baseline
```

The current trainer uses FP32 parameters and eager execution. BF16 mixed
precision and `torch.compile` experiment switches are not implemented yet.

`--device` selects one device:

| Value | Execution hardware |
| --- | --- |
| `cuda` or `cuda:0` | First visible NVIDIA GPU |
| `cuda:1` | Second visible NVIDIA GPU |
| `cpu` | CPU |
| `mps` | Apple Silicon GPU |
| `auto` (default) | CUDA if available, otherwise MPS, otherwise CPU |

An eight-B200 node has eight GPUs, but this trainer uses only the selected one.
It does not implement DDP; launching eight processes with `torchrun` will not
correctly distribute this implementation. Data loading and file logging use
the CPU even when model computation uses CUDA.

## Outputs and checkpoints

Paths below are relative to the run/checkpoint directories you select:

| Output | Contents |
| --- | --- |
| `metrics.jsonl` | Step, loss, learning rate, gradient norm, cumulative tokens/sec, elapsed time; validation loss and generated text at evaluation points |
| `weights.jsonl` | Initial and periodic samples of weight values and their changes |
| `profiler/trace-*.json` | CPU/CUDA execution timeline when profiling is enabled |
| `profiler/trace-*.txt` | Top-operation timing summary |
| `step-000500.pt`, etc. | Binary PyTorch checkpoints with all model weights, optimizer/scheduler state, completed steps, configuration, and RNG states |

The model architecture is defined in `model.py`; checkpoints store its learned
values and training state. Training does not produce a standalone inference
server. Metrics and weight logs append to existing files; use a fresh directory
for each independent trial so their results do not mix. Checkpoints with the
same step and directory can be overwritten.

To continue an interrupted run, use the same model settings, batch/data settings,
seed, and originally planned `--max-steps`:

```bash
python train.py \
  --device cuda:0 --seed 42 \
  --batch-size 32 --seq-length 128 --max-steps 1000 \
  --resume checkpoints/baseline/step-000500.pt \
  --run-dir runs/baseline --checkpoint-dir checkpoints/baseline
```

`--max-steps` is the total target step, not additional steps. Resume restores the
optimizer and scheduler too; changing `--max-steps` does not rebuild the saved
scheduler's original schedule. Use the original configuration when continuing.

## Keep initial weights consistent between trials

Use the same `--seed` (default `42`), model configuration, source code, and
software environment. The trainer seeds Python, NumPy, and PyTorch before
creating the model. A fresh run initializes weights; `--resume` loads existing
weights **and** training progress, so it is not a fresh-trial initialization flag.

For example, compare batch sizes using separate output directories:

```bash
python train.py --device cuda:0 --seed 42 --batch-size 32 --max-steps 100 \
  --no-profile --run-dir runs/batch32 --checkpoint-dir checkpoints/batch32
python train.py --device cuda:0 --seed 42 --batch-size 64 --max-steps 100 \
  --no-profile --run-dir runs/batch64 --checkpoint-dir checkpoints/batch64
```

Both runs should start with the same weights in the same environment. Batch size
does not change the model architecture. Keep `--seq-length`, `--embed-dim`,
`--n-heads`, and `--n-layers` fixed for this comparison. Changing sequence length
changes positional-embedding dimensions and can also shift later initialization
draws, so the complete initial state will no longer match.

To compare the logged initial samples:

```bash
python - <<'PY'
import json

with open("runs/batch32/weights.jsonl") as f:
    first = json.loads(next(f))
with open("runs/batch64/weights.jsonl") as f:
    second = json.loads(next(f))
assert first["event"] == second["event"] == "initial"
assert first["weights"] == second["weights"], "Initial weight samples differ"
print("All 15 sampled initial values match.")
PY
```

Matching samples do not prove every weight is identical. A full-state checksum
or dedicated initialization checkpoint would provide that stronger check;
neither is currently emitted at initialization. Step 0 is logged only as a
sample, not saved as a full checkpoint.

Identical initialization does not guarantee identical training results. GPU
operations can be nondeterministic, and changing batch size changes the
optimization updates. The trainer does not enforce deterministic GPU algorithms.
For learning-quality comparisons, compare after equal numbers of training tokens,
not just equal steps: batch 64 processes twice as many tokens per step as batch
32 at the same sequence length. Keep `--num-workers 0` for the current
deterministic data-order resume implementation.

## Lightweight weight logs and optional profiling

By default, metrics are written on the first update, every 100 steps, validation
steps, and the final step. Set `--log-interval N` to change the metric interval.
Files stay open during training rather than being reopened for every update.

`runs/weights.jsonl` records five fixed values from each of three tensors: the
token embedding, first attention QKV projection, and first MLP expansion. Each
record includes the tensor name, shape, dtype, flattened indices, values, and
their changes from the starting sample. It records initialization (step 0), the
first update, every 100 steps, and the final update. These are representative
samples, not a dump of all weights or gradients. Full weights remain in checkpoints.
On resume, the starting record is labeled `resumed`, and changes are relative to
the loaded weights. Existing logs are appended to.

```bash
python train.py --device cuda --weight-log-interval 100 --log-interval 100
# Disable weight sampling entirely:
python train.py --device cuda --weight-log-interval 0
```

PyTorch Profiler is **off by default** (`--no-profile` also explicitly disables
it). Enable it for a diagnostic run:

```bash
python train.py --device cuda --profile --max-steps 100
```

The profiler skips one training step, warms up for one, records three, and then
stops recording. Use at least five remaining steps for a complete capture.
It captures CPU operations and, on CUDA, GPU operations, with named regions for
batch loading, transfer, forward, backward, and optimizer work. MPS runs capture
CPU operations only. Profiling adds overhead during capture; use an unprofiled
run for throughput comparisons. Custom frequent logging/evaluation can appear
in the trace too.

Outputs under `runs/profiler/` have unique filenames:

- `trace-*.txt`: a summary of the top 20 operations by self CPU/CUDA time.
- `trace-*.json`: a Chrome trace, which can be opened in Perfetto's trace viewer
  or `chrome://tracing` for a timeline.

Weight sampling copies only 15 numbers to the CPU at each sampling point; no
whole-model copies or per-step weight logging are performed. Sampling and sparse
metric collection can still synchronize the GPU. The existing tokens/sec metric
is cumulative wall-clock throughput, including previous evaluation, checkpoint,
logging, and profiling overhead; it is not an isolated GPU step benchmark.

## Saturday 1 — Make training correct
### 1. Set the fixed workload
- Dataset: Tiny Shakespeare or a small fixed slice of TinyStories
- Tokenizer: GPT-2 BPE tokenizer via tiktoken or Hugging Face tokenizers
- Model: small decoder-only Transformer, roughly 5–20M parameters
- Device: CPU or local GPU

### 2. Build the data pipeline
Implement data.py:
- Download/load a version-pinned corpus, split deterministically, tokenize once and cache.
- Create fixed-length token windows: input and target.
- Use a PyTorch DataLoader. Make batch size, sequence length, worker count, and seed configurable.

### 3. Implement the model
Implement model.py yourself using raw PyTorch:
- Token embeddings, Positional embeddings, Causal self-attention mask, Transformer blocks, Layer norm, Output LM head.

### 4. Write the manual training loop
Implement train.py with explicit steps. Add AdamW, LR schedule, gradient clipping, periodic validation, progress logging, periodic text generation. Track step, train_loss, validation_loss, learning_rate, tokens_per_second, elapsed_seconds.

### 5. Establish a baseline run
Run a fixed baseline config. Write runs/baseline.md with hardware/software versions, metrics, generated text, etc.

## Saturday 2 — Make training resumable and operationally usable
### 6. Add complete checkpointing
Include model weights, optimizer/scheduler state, current step/epoch, RNG states, config.

### 7. Implement resume
Must restore state, continue from saved step, continue LR schedule, write new metrics without overwriting history.

### 8. Add configuration discipline
Use a YAML or JSON config.

### 9. Add basic tests
For batch shifts, model output shape, causal mask, checkpoint round-trips.

### 10. Finish the README
Make the project reproducible.

## Definition of done
1. A hand-written PyTorch language model trains.
2. Its loss decreases on a known dataset.
3. A run can be interrupted and resumed correctly.
4. Every important choice is configurable.
5. You have one baseline run.
