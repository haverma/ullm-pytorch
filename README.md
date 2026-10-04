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

## Run the current implementation

Install PyTorch plus the small data-pipeline dependencies, then start a short
CPU smoke run:

```bash
python -m pip install torch tiktoken requests numpy
python train.py --device cpu --max-steps 10 --eval-interval 5 --checkpoint-interval 10
```

`train.py` downloads and tokenizes Tiny Shakespeare on its first run, writes
step metrics to `runs/metrics.jsonl`, and saves complete checkpoints under
`checkpoints/`. Resume a run with:

```bash
python train.py --resume checkpoints/step-000010.pt --max-steps 100
```

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
