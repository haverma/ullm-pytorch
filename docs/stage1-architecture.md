# Stage 1 architecture

This diagram describes the current relationship between `data.py`, `model.py`,
and `train.py`. Shapes in square brackets are tensor shapes; `V` is the
tokenizer vocabulary size (`50,257` for GPT-2 BPE).

```mermaid
classDiagram
    direction LR

    class train_py {
        +parse_args() Namespace
        +main() None
        +evaluate(model, loader, device, batches) float
        +generate(model, tokenizer, prompt, tokens, device) str
        +save_checkpoint(path, model, optimizer, scheduler, completed_steps, args) None
        +restore_checkpoint(path, model, optimizer, scheduler, device) int
    }

    class data_py {
        +download_and_tokenize() None
        +get_dataloader(split, batch_size, seq_length, num_workers, seed) DataLoader
    }

    class LMDataset {
        +seq_length: int
        +data: np.memmap[uint16]
        +__len__() int
        +__getitem__(idx) tuple~Tensor, Tensor~
    }

    class DataLoader {
        +batch_size: int
        +shuffle: bool
        +generator: torch.Generator
        +__iter__() Iterator
    }

    class TinyLM {
        +vocab_size: int = 50257
        +max_seq_len: int = 128
        +tok_emb: nn.Embedding[V, C]
        +pos_emb: nn.Embedding[T, C]
        +blocks: ModuleList[Block]
        +ln_f: LayerNorm
        +lm_head: Linear[C, V]
        +forward(idx, targets) tuple~logits, loss~
    }

    class Block {
        +ln_1: LayerNorm
        +attn: CausalSelfAttention
        +ln_2: LayerNorm
        +mlp: MLP
        +forward(x) Tensor[B, T, C]
    }

    class CausalSelfAttention {
        +n_heads: int
        +c_attn: Linear[C, 3C]
        +c_proj: Linear[C, C]
        +forward(x) Tensor[B, T, C]
    }

    class MLP {
        +c_fc: Linear[C, 4C]
        +c_proj: Linear[4C, C]
        +forward(x) Tensor[B, T, C]
    }

    class AdamW {
        +step() None
        +zero_grad() None
    }

    class CosineAnnealingLR {
        +step() None
    }

    data_py ..> LMDataset : creates
    data_py ..> DataLoader : returns
    LMDataset --> DataLoader : supplies examples
    TinyLM *-- Block : stacks n_layers
    Block *-- CausalSelfAttention
    Block *-- MLP
    train_py ..> data_py : initializes data
    train_py ..> TinyLM : constructs and calls
    train_py ..> AdamW : updates parameters
    train_py ..> CosineAnnealingLR : updates learning rate
```

## One training step

```mermaid
sequenceDiagram
    participant Train as train.py / main()
    participant Loader as DataLoader
    participant Dataset as LMDataset
    participant Model as TinyLM
    participant Opt as AdamW + Scheduler
    participant Disk as metrics.jsonl + checkpoint

    Train->>Loader: next(train_iterator)
    Loader->>Dataset: __getitem__(idx), repeated B times
    Dataset-->>Loader: x, y per example [T], [T]
    Loader-->>Train: input_tokens, target_tokens [B, T] = [32, 128]
    Train->>Model: forward(inputs, targets)
    Model->>Model: tok_emb(inputs) [B, T, C]
    Model->>Model: add pos_emb [T, C]
    Model->>Model: n Transformer Blocks with causal attention
    Model->>Model: lm_head → logits [B, T, V]
    Model-->>Train: logits [32, 128, 50257], cross-entropy loss scalar
    Train->>Model: loss.backward()
    Train->>Opt: clip gradients, optimizer.step(), scheduler.step()
    Train->>Disk: append metrics; periodically save complete state
```

## Value legend

| Symbol | Meaning | Stage 1 default |
|---|---|---:|
| `B` | batch size: examples processed per optimizer update | 32 |
| `T` | sequence length: token positions per example | 128 |
| `V` | GPT-2 BPE vocabulary size | 50,257 |
| `C` | learned token-vector / model dimension | 192 |

`target_tokens` holds an integer token ID at each `[B, T]` position. The model
creates one `V`-length logit vector for each position, then cross-entropy
compares it to that single correct target ID.
