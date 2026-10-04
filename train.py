"""Train the Stage 1 decoder-only language model.

This intentionally uses an explicit PyTorch training loop: the goal is to make
every step visible before later stages add profiling, distribution, and
orchestration.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tiktoken

from data import download_and_tokenize, get_dataloader
from model import TinyLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small GPT-style language model.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-length", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=1_000)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--checkpoint-interval", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--embed-dim", type=int, default=192)
    parser.add_argument("--n-heads", type=int, default=6)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, cuda, or cuda:0")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--run-dir", type=Path, default=Path("runs"))
    parser.add_argument("--resume", type=Path, help="Checkpoint path to resume from")
    parser.add_argument("--sample-prompt", default="ROMEO:")
    parser.add_argument("--sample-tokens", type=int, default=80)
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


@torch.no_grad()
def evaluate(model: TinyLM, loader: torch.utils.data.DataLoader, device: torch.device, batches: int) -> float:
    model_was_training = model.training
    model.eval()
    losses: list[float] = []
    for batch_index, (inputs, targets) in enumerate(loader):
        if batch_index >= batches:
            break
        _, loss = model(inputs.to(device), targets.to(device))
        assert loss is not None
        losses.append(loss.item())
    if model_was_training:
        model.train()
    return sum(losses) / len(losses)


@torch.no_grad()
def generate(
    model: TinyLM,
    tokenizer: tiktoken.Encoding,
    prompt: str,
    tokens_to_generate: int,
    device: torch.device,
) -> str:
    model_was_training = model.training
    model.eval()
    token_ids = tokenizer.encode(prompt, allowed_special={"<|endoftext|>"})
    tokens = torch.tensor([token_ids], dtype=torch.long, device=device)

    for _ in range(tokens_to_generate):
        context = tokens[:, -model.max_seq_len :]
        logits, _ = model(context)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        tokens = torch.cat((tokens, next_token), dim=1)

    if model_was_training:
        model.train()
    return tokenizer.decode(tokens[0].tolist())


def save_checkpoint(
    path: Path,
    model: TinyLM,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    completed_steps: int,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "completed_steps": completed_steps,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "rng_state": capture_rng_state(),
            "config": vars(args),
        },
        path,
    )
    print(f"Saved checkpoint: {path}")


def restore_checkpoint(
    path: Path,
    model: TinyLM,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
) -> int:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler.load_state_dict(checkpoint["scheduler_state"])
    restore_rng_state(checkpoint["rng_state"])
    completed_steps = int(checkpoint["completed_steps"])
    print(f"Resumed from {path} after {completed_steps} completed steps.")
    return completed_steps


def make_train_iterator(loader: torch.utils.data.DataLoader, batches_to_skip: int):
    """Create the deterministic shuffled stream and advance it when resuming.

    With the same seed and num_workers=0, this reproduces the same batch order.
    It is deliberately simple for Stage 1; a distributed sampler will replace it
    in Stage 3.
    """
    iterator = iter(loader)
    for _ in range(batches_to_skip):
        try:
            next(iterator)
        except StopIteration:
            iterator = iter(loader)
            next(iterator)
    return iterator


def main() -> None:
    args = parse_args()
    if min(args.max_steps, args.eval_interval, args.eval_batches, args.checkpoint_interval) <= 0:
        raise ValueError(
            "max_steps, eval_interval, eval_batches, and checkpoint_interval must be positive"
        )

    device = choose_device(args.device)
    seed_everything(args.seed)
    print(f"Using device: {device}")

    download_and_tokenize()
    train_loader = get_dataloader(
        split="train",
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    val_loader = get_dataloader(
        split="val",
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    tokenizer = tiktoken.get_encoding("gpt2")
    model = TinyLM(
        vocab_size=tokenizer.n_vocab,
        max_seq_len=args.seq_length,
        embed_dim=args.embed_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_steps)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Model parameters: {parameter_count / 1e6:.2f}M")

    completed_steps = 0
    if args.resume:
        completed_steps = restore_checkpoint(args.resume, model, optimizer, scheduler, device)
        if completed_steps >= args.max_steps:
            raise ValueError("Checkpoint has already reached --max-steps")

    args.run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.run_dir / "metrics.jsonl"
    train_iterator = make_train_iterator(train_loader, completed_steps)
    model.train()
    started_at = time.perf_counter()

    for step in range(completed_steps + 1, args.max_steps + 1):
        try:
            inputs, targets = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            inputs, targets = next(train_iterator)

        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(inputs, targets)
        assert loss is not None
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()

        elapsed_seconds = time.perf_counter() - started_at
        tokens_per_second = (step - completed_steps) * inputs.numel() / elapsed_seconds
        metric = {
            "step": step,
            "train_loss": loss.item(),
            "learning_rate": scheduler.get_last_lr()[0],
            "gradient_norm": float(gradient_norm),
            "tokens_per_second": tokens_per_second,
            "elapsed_seconds": elapsed_seconds,
        }

        if step % args.eval_interval == 0 or step == args.max_steps:
            metric["validation_loss"] = evaluate(model, val_loader, device, args.eval_batches)
            metric["sample"] = generate(
                model, tokenizer, args.sample_prompt, args.sample_tokens, device
            )
            print(
                f"step={step:>5} train_loss={metric['train_loss']:.4f} "
                f"val_loss={metric['validation_loss']:.4f} "
                f"tokens/sec={tokens_per_second:.0f}"
            )
            print(f"Sample:\n{metric['sample']}\n")

        with metrics_path.open("a", encoding="utf-8") as metrics_file:
            metrics_file.write(json.dumps(metric) + "\n")

        if step % args.checkpoint_interval == 0 or step == args.max_steps:
            save_checkpoint(
                args.checkpoint_dir / f"step-{step:06d}.pt",
                model,
                optimizer,
                scheduler,
                step,
                args,
            )


if __name__ == "__main__":
    main()
