"""Train the Stage 1 decoder-only language model.

This intentionally uses an explicit PyTorch training loop: the goal is to make
every step visible before later stages add distribution and orchestration.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from contextlib import nullcontext
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
    parser.add_argument("--log-interval", type=int, default=100,
                        help="Write metrics every N steps, plus the first and final step")
    parser.add_argument("--weight-log-interval", type=int, default=100,
                        help="Sample weights every N steps; 0 disables weight logging")
    parser.add_argument("--profile", action=argparse.BooleanOptionalAction, default=False,
                        help="Capture a short PyTorch CPU/CUDA profile (default: off)")
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


def sampled_weights(model: TinyLM) -> dict[str, dict[str, Any]]:
    """Copy only five fixed values from three representative weight tensors."""
    names = ("tok_emb.weight", "blocks.0.attn.c_attn.weight", "blocks.0.mlp.c_fc.weight")
    parameters = dict(model.named_parameters())
    return {
        name: {
            "shape": list(parameters[name].shape),
            "dtype": str(parameters[name].dtype),
            "flat_indices": list(range(min(5, parameters[name].numel()))),
            "values": parameters[name].detach().view(-1)[:5].cpu().tolist(),
        }
        for name in names
    }


def log_weights(stream, model: TinyLM, step: int, event: str,
                baseline: dict[str, dict[str, Any]]) -> None:
    samples = sampled_weights(model) if event == "updated" else dict(baseline)
    for name, sample in samples.items():
        sample = dict(sample)
        sample["delta_from_start"] = [
            current - initial
            for current, initial in zip(sample["values"], baseline[name]["values"])
        ]
        samples[name] = sample
    stream.write(json.dumps({"step": step, "event": event, "weights": samples}) + "\n")
    stream.flush()


def make_profiler(enabled: bool, device: torch.device, run_dir: Path):
    if not enabled:
        return nullcontext(None)
    profile_dir = run_dir / "profiler"
    profile_dir.mkdir(parents=True, exist_ok=True)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    def save_profile(profiler):
        # Unique names keep profiles from resumed/repeated runs separate.
        stem = profile_dir / f"trace-{time.time_ns()}"
        profiler.export_chrome_trace(str(stem.with_suffix(".json")))
        sort_key = "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
        stem.with_suffix(".txt").write_text(
            profiler.key_averages().table(sort_by=sort_key, row_limit=20), encoding="utf-8"
        )
        print(f"Saved profiler trace and summary: {stem}")

    return torch.profiler.profile(
        activities=activities,
        # Skip one step, warm up one, capture three, then stop recording.
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=save_profile,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    )


def profile_region(name: str, enabled: bool):
    return torch.profiler.record_function(name) if enabled else nullcontext()


def main() -> None:
    args = parse_args()
    if min(args.max_steps, args.eval_interval, args.eval_batches,
           args.checkpoint_interval, args.log_interval) <= 0:
        raise ValueError(
            "max_steps, eval_interval, eval_batches, checkpoint_interval, and log_interval must be positive"
        )
    if args.weight_log_interval < 0:
        raise ValueError("weight_log_interval must be nonnegative")

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
    weight_context = (
        (args.run_dir / "weights.jsonl").open("a", encoding="utf-8")
        if args.weight_log_interval else nullcontext(None)
    )
    with weight_context as weight_file, metrics_path.open("a", encoding="utf-8") as metrics_file:
        baseline = sampled_weights(model) if weight_file is not None else {}
        if weight_file is not None:
            log_weights(weight_file, model, completed_steps,
                        "resumed" if args.resume else "initial", baseline)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started_at = time.perf_counter()

        with make_profiler(args.profile, device, args.run_dir) as profiler:
            for step in range(completed_steps + 1, args.max_steps + 1):
                capture_regions = args.profile and step - completed_steps <= 5
                with profile_region("train/load_batch", capture_regions):
                    try:
                        inputs, targets = next(train_iterator)
                    except StopIteration:
                        train_iterator = iter(train_loader)
                        inputs, targets = next(train_iterator)

                with profile_region("train/transfer", capture_regions):
                    inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad(set_to_none=True)
                with profile_region("train/forward", capture_regions):
                    _, loss = model(inputs, targets)
                    assert loss is not None
                with profile_region("train/backward", capture_regions):
                    loss.backward()
                with profile_region("train/optimizer", capture_regions):
                    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    optimizer.step()
                    scheduler.step()

                # Mark the next profiler interval; telemetry is sparse by default.
                if profiler is not None:
                    profiler.step()
                first_step = step == completed_steps + 1
                final_step = step == args.max_steps
                evaluate_now = step % args.eval_interval == 0 or final_step
                log_now = first_step or step % args.log_interval == 0 or evaluate_now
                if log_now:
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
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
                    if evaluate_now:
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
                    metrics_file.write(json.dumps(metric) + "\n")
                    metrics_file.flush()

                if weight_file is not None and (
                    first_step or step % args.weight_log_interval == 0 or final_step
                ):
                    log_weights(weight_file, model, step, "updated", baseline)

                if step % args.checkpoint_interval == 0 or final_step:
                    save_checkpoint(
                        args.checkpoint_dir / f"step-{step:06d}.pt",
                        model, optimizer, scheduler, step, args,
                    )


if __name__ == "__main__":
    main()
