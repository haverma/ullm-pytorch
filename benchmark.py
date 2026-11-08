"""Measure an unchanged trainer; profile a separate diagnostic run after warmup."""
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
import train


def main():
    torch.set_num_threads(1)
    args = train.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    if (args.run_dir / "summary.json").exists():
        raise RuntimeError("Use a fresh run directory")
    if args.profile:
        def profiler(enabled, device, run_dir):
            if not enabled:
                from contextlib import nullcontext
                return nullcontext(None)
            output = run_dir / "profiler"
            output.mkdir(exist_ok=True)
            def save(p):
                p.export_chrome_trace(str(output / "trace.json"))
                (output / "operators.txt").write_text(p.key_averages().table(
                    sort_by="self_cuda_time_total", row_limit=40))
            return torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=20, warmup=2, active=5, repeat=1),
                on_trace_ready=save, record_shapes=True, profile_memory=True)
        train.make_profiler = profiler
        # The legacy trainer labels only its first five steps. Label the diagnostic
        # run throughout so the later, warmed-up capture has the same regions.
        train.profile_region = lambda name, enabled: torch.profiler.record_function(name)
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    train.main()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    rows = [json.loads(line) for line in (args.run_dir / "metrics.jsonl").read_text().splitlines()]
    validations = [r for r in rows if "validation_loss" in r]
    summary = {
        "world_size": 1, "global_batch_size": args.batch_size,
        "local_batch_size": args.batch_size, "steps": args.max_steps,
        "tokens": args.max_steps * args.batch_size * args.seq_length,
        "wall_seconds": elapsed,
        "end_to_end_tokens_per_second": args.max_steps * args.batch_size * args.seq_length / elapsed,
        "final_validation_loss": validations[-1]["validation_loss"],
        "best_validation": min(validations, key=lambda r: r["validation_loss"]),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "profile": args.profile, "torch": torch.__version__, "cuda": torch.version.cuda,
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "source_sha256": {n: hashlib.sha256(Path(n).read_bytes()).hexdigest()
                          for n in ("train.py", "model.py", "data.py", "benchmark.py")},
    }
    (args.run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
