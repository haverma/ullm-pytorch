"""NCCL DDP trainer with a fixed global batch and legacy-compatible data order.

--batch-size is GLOBAL, not per rank. Each rank loads only its slice of the
same seeded global batch. This preserves the original trainer's batch stream.
"""
import hashlib
import json
import os
import time
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import tiktoken
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

import train
from data import download_and_tokenize, get_dataloader
from model import TinyLM


class ShardBatches:
    """Slice the legacy BatchSampler, retaining its RNG and epoch behavior."""
    def __init__(self, batches, rank, world_size):
        self.batches, self.rank, self.world_size = batches, rank, world_size

    def __iter__(self):
        for batch in self.batches:
            size = len(batch) // self.world_size
            yield batch[self.rank * size:(self.rank + 1) * size]

    def __len__(self):
        return len(self.batches)


def sharded_loader(args, rank, world_size):
    original = get_dataloader('train', args.batch_size, args.seq_length,
                              args.num_workers, args.seed)
    return DataLoader(original.dataset,
                      batch_sampler=ShardBatches(original.batch_sampler, rank, world_size),
                      num_workers=args.num_workers, generator=original.generator)


def make_profiler(args, rank):
    if not args.profile:
        return nullcontext(None)
    output = args.run_dir / 'profiler' / f'rank-{rank}'
    output.mkdir(parents=True, exist_ok=True)
    def save(p):
        p.export_chrome_trace(str(output / 'trace.json'))
        (output / 'operators.txt').write_text(p.key_averages().table(
            sort_by='self_cuda_time_total', row_limit=40))
    return torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=20, warmup=2, active=5, repeat=1),
        on_trace_ready=save, record_shapes=True, profile_memory=True)


def main():
    started = time.perf_counter()
    args = train.parse_args()
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world = int(os.environ['WORLD_SIZE'])
    if args.batch_size % world or args.batch_size < world:
        raise ValueError('--batch-size must be divisible by world size')
    if args.resume:
        raise ValueError('Resume is not implemented in this comparison trainer')
    if min(args.max_steps, args.eval_interval, args.eval_batches,
           args.checkpoint_interval, args.log_interval) <= 0 or args.weight_log_interval < 0:
        raise ValueError('Invalid interval or step count')
    torch.set_num_threads(1)
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    dist.init_process_group('nccl', timeout=timedelta(minutes=3), device_id=device)
    try:
        if rank == 0:
            args.run_dir.mkdir(parents=True, exist_ok=True)
            if (args.run_dir / 'summary.json').exists():
                raise RuntimeError('Use a fresh run directory')
            download_and_tokenize()
        dist.barrier()
        train.seed_everything(args.seed)
        loader = sharded_loader(args, rank, world)
        val_loader = get_dataloader('val', args.batch_size, args.seq_length,
                                    args.num_workers, args.seed)
        tokenizer = tiktoken.get_encoding('gpt2')
        model = TinyLM(tokenizer.n_vocab, args.seq_length, args.embed_dim,
                       args.n_heads, args.n_layers).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                     weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_steps)
        ddp = DistributedDataParallel(model, device_ids=[local_rank])
        initial = train.sampled_weights(model)
        initial_hash = hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes()
                                             for p in model.parameters())).hexdigest()
        iterator = iter(loader)
        metrics = []
        torch.cuda.reset_peak_memory_stats()
        dist.barrier()
        torch.cuda.synchronize()
        loop_start = time.perf_counter()
        weights = ((args.run_dir / 'weights.jsonl').open('w')
                   if rank == 0 and args.weight_log_interval else nullcontext(None))
        metric_file = ((args.run_dir / 'metrics.jsonl').open('w')
                       if rank == 0 else nullcontext(None))
        with weights as wf, metric_file as mf, make_profiler(args, rank) as profiler:
            if wf:
                train.log_weights(wf, model, 0, 'initial', initial)
            for step in range(1, args.max_steps + 1):
                region = torch.profiler.record_function if args.profile else lambda _: nullcontext()
                with region('train/load_batch'):
                    try:
                        inputs, targets = next(iterator)
                    except StopIteration:
                        iterator = iter(loader)
                        inputs, targets = next(iterator)
                with region('train/transfer'):
                    inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad(set_to_none=True)
                with region('train/forward'):
                    _, loss = ddp(inputs, targets)
                with region('train/backward'):
                    loss.backward()
                with region('train/optimizer'):
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    optimizer.step()
                    scheduler.step()
                if profiler:
                    profiler.step()
                final = step == args.max_steps
                evaluate_now = step % args.eval_interval == 0 or final
                log_now = step == 1 or step % args.log_interval == 0 or evaluate_now
                if log_now:
                    mean_loss = loss.detach().clone()
                    dist.all_reduce(mean_loss)
                    mean_loss /= world
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - loop_start
                    if rank == 0:
                        metric = dict(step=step, train_loss=mean_loss.item(),
                                      learning_rate=scheduler.get_last_lr()[0],
                                      gradient_norm=float(norm), elapsed_seconds=elapsed,
                                      tokens_per_second=step * args.batch_size * args.seq_length / elapsed)
                        if evaluate_now:
                            # Bypass DDP: only rank 0 evaluates/generates, using the
                            # exact same validation batches as the baseline.
                            metric['validation_loss'] = train.evaluate(model, val_loader, device, args.eval_batches)
                            metric['sample'] = train.generate(model, tokenizer, args.sample_prompt,
                                                              args.sample_tokens, device)
                        metrics.append(metric)
                        mf.write(json.dumps(metric) + '\n')
                        mf.flush()
                        print(json.dumps({k: v for k, v in metric.items() if k != 'sample'}), flush=True)
                    if evaluate_now:
                        dist.barrier()
                if wf and (step == 1 or step % args.weight_log_interval == 0 or final):
                    train.log_weights(wf, model, step, 'updated', initial)
                if step % args.checkpoint_interval == 0 or final:
                    if rank == 0:
                        train.save_checkpoint(args.checkpoint_dir / f'step-{step:06d}.pt',
                                              model, optimizer, scheduler, step, args)
                    dist.barrier()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        peaks = torch.tensor([torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()],
                             device=device, dtype=torch.int64)
        gathered = [torch.zeros_like(peaks) for _ in range(world)]
        dist.all_gather(gathered, peaks)
        # Verify replicas match after all updates (a checksum of every parameter).
        final_hash = hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes()
                                           for p in model.parameters())).hexdigest()
        hashes = [None] * world
        dist.all_gather_object(hashes, final_hash)
        assert len(set(hashes)) == 1, 'DDP replicas diverged'
        if rank == 0:
            vals = [r for r in metrics if 'validation_loss' in r]
            summary = dict(world_size=world, global_batch_size=args.batch_size,
                           local_batch_size=args.batch_size // world, steps=args.max_steps,
                           tokens=args.max_steps * args.batch_size * args.seq_length,
                           wall_seconds=elapsed,
                           end_to_end_tokens_per_second=args.max_steps * args.batch_size * args.seq_length / elapsed,
                           final_validation_loss=vals[-1]['validation_loss'],
                           best_validation=min(vals, key=lambda r: r['validation_loss']),
                           peak_allocated_bytes=max(p[0].item() for p in gathered),
                           peak_reserved_bytes=max(p[1].item() for p in gathered),
                           per_rank_peak_bytes=[p.tolist() for p in gathered],
                           initial_parameter_sha256=initial_hash,
                           final_parameter_sha256=final_hash, replicas_identical=True,
                           profile=args.profile, torch=torch.__version__, cuda=torch.version.cuda,
                           config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                           source_sha256={n: hashlib.sha256(Path(n).read_bytes()).hexdigest()
                                          for n in ('train.py', 'train_ddp.py', 'model.py', 'data.py')})
            (args.run_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
            print(json.dumps(summary, indent=2), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
