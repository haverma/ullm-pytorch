"""Run with torchrun --standalone --nproc_per_node=2 check_ddp.py.

Checks actual data-loader ordering and full-model gradient equivalence before
trusting scaling results. This is a correctness check, not a performance run.
"""
import hashlib
import json
import os
from datetime import timedelta
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

import train
from data import get_dataloader
from model import TinyLM
from train_ddp import sharded_loader


def main():
    rank, local, world = (int(os.environ[n]) for n in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE'))
    torch.set_num_threads(1)
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    dist.init_process_group('nccl', device_id=device, timeout=timedelta(minutes=3))
    try:
        args = SimpleNamespace(batch_size=32, seq_length=128, num_workers=0, seed=42)
        original = get_dataloader('train', 32, 128, 0, 42)
        sharded = sharded_loader(args, rank, world)
        original_it, shard_it = iter(original), iter(sharded)
        for _ in range(8):
            full_x, full_y = next(original_it)
            x, y = next(shard_it)
            width = 32 // world
            assert torch.equal(x, full_x[rank * width:(rank + 1) * width])
            assert torch.equal(y, full_y[rank * width:(rank + 1) * width])
        train.seed_everything(42)
        model = TinyLM().to(device)
        reference = TinyLM().to(device)
        reference.load_state_dict(model.state_dict())
        ddp = DistributedDataParallel(model, device_ids=[local])
        _, loss = ddp(x.to(device), y.to(device))
        loss.backward()
        _, ref_loss = reference(full_x.to(device), full_y.to(device))
        ref_loss.backward()
        max_error = 0.0
        for actual, expected in zip(model.parameters(), reference.parameters()):
            max_error = max(max_error, (actual.grad - expected.grad).abs().max().item())
            torch.testing.assert_close(actual.grad, expected.grad, rtol=2e-3, atol=2e-6)
        averaged_loss = loss.detach().clone()
        dist.all_reduce(averaged_loss)
        averaged_loss /= world
        torch.testing.assert_close(averaged_loss, ref_loss, rtol=1e-5, atol=1e-6)
        if rank == 0:
            print(json.dumps(dict(data_shards_match_legacy=True, batches_checked=8,
                                  gradient_equivalence=True, max_gradient_absolute_error=max_error,
                                  single_loss=ref_loss.item(), ddp_loss=averaged_loss.item()), indent=2))
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
