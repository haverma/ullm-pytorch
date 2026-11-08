"""Aggregate the benchmark runs and summarize captured GPU kernel events."""
import argparse
import collections
import json
import statistics
from pathlib import Path


def profile_summary(path):
    events = json.loads(path.read_text())['traceEvents']
    steps = [e for e in events if e.get('name', '').startswith('ProfilerStep#') and 'dur' in e]
    kernels = [e for e in events if e.get('cat') == 'kernel' and 'dur' in e]
    durations = collections.defaultdict(float)
    for e in kernels:
        durations[e['name']] += e['dur'] / 1000
    nccl = sum(v for k, v in durations.items() if 'nccl' in k.lower())
    regions = collections.defaultdict(float)
    for e in events:
        if e.get('cat') == 'user_annotation' and e.get('name', '').startswith('train/'):
            regions[e['name']] += e.get('dur', 0) / 1000
    return dict(path=str(path), captured_steps=len(steps),
                profiled_cpu_step_ms=statistics.mean(e['dur'] / 1000 for e in steps),
                gpu_kernel_ms_per_step=sum(durations.values()) / len(steps),
                nccl_kernel_ms_per_step=nccl / len(steps),
                nccl_kernel_time_fraction=nccl / sum(durations.values()),
                cpu_region_ms_per_step={k: v / len(steps) for k, v in regions.items()},
                top_kernels_ms=sorted(durations.items(), key=lambda x: -x[1])[:10])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', nargs='?', type=Path, default=Path('runs/comparison-1300'))
    args = parser.parse_args()
    groups = collections.defaultdict(list)
    for path in sorted(args.root.glob('*/summary.json')):
        row = json.loads(path.read_text())
        if not row['profile']:
            row['run'] = path.parent.name
            metrics = [json.loads(line) for line in (path.parent / 'metrics.jsonl').read_text().splitlines()]
            row['loop_elapsed_before_final_eval'] = metrics[-1]['elapsed_seconds']
            groups[row['world_size']].append(row)
    baseline = statistics.median(r['wall_seconds'] for r in groups[1])
    aggregated = []
    for world, rows in sorted(groups.items()):
        wall = statistics.median(r['wall_seconds'] for r in rows)
        aggregated.append(dict(
            world_size=world, repeats=len(rows), local_batch_size=32 // world,
            wall_seconds_median=wall,
            wall_seconds_range=[min(r['wall_seconds'] for r in rows), max(r['wall_seconds'] for r in rows)],
            loop_elapsed_before_final_eval_median=statistics.median(r['loop_elapsed_before_final_eval'] for r in rows),
            tokens_per_second_median=rows[0]['tokens'] / wall,
            speedup=baseline / wall, scaling_efficiency=baseline / wall / world,
            gpu_seconds=world * wall,
            final_validation_loss_median=statistics.median(r['final_validation_loss'] for r in rows),
            peak_allocated_gib=max(r['peak_allocated_bytes'] for r in rows) / 2**30,
            peak_reserved_gib=max(r['peak_reserved_bytes'] for r in rows) / 2**30,
            runs=[dict(run=r['run'], wall_seconds=r['wall_seconds'],
                       final_validation_loss=r['final_validation_loss']) for r in rows]))
    result = dict(aggregate=aggregated,
                  profiles=[profile_summary(p) for p in sorted(args.root.glob('*/profiler/**/trace.json'))])
    (args.root / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
