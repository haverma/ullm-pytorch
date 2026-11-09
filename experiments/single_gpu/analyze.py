"""Extract compact evidence from raw lab outputs without copying tensor files."""
import argparse
import collections
import hashlib
import json
import pickle
import statistics
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from analyze_scaling import profile_summary


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=ROOT/'runs/single-gpu-b200-2026-11-08')
    p.add_argument('--output',type=Path,default=ROOT/'findings/single-gpu-b200-2026-11-08')
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
    results=[]
    for path in sorted(a.root.glob('*/result.json')):
        r=json.loads(path.read_text());r['name']=path.parent.name
        results.append(r)
    groups=collections.defaultdict(list)
    for r in results:
        name=r['name']
        if '-r' in name and name.rsplit('-r',1)[1].isdigit() and r['status']=='completed':
            groups[name.rsplit('-r',1)[0]].append(r)
    aggregates=[]
    for name,rows in sorted(groups.items()):
        aggregates.append(dict(name=name,repeats=len(rows),
            timed_training_tokens_per_second=statistics.median(r['timed_training_tokens_per_second'] for r in rows),
            timed_training_tokens_per_second_range=[min(r['timed_training_tokens_per_second'] for r in rows),max(r['timed_training_tokens_per_second'] for r in rows)],
            wall_seconds=statistics.median(r['wall_seconds'] for r in rows),
            peak_allocated_gib=max(r['peak_allocated_bytes'] for r in rows)/2**30,
            peak_reserved_gib=max(r['peak_reserved_bytes'] for r in rows)/2**30,
            gpu_step_ms=statistics.median(r['gpu_event_ms']['step']['median'] for r in rows),
            evaluation_seconds=statistics.median(r['evaluation_seconds'] for r in rows),
            generation_seconds=statistics.median(r['generation_seconds'] for r in rows),
            checkpoint_seconds=statistics.median(r['checkpoint_seconds'] for r in rows),
            validation_loss=statistics.median(v) if (v:=[r['final_validation_loss'] for r in rows if r['final_validation_loss'] is not None]) else None))
    profiles=[]; manifest=[]
    for path in sorted(a.root.glob('*/trace.json')):
        profiles.append(profile_summary(path))
    for pattern in ('*/trace.json','*/*-trace.json','*/*.pickle','*/checkpoint.pt'):
        for path in sorted(a.root.glob(pattern)):
            digest=hashlib.sha256()
            with path.open('rb') as f:
                for chunk in iter(lambda:f.read(1024*1024),b''):digest.update(chunk)
            manifest.append(dict(path=str(path.relative_to(ROOT)),bytes=path.stat().st_size,sha256=digest.hexdigest()))
    compact=[]
    for r in results:
        copy=dict(r)
        if r['status']=='oom':
            # These snapshots are created by our own worker, not third-party files.
            snapshot=a.root/r['name']/'oom.pickle'
            if snapshot.exists():
                with snapshot.open('rb') as stream: allocator=pickle.load(stream)
                copy['allocator_oom_events']=[e for trace in allocator.get('device_traces',[])
                                              for e in trace if e.get('action')=='oom']
        series=copy.pop('memory_series',[])
        # Retention drills keep all points; normal jobs keep a small trend sample.
        if series:
            stride=max(1,len(series)//20)
            copy['memory_series']=series if len(series)<=30 else series[::stride]+[series[-1]]
        compact.append(copy)
    correctness=json.loads((a.root/'correctness.json').read_text()) if (a.root/'correctness.json').exists() else None
    compiled=[r for r in results if r.get('config',{}).get('compile') and r['status']=='completed']
    compilation=dict(repeats=len(compiled),
                     median_training_tokens_per_second=statistics.median(r['timed_training_tokens_per_second'] for r in compiled),
                     starts=[dict(name=r['name'],first_step_wall_seconds=r['first_step_wall_seconds'],wall_seconds=r['wall_seconds']) for r in compiled]) if compiled else None
    (a.output/'results.json').write_text(json.dumps(dict(aggregates=aggregates,runs=compact,profiles=profiles,correctness=correctness,compilation=compilation),indent=2)+'\n')
    (a.output/'artifact-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(aggregates=aggregates,
                         individual=[dict(name=r['name'],status=r['status'],
                                          training_tokens_per_second=r.get('timed_training_tokens_per_second'),
                                          peak_allocated_gib=r.get('peak_allocated_bytes',0)/2**30,
                                          validation_loss=r.get('final_validation_loss')) for r in results],
                         profiles=profiles),indent=2))


if __name__=='__main__':main()
