"""Audit all requested experiment outcomes and numerical equivalence evidence."""
import argparse
import json
import math
from pathlib import Path

import torch

ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=ROOT/'runs/single-gpu-b200-2026-11-08')
    p.add_argument('--output',type=Path,default=ROOT/'findings/single-gpu-b200-2026-11-08')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
    runs={path.parent.name:json.loads(path.read_text()) for path in a.root.glob('*/result.json')}
    expected={f'{precision}-baseline-r{r}' for precision in ('fp32','bf16') for r in (1,2,3)}
    expected|={f'batch-{b}-r{r}' for b in (32,64,128,256,512,1024) for r in (1,2,3)}
    expected|={f'{name}-r{r}' for name in ('input-baseline','input-workers2','input-pinned','input-prefetch4','stall-workers0','stall-workers2') for r in (1,2,3)}
    expected|=set(('baseline-profile','bf16-profile','oom-batch-128','oom-batch-512','oom-batch-1024','oom-batch-2048',
                  'oom-recovery-batch-2048-micro32','accumulation-8x4','activation-checkpointing',
                  'accumulation-and-checkpointing','memory-training-control','memory-training-accum',
                  'memory-training-checkpoint','memory-training-both','detached-control','retained-graphs',
                  'post-oom-health','physical-oom-batch-2048','post-physical-oom-health','stall-profile',
                  'stall-fixed-profile','nan-loss','nan-gradient','finite-control','compiled-fp32',
                  'compiled-fp32-cached-r2','compiled-fp32-cached-r3','resume-after-fix','generation-check',
                  'generation-backend-auto','generation-backend-no-cudnn'))
    assert expected<=runs.keys(),sorted(expected-runs.keys())
    allowed={'oom-batch-2048':'oom','physical-oom-batch-2048':'oom',
             'nan-loss':'nonfinite_detected','nan-gradient':'nonfinite_detected',
             'resume-before-fix':'resume_failed'}
    initials=set();model_sources=set();data_sources=set()
    for name,r in runs.items():
        assert r['status']==allowed.get(name,'completed'),(name,r['status'])
        if 'initial_parameter_sha256' in r:initials.add(r['initial_parameter_sha256'])
        if 'source_sha256' in r:
            model_sources.add(r['source_sha256']['model.py']);data_sources.add(r['source_sha256']['data.py'])
        if r.get('total_training_tokens') is not None:
            c=r['config']
            assert r['total_training_tokens']==c['steps']*c['batch_size']*c['seq_length']
            assert r['timed_optimizer_steps']==c['steps']-c['warmup']
            assert r['timed_training_tokens_per_second']>0
        for metric in r.get('metrics',[]):
            assert all(math.isfinite(v) for k,v in metric.items() if isinstance(v,(int,float)))
    assert len(initials)==len(model_sources)==len(data_sources)==1
    for name in ('nan-loss','nan-gradient'):assert runs[name]['failure']['step']==5
    resume=runs['resume-after-fix']
    assert all(resume[k] for k in ('parameters_bit_exact','optimizer_bit_exact','scheduler_equal','rng_state_equal','completed_steps_equal'))
    gradient=json.loads((a.root/'correctness.json').read_text())
    assert len(gradient['checks'])==4
    assert max(r['max_gradient_absolute_error'] for r in gradient['checks'])<2e-6
    baseline=torch.load(a.root/'fp32-baseline-r1/checkpoint.pt',map_location='cpu',weights_only=False)['model_state']
    comparisons=[]
    for name in ('compiled-fp32','compiled-fp32-cached-r2','compiled-fp32-cached-r3'):
        state=torch.load(a.root/name/'checkpoint.pt',map_location='cpu',weights_only=False)['model_state']
        maximum=max((state[k]-v).abs().max().item() for k,v in baseline.items())
        squared=sum((state[k].double()-v.double()).square().sum().item() for k,v in baseline.items())
        count=sum(v.numel() for v in baseline.values())
        loss_delta=abs(runs[name]['final_validation_loss']-runs['fp32-baseline-r1']['final_validation_loss'])
        assert loss_delta<0.01
        comparisons.append(dict(name=name,max_parameter_absolute_difference=maximum,
                                parameter_rms_difference=math.sqrt(squared/count),validation_loss_absolute_difference=loss_delta,
                                first_loss_absolute_difference=abs(runs[name]['first_training_loss']-runs['fp32-baseline-r1']['first_training_loss'])))
    result=dict(all_expected_outcomes_verified=True,core_cases=len(expected),total_results=len(runs),
                identical_initial_parameter_hash=True,identical_model_and_data_source=True,
                gradient_checks=gradient,compiled_comparisons=comparisons,
                expected_failures={k:runs[k]['status'] for k in allowed if k in runs},
                deterministic_resume=resume)
    (a.output/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
