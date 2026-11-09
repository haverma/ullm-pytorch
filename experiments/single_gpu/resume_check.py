"""Exercise the original trainer's GPU checkpoint/resume in separate processes."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]


def same(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(same(v,b[k]) for k,v in a.items())
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    common=['--device','cuda:0','--seed','42','--batch-size','32','--seq-length','128',
            '--max-steps','20','--eval-interval','10','--eval-batches','2',
            '--checkpoint-interval','10','--sample-tokens','8','--no-profile']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    entry=[sys.executable,'-c','import torch,train; torch.use_deterministic_algorithms(True); train.main()']
    def launch(label,extra):
        command=entry+common+['--run-dir',str(a.output/label),
                             '--checkpoint-dir',str(a.output/(label+'-checkpoints'))]+extra
        with (a.output/(label+'.log')).open('w') as log:
            return subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=180).returncode
    result=dict(reference_returncode=launch('reference',[]))
    if result['reference_returncode']:
        result['status']='reference_failed'
    else:
        checkpoint=a.output/'reference-checkpoints/step-000010.pt'
        result['resume_returncode']=launch('resumed',['--resume',str(checkpoint)])
        if result['resume_returncode']:
            result['status']='resume_failed'
            result['error_log']=(a.output/'resumed.log').read_text()[-8000:]
        else:
            ref=torch.load(a.output/'reference-checkpoints/step-000020.pt',map_location='cpu',weights_only=False)
            resumed=torch.load(a.output/'resumed-checkpoints/step-000020.pt',map_location='cpu',weights_only=False)
            result.update(status='completed',
                parameters_bit_exact=same(ref['model_state'],resumed['model_state']),
                parameter_max_abs_difference=max((ref['model_state'][k]-v).abs().max().item() for k,v in resumed['model_state'].items()),
                optimizer_bit_exact=same(ref['optimizer_state'],resumed['optimizer_state']),
                scheduler_equal=same(ref['scheduler_state'],resumed['scheduler_state']),
                rng_state_equal=same(ref['rng_state'],resumed['rng_state']),
                completed_steps_equal=ref['completed_steps']==resumed['completed_steps']==20)
    (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
