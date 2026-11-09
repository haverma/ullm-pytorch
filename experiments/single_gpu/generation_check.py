"""Separate generation timing/profiles to investigate autocast behavior."""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import tiktoken
import torch
import train
from model import TinyLM


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.cuda.set_device(0)
    model=TinyLM().cuda()
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    model.load_state_dict(state['model_state']);model.eval()
    tokenizer=tiktoken.get_encoding('gpt2')
    results=[]
    for precision in ('fp32','bf16'):
        def context():return torch.autocast('cuda',dtype=torch.bfloat16,enabled=precision=='bf16')
        with context():train.generate(model,tokenizer,'ROMEO:',8,torch.device('cuda'))
        times=[]
        for _ in range(3):
            torch.cuda.synchronize();start=time.perf_counter()
            with context():text=train.generate(model,tokenizer,'ROMEO:',80,torch.device('cuda'))
            torch.cuda.synchronize();times.append(time.perf_counter()-start)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True) as prof:
            with torch.profiler.record_function('generation'),context():
                train.generate(model,tokenizer,'ROMEO:',12,torch.device('cuda'))
            torch.cuda.synchronize()
        prof.export_chrome_trace(str(a.output/f'{precision}-trace.json'))
        (a.output/f'{precision}-operators.txt').write_text(prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=40))
        results.append(dict(precision=precision,seconds=times,median_seconds=statistics.median(times),generated_tokens=80,sample=text))
    (a.output/'result.json').write_text(json.dumps(dict(status='completed',checkpoint=str(a.checkpoint),generation=results),indent=2)+'\n')
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
