"""Isolate the cuDNN SDPA cold-start hypothesis in fresh BF16 processes."""
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
from torch.nn.attention import SDPBackend, sdpa_kernel
import train
from model import TinyLM


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--without-cudnn',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.cuda.set_device(0)
    model=TinyLM().cuda()
    model.load_state_dict(torch.load(a.checkpoint,map_location='cpu',weights_only=False)['model_state'])
    model.eval();tokenizer=tiktoken.get_encoding('gpt2')
    backends=[SDPBackend.FLASH_ATTENTION,SDPBackend.EFFICIENT_ATTENTION,SDPBackend.MATH]
    if not a.without_cudnn:backends.append(SDPBackend.CUDNN_ATTENTION)
    def generate(tokens):
        with sdpa_kernel(backends),torch.autocast('cuda',dtype=torch.bfloat16):
            return train.generate(model,tokenizer,'ROMEO:',tokens,torch.device('cuda'))
    generate(8)
    torch.cuda.synchronize();start=time.perf_counter()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True) as prof:
        with torch.profiler.record_function('generation/cold80'):
            generate(80)
        torch.cuda.synchronize()
    cold=time.perf_counter()-start
    prof.export_chrome_trace(str(a.output/'generation-trace.json'))
    averages=prof.key_averages()
    (a.output/'operators.txt').write_text(averages.table(sort_by='cpu_time_total',row_limit=35))
    cpu_ops=[dict(name=e.key,calls=e.count,cpu_total_ms=e.cpu_time_total/1000,
                  cpu_self_ms=e.self_cpu_time_total/1000) for e in averages]
    warm=[]
    for _ in range(2):
        torch.cuda.synchronize();start=time.perf_counter();generate(80)
        torch.cuda.synchronize();warm.append(time.perf_counter()-start)
    result=dict(status='completed',without_cudnn=a.without_cudnn,
                cold_profiled_80_tokens_seconds=cold,warm_80_tokens_seconds=warm,
                warm_median_seconds=statistics.median(warm),
                cudnn_attention_cpu_total_ms=sum(e['cpu_total_ms'] for e in cpu_ops if e['name']=='aten::_cudnn_attention_forward'),
                attention_ops=[e for e in cpu_ops if 'attention' in e['name'].lower()],
                top_cpu_ops=sorted(cpu_ops,key=lambda e:-e['cpu_total_ms'])[:20])
    (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
