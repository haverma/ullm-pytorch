"""Render standard matplotlib figures from compact lab evidence (no GPU needed)."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser()
    p.add_argument('findings',type=Path)
    a=p.parse_args()
    data=json.loads((a.findings/'results.json').read_text())
    groups={r['name']:r for r in data['aggregates']}
    runs={r['name']:r for r in data['runs']}
    fig,axes=plt.subplots(2,3,figsize=(17,9),constrained_layout=True)
    fig.suptitle('Single B200: controlled performance and debugging experiments',fontsize=17)
    ax=axes[0,0]
    batches=[b for b in (32,64,128,256,512,1024) if f'batch-{b}' in groups]
    rates=[groups[f'batch-{b}']['timed_training_tokens_per_second']/1000 for b in batches]
    ranges=[groups[f'batch-{b}']['timed_training_tokens_per_second_range'] for b in batches]
    ax.errorbar(batches,rates,yerr=[[v-lo/1000 for v,(lo,hi) in zip(rates,ranges)],
                                  [hi/1000-v for v,(lo,hi) in zip(rates,ranges)]],
                marker='o',capsize=3,color='#2563eb')
    ax.set_xscale('log',base=2);ax.set_xticks(batches,labels=batches)
    ax.set(title='Batch size vs training throughput',xlabel='Effective batch',ylabel='Thousand tokens / second')
    ax=axes[0,1]
    modes=['fp32-baseline','bf16-baseline'];x=[0,1]
    rates=[groups[m]['timed_training_tokens_per_second']/1000 for m in modes]
    ranges=[groups[m]['timed_training_tokens_per_second_range'] for m in modes]
    ax.bar(x,rates,yerr=[[v-lo/1000 for v,(lo,hi) in zip(rates,ranges)],
                        [hi/1000-v for v,(lo,hi) in zip(rates,ranges)]],capsize=4,
           color=['#2563eb','#0d9488'])
    ax.set_xticks(x,labels=['FP32','BF16']);ax.set(title='Precision: steady training',ylabel='Thousand tokens / second')
    ax=axes[0,2]
    keys=['memory-training-control','memory-training-accum','memory-training-checkpoint','memory-training-both']
    ax.bar(range(4),[runs[k]['peak_allocated_bytes']/2**30 for k in keys],color=['#2563eb','#0d9488','#d97706','#7c3aed'])
    ax.set_xticks(range(4),labels=['Control','Accum.','Block\ncheckpoint','Both'])
    ax.set(title='Training-only peak allocation',ylabel='GiB')
    ax=axes[1,0]
    for key,label,color in [('detached-control','Healthy','#0d9488'),('retained-graphs','Retained graphs','#dc2626')]:
        series=runs[key]['memory_series']
        ax.plot([r['step'] for r in series],[r['allocated_bytes']/2**30 for r in series],marker='o',label=label,color=color)
    ax.set(title='Retaining graphs creates memory growth',xlabel='Optimizer update',ylabel='Live allocated GiB');ax.legend()
    ax=axes[1,1]
    keys=['input-baseline','input-pinned','stall-workers0','stall-workers2']
    ax.bar(range(4),[groups[k]['timed_training_tokens_per_second']/1000 for k in keys],color=['#2563eb','#0d9488','#dc2626','#7c3aed'])
    ax.set_xticks(range(4),labels=['Normal\n0 workers','Normal\n2 + pinned','Stall\n0 workers','Stall\n2 + pinned'])
    ax.set(title='Input tuning depends on the bottleneck',ylabel='Thousand tokens / second')
    ax=axes[1,2]
    values=[groups[m]['generation_seconds'] for m in modes]
    ax.bar(x,values,color=['#2563eb','#0d9488'])
    ax.set_xticks(x,labels=['FP32','BF16'])
    ax.set(title='Generation overhead in 1,300-step jobs\n(includes cold backend costs)',ylabel='Total generation seconds')
    for ax in axes.flat:
        ax.grid(axis='y',alpha=0.25);ax.set_axisbelow(True)
        ax.spines[['top','right']].set_visible(False)
    fig.savefig(a.findings/'summary.png',dpi=160)
    fig.savefig(a.findings/'summary.svg')


if __name__=='__main__':main()
