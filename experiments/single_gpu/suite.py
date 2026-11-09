"""Run experiments sequentially in isolated GPU processes; persist every outcome."""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKER = ROOT / 'experiments/single_gpu/worker.py'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=ROOT/'runs/single-gpu-b200-2026-11-08')
    p.add_argument('--stage', choices=['performance', 'memory', 'pipeline', 'numerical', 'compile'], required=True)
    a = p.parse_args()
    a.root.mkdir(parents=True, exist_ok=True)
    progress = a.root / f'{a.stage}-progress.json'
    outcomes = json.loads(progress.read_text()) if progress.exists() else []
    def launch(name, flags, timeout=240, expected=('completed',)):
        output = a.root / name
        if (output/'result.json').exists():
            result = json.loads((output/'result.json').read_text())
            print(f'Existing result: {name}: {result["status"]}', flush=True)
            return result
        if output.exists():
            raise RuntimeError(f'Incomplete prior output exists: {output}')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='1',
                   PYTHONUNBUFFERED='1', TORCHINDUCTOR_COMPILE_THREADS='4',
                   TORCHINDUCTOR_CACHE_DIR=str(a.root/'compile-cache'),
                   CUBLAS_WORKSPACE_CONFIG=':4096:8')
        command = [sys.executable, str(WORKER), '--output', str(output), *flags]
        print(f'Starting {name}', flush=True)
        start = time.perf_counter()
        with (a.root/f'{name}.log').open('w') as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                code = -1
        result = json.loads((output/'result.json').read_text()) if (output/'result.json').exists() else dict(status='process_failed')
        record = dict(name=name, status=result['status'], returncode=code,
                      seconds=time.perf_counter()-start, command=command)
        outcomes.append(record)
        (a.root/f'{a.stage}-progress.json').write_text(json.dumps(outcomes,indent=2)+'\n')
        print(json.dumps(record),flush=True)
        if code != 0 or result['status'] not in expected:
            print(f'Unexpected outcome; see {a.root/name}.log',flush=True)
        return result
    if a.stage == 'performance':
        for precision in ('fp32','bf16'):
            for repeat in range(1,4):
                launch(f'{precision}-baseline-r{repeat}', ['--precision',precision,'--save-checkpoint'])
        for batch in (32,64,128,256,512,1024):
            # Same token budget and cosine progress in token space; updates differ.
            for repeat in range(1,4):
                launch(f'batch-{batch}-r{repeat}', ['--batch-size',str(batch),
                       '--steps',str(32768//batch),'--save-checkpoint'])
        launch('baseline-profile',['--steps','80','--eval-batches','0','--profile'])
        launch('bf16-profile',['--steps','80','--eval-batches','0','--profile','--precision','bf16'])
    elif a.stage == 'memory':
        boundary = []
        for batch in (128,512,1024,2048,4096):
            result = launch(f'oom-batch-{batch}', ['--batch-size',str(batch),'--steps','2',
                            '--warmup','0','--eval-batches','0','--snapshot','--memory-fraction','0.85'],
                            timeout=180, expected=('completed','oom'))
            boundary.append(dict(batch=batch,status=result['status'],peak_allocated_bytes=result.get('peak_allocated_bytes')))
            if result['status'] == 'oom':
                launch(f'oom-recovery-batch-{batch}-micro32',
                       ['--batch-size',str(batch),'--microbatch','32','--steps','3',
                        '--warmup','1','--eval-batches','0','--memory-fraction','0.85'], timeout=240)
                break
        (a.root/'oom-boundary.json').write_text(json.dumps(boundary,indent=2)+'\n')
        launch('accumulation-8x4',['--microbatch','8','--save-checkpoint'])
        launch('activation-checkpointing',['--activation-checkpointing','--save-checkpoint'])
        launch('accumulation-and-checkpointing',['--microbatch','8','--activation-checkpointing','--save-checkpoint'])
        for name,flags in (
            ('memory-training-control',[]),
            ('memory-training-accum',['--microbatch','8']),
            ('memory-training-checkpoint',['--activation-checkpointing']),
            ('memory-training-both',['--microbatch','8','--activation-checkpointing'])):
            launch(name,['--steps','80','--eval-batches','0',*flags])
        for retain in (False,True):
            launch('retained-graphs' if retain else 'detached-control',
                   ['--batch-size','8','--steps','12','--warmup','2','--eval-batches','0',
                    '--snapshot','--memory-fraction','0.25'] + (['--retain-graphs'] if retain else []))
        launch('post-oom-health',['--steps','30','--eval-batches','1'])
        launch('physical-oom-batch-2048',['--batch-size','2048','--steps','2',
               '--warmup','0','--eval-batches','0','--snapshot'],expected=('completed','oom'))
        launch('post-physical-oom-health',['--steps','30','--eval-batches','1'])
    elif a.stage == 'pipeline':
        cases = [
            ('input-baseline', []),
            ('input-workers2', ['--workers','2']),
            ('input-pinned', ['--workers','2','--pin-memory','--nonblocking']),
            ('input-prefetch4', ['--workers','2','--pin-memory','--nonblocking','--prefetch','4']),
            ('stall-workers0', ['--delay-ms','30']),
            ('stall-workers2', ['--delay-ms','30','--workers','2','--pin-memory','--nonblocking']),
        ]
        for name, flags in cases:
            for repeat in range(1,4):
                launch(f'{name}-r{repeat}', ['--steps','100','--eval-batches','0',*flags])
        launch('stall-profile',['--steps','80','--eval-batches','0','--delay-ms','30','--profile'])
        launch('stall-fixed-profile',['--steps','80','--eval-batches','0','--delay-ms','30',
                                   '--workers','2','--pin-memory','--nonblocking','--profile'])
    elif a.stage == 'numerical':
        launch('nan-loss',['--steps','12','--warmup','2','--eval-batches','0','--nan-step','5'],expected=('nonfinite_detected',))
        launch('nan-gradient',['--steps','12','--warmup','2','--eval-batches','0','--nan-gradient-step','5'],expected=('nonfinite_detected',))
        launch('finite-control',['--steps','30','--eval-batches','1'])
    elif a.stage == 'compile':
        launch('compiled-fp32',['--compile','--save-checkpoint'],timeout=900)
        for repeat in (2,3):
            launch(f'compiled-fp32-cached-r{repeat}',['--compile','--save-checkpoint'],timeout=900)


if __name__ == '__main__':
    main()
