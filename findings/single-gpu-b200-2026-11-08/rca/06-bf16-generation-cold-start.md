# BF16 generation cold-start cost in cuDNN attention

## Initial symptom

Three 1,300-update BF16 jobs take approximately 48 seconds each, versus roughly
26 seconds for FP32, despite similar steady training throughput. Phase timing
localizes the difference to generation: approximately 25 seconds of generation
per BF16 job, versus four seconds in FP32. This alone is not evidence that BF16
arithmetic or every BF16 generation call is persistently slow.

## Separate cold from warm

`generation_check.py` loads the same trained FP32 parameter checkpoint for both
precision paths, warms up eight generated tokens, and times three 80-token
generations. The first BF16 80-token generation takes **18.994 seconds**; its
next calls take **0.1219 and 0.1213 seconds**. FP32's median is **0.0909 seconds**.
Warming up eight tokens did not warm all the longer sequence shapes used later
in this model's autoregressive generation loop.

The warm trace shows different attention operators: FP32 uses efficient
attention, while automatic BF16 selection uses cuDNN attention. A warm trace
alone cannot explain the earlier cold call, so a fresh-process paired experiment
was run next.

## Paired backend experiment

Both fresh processes use the same checkpoint, BF16 autocast, prompt, eight-token
warmup, and an 80-token profiled generation. One allows default SDPA backend
selection; the other excludes cuDNN through `sdpa_kernel` and selects Flash
Attention on this system.

| Measurement | Automatic selection | cuDNN excluded |
|---|---:|---:|
| Cold profiled 80-token call | 19.970 s | 1.577 s |
| CPU total within `aten::_cudnn_attention_forward` | 18.592 s | 0 |
| Warm 80-token median, unprofiled | 0.1205 s | 0.1154 s |

The cold trace records 480 attention calls (six blocks × 80 generated tokens).
Excluding cuDNN removes nearly all the long cold CPU cost. Warm generation
changes little. This localizes the regression to cold work inside the cuDNN
attention path, consistent with setup/planning for varying sequence shapes.
The profiler does not separately expose every internal cuDNN planning operation;
the narrower, directly supported finding is the expensive cold cuDNN call path.

Cold paired times include profiler overhead; they are not serving benchmarks
or directly comparable to the unprofiled cold-call latency above. The backend
restriction is confined to the experiment's context and process; no system-wide
backend configuration was changed.

## Implication

Report cold and warm latency separately. For this model/environment, disabling
cuDNN attention is a useful candidate for avoiding expensive cold generation;
it is not a universal recommendation for B200 workloads. Different backends may
change floating-point results and performance at other shapes. No inference
service, KV cache, concurrent-request batching, or serving SLO was tested here.

Artifacts: `generation-check/`, `generation-backend-auto/`, and
`generation-backend-no-cudnn/` under `runs/single-gpu-b200-2026-11-08/`.
Both traces and top-operator tables are retained. The selector API is documented
in [PyTorch's SDPA backend reference](https://docs.pytorch.org/docs/stable/generated/torch.nn.attention.sdpa_kernel.html).
