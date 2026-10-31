"""Environment and observability checks only; never execute model training."""
import hashlib
import io
import json
import platform
from pathlib import Path

import numpy as np
import torch
import data
import train
from model import TinyLM

root = Path(__file__).resolve().parent
for filename in ("train.py", "model.py", "data.py"):
    compile((root / filename).read_text(), filename, "exec")
assert torch.cuda.is_available(), "CUDA is unavailable"
assert torch.cuda.get_device_capability(0) == (10, 0), "Expected B200 capability"
assert "sm_100" in torch.cuda.get_arch_list(), "PyTorch build lacks native B200 support"

data.download_and_tokenize()
tokens = np.memmap(data.TRAIN_BIN, dtype=np.uint16, mode="r")
dataset = data.LMDataset("train", seq_length=128)
x, y = dataset[0]
assert torch.equal(x[1:], y[:-1]), "Training targets are not shifted by one token"

# CPU construction and controlled logger checks; no forward/backward/update.
train.seed_everything(42)
model = TinyLM()
baseline = train.sampled_weights(model)
initial_values = {name: sample["values"][:] for name, sample in baseline.items()}
stream = io.StringIO()
train.log_weights(stream, model, 0, "initial", baseline)
with torch.no_grad():
    model.tok_emb.weight.view(-1)[0].add_(0.001)
train.log_weights(stream, model, 1, "updated", baseline)
records = [json.loads(line) for line in stream.getvalue().splitlines()]
assert len(records) == 2
assert all(sample["delta_from_start"] == [0.0] * 5
           for sample in records[0]["weights"].values())
assert records[1]["weights"]["tok_emb.weight"]["delta_from_start"][0] > 0
assert all(baseline[name]["values"] == values for name, values in initial_values.items())
with train.make_profiler(False, torch.device("cpu"), root / "preflight") as disabled:
    assert disabled is None

# Validate trace/summary export using five tiny CPU additions, not the model.
profile_root = root / "preflight"
with train.make_profiler(True, torch.device("cpu"), profile_root) as profiler:
    for _ in range(5):
        with train.profile_region("preflight/cpu_add", True):
            torch.ones(4) + 1
        profiler.step()
traces = list((profile_root / "profiler").glob("trace-*.json"))
summaries = list((profile_root / "profiler").glob("trace-*.txt"))
assert traces and summaries, "Profiler outputs missing"
trace = json.loads(traces[-1].read_text())
assert any(event.get("name") == "preflight/cpu_add" for event in trace["traceEvents"])

report = {
    "python": platform.python_version(),
    "torch": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "gpu_count": torch.cuda.device_count(),
    "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    "compiled_architectures": torch.cuda.get_arch_list(),
    "bf16_supported": torch.cuda.is_bf16_supported(),
    "cuda_profiler_supported": torch.profiler.ProfilerActivity.CUDA in torch.profiler.supported_activities(),
    "model_parameter_count": sum(p.numel() for p in model.parameters()),
    "initial_weight_samples": baseline,
    "training_token_count": len(tokens),
    "validation_token_count": len(np.memmap(data.VAL_BIN, dtype=np.uint16, mode="r")),
    "data_sha256": hashlib.sha256(Path(data.INPUT_FILE).read_bytes()).hexdigest(),
    "source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                      for name in ("train.py", "model.py", "data.py", "README.md")},
    "checks": ["syntax", "CUDA visibility and B200 architecture", "data preparation and shifted targets",
               "initial weight sampling and controlled delta", "disabled profiler", "CPU trace and summary export"],
    "model_forward_passes": 0,
    "optimizer_steps": 0,
    "training_started": False,
    "cuda_kernel_execution_tested": False,
}
(profile_root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
