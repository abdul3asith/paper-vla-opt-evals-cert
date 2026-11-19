import torch

from benchmarks.models import get_model
from benchmarks.policy import act, free_memory, load_policy, make_observation
from benchmarks.techniques import get_technique

spec = get_model("smolvla")
torch.manual_seed(0)
obs = None
actions = {}

for name in ["native", "fp32_cast"]:
    b = load_policy(spec, get_technique(name))
    if obs is None:
        obs = make_observation(b)
    p = next(b["policy"].parameters())
    size_gb = sum(x.numel() * x.element_size() for x in b["policy"].parameters()) / 1e9
    torch.manual_seed(0)  # same denoising noise for both
    actions[name] = act(b, obs)
    print(f"{name:10s} dtype={p.dtype}  weights={size_gb:.2f} GB")
    del b
    free_memory()

diff = (actions["native"] - actions["fp32_cast"]).abs().max().item()
print(f"max action difference: {diff:.5f}")
