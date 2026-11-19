import io

import torch

from benchmarks.models import get_model
from benchmarks.policy import act, free_memory, load_policy, make_observation
from benchmarks.techniques import get_technique

TECHNIQUES = ["native", "fp32_cast", "int8_weight"]  # CPU-safe ones


def real_size_gb(policy) -> float:
    """Real bytes of the weights, measured by saving them."""
    buf = io.BytesIO()
    torch.save(policy.state_dict(), buf)
    return buf.tell() / 1e9


def quantized_linears(policy) -> str:
    linears = [m for m in policy.modules() if isinstance(m, torch.nn.Linear)]
    quant = [
        m for m in linears if type(m.weight).__name__ not in ("Parameter", "Tensor")
    ]
    return f"{len(quant)}/{len(linears)}"


spec = get_model("smolvla")
obs = None
actions = {}

print(f"{'technique':16s} {'size':>8s} {'quantized':>10s} {'diff vs native':>15s}")
for name in TECHNIQUES:
    bundle = load_policy(spec, get_technique(name))
    if obs is None:
        torch.manual_seed(0)
        obs = make_observation(bundle)  # one fixed input for all techniques

    torch.manual_seed(0)  # same denoising noise for every technique
    actions[name] = act(bundle, obs)

    diff = (actions[name] - actions["native"]).abs().max().item()
    print(
        f"{name:16s} {real_size_gb(bundle['policy']):7.2f}G "
        f"{quantized_linears(bundle['policy']):>10s} {diff:15.5f}"
    )

    del bundle
    free_memory()
