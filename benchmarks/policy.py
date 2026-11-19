import gc
import importlib

import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors

from .models import ModelSpec
from .techniques import (
    Technique,
    autocast_context,
    cast_weights,
    check_compile,
    quantize,
)


def pick_device() -> str:
    """Use the NVIDIA GPU if there is one, otherwise the CPU."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_policy(
    spec: ModelSpec, technique: Technique, device: str | None = None
) -> dict:
    device = device or pick_device()

    module = importlib.import_module(spec.policy_module)
    policy_cls = getattr(module, spec.policy_class)

    config = PreTrainedConfig.from_pretrained(spec.policy_path)
    config.device = device
    config.compile_model = technique.compile_model
    if technique.compile_model:
        config.compile_mode = technique.compile_mode
    if spec.n_action_steps is not None:
        config.n_action_steps = spec.n_action_steps

    policy = policy_cls.from_pretrained(spec.policy_path, config=config)
    check_compile(policy, technique)
    policy.to(device)
    if technique.cast_dtype is not None:
        cast_weights(policy, technique.cast_dtype)
    quantize(policy, technique)
    policy.eval()

    pre, post = make_pre_post_processors(
        policy.config,
        pretrained_path=spec.policy_path,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    return {
        "policy": policy,
        "pre": pre,
        "post": post,
        "spec": spec,
        "technique": technique,
        "device": device,
    }


def make_observation(bundle: dict) -> dict:
    """Random inputs with exactly the shapes the model expects (for speed tests)."""
    obs = {}
    for key, feature in bundle["policy"].config.input_features.items():
        if "empty_camera" in key:
            continue  # unused camera slots: the model pads these itself
        kind = str(feature.type)
        if "VISUAL" in kind or "STATE" in kind:
            obs[key] = torch.rand(1, *feature.shape, device=bundle["device"])
    obs["task"] = [bundle["spec"].task]
    return obs


def act(bundle: dict, obs: dict) -> torch.Tensor:
    with torch.no_grad(), autocast_context(bundle["technique"], bundle["device"]):
        bundle["policy"].reset()
        batch = bundle["pre"](obs)
        action = bundle["policy"].select_action(batch)
        return bundle["post"](action)


def free_memory() -> None:
    """Release a dropped model's memory before loading the next one."""
    torch._dynamo.reset()  # torch.compile caches keep old models alive
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
