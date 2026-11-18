import gc
import importlib

import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors

from .models import ModelSpec


def pick_device() -> str:
    """Use the NVIDIA GPU if there is one, otherwise the CPU."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_policy(spec: ModelSpec, device: str | None = None) -> dict:
    """Load the model, its settings and its input/output processors."""
    device = device or pick_device()

    # 1. Find the policy class from its text name.
    module = importlib.import_module(spec.policy_module)
    policy_cls = getattr(module, spec.policy_class)

    # 2. Load the settings, then apply our overrides.
    config = PreTrainedConfig.from_pretrained(spec.policy_path)
    config.device = device  # NEW: don't let it default to cuda or mps
    if spec.n_action_steps is not None:
        config.n_action_steps = spec.n_action_steps

    # 3. Load the trained weights and move them to the device.
    policy = policy_cls.from_pretrained(spec.policy_path, config=config)
    policy.to(device)
    policy.eval()

    # 4. Load the processors that normalize inputs and un-normalize outputs.
    pre, post = make_pre_post_processors(
        policy.config,
        pretrained_path=spec.policy_path,
        preprocessor_overrides={"device_processor": {"device": device}},
    )

    return {"policy": policy, "pre": pre, "post": post, "spec": spec, "device": device}


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
    """One full step: preprocess -> model -> postprocess."""
    with torch.no_grad():
        bundle["policy"].reset()  # empty the action queue, so this is a real model run
        batch = bundle["pre"](obs)
        action = bundle["policy"].select_action(batch)
        return bundle["post"](action)


def free_memory() -> None:
    """Release a dropped model's memory before loading the next one."""
    torch._dynamo.reset()  # torch.compile caches keep old models alive
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
