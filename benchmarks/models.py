from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    name: str  # short key, e.g. "smolvla"
    policy_path: str  # Hugging Face repo ID of the checkpoint
    policy_module: str  # Python module that holds the class
    policy_class: str  # class name inside that module
    task: str  # instruction text for speed tests
    suite: str = "libero_object"  # LIBERO task suite for success tests
    denoise_attr: str = "num_steps"  # config field holding the denoise step count
    n_action_steps: int | None = None  # override; None = keep the checkpoint's value


MODELS: dict[str, ModelSpec] = {
    "smolvla": ModelSpec(
        name="smolvla",
        policy_path="HuggingFaceVLA/smolvla_libero",
        policy_module="lerobot.policies.smolvla.modeling_smolvla",
        policy_class="SmolVLAPolicy",
        task="pick up the ketchup and place it in the basket",
    ),
    "pi05": ModelSpec(
        name="pi05",
        policy_path="lerobot/pi05_libero_finetuned_v044",
        policy_module="lerobot.policies.pi05.modeling_pi05",
        policy_class="PI05Policy",
        task="pick up the ketchup and place it in the basket",
        denoise_attr="num_inference_steps",
        n_action_steps=10,
    ),
}


def get_model(name: str) -> ModelSpec:
    if name not in MODELS:
        raise KeyError(f"unknown model {name!r}; choose from {sorted(MODELS)}")
    return MODELS[name]
