from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Technique:
    name: str
    cast_dtype: torch.dtype | None = None  # convert the weights; None = as shipped
    autocast_dtype: torch.dtype | None = None  # run the math in this dtype; None = off
    compile_mode: str | None = None  # torch.compile mode; None = off
    cuda_graphs: bool = False  # record and replay the kernels
    quant: str | None = None  # "int8" / "int4"; None = off

    @property
    def compile_model(self) -> bool:
        return self.compile_mode is not None


TECHNIQUES: dict[str, Technique] = {
    "native": Technique("native"),
    "fp32_cast": Technique("fp32_cast", cast_dtype=torch.float32),
}


def get_technique(name: str) -> Technique:
    if name not in TECHNIQUES:
        raise KeyError(f"unknown technique {name!r}; choose from {sorted(TECHNIQUES)}")
    return TECHNIQUES[name]


def cast_weights(policy, dtype: torch.dtype) -> None:
    policy.to(dtype)

    expert = getattr(getattr(policy, "model", None), "paligemma_with_expert", None)
    if expert is not None:
        expert.precision = dtype

    got = {p.dtype for p in policy.parameters() if p.is_floating_point()}
    if got != {dtype}:
        raise RuntimeError(f"cast failed: wanted {dtype}, got {got}")
