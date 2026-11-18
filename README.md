# Does a Faster VLA Still Work?
Certifying Compilation and Quantization Across GPUs

[Paper (arXiv)] · [Leaderboard] · [Results CSVs]

![Figure 1](figures/fig1.png)

## TL;DR
Speed-ups can silently change robot task success.
We measure speed and certify success with paired tests.

## Results
| Model | Technique | GPU | Speed-up | Success | Certified |
|---|---|---|---|---|---|

## Install
uv sync

## Reproduce the paper
make speed     # ~10 min, ~$2
make success   # ~7 GPU-hours, ~$15
make figures

## Add your model
See docs/add_model.md

## Citation
@article{basith2026faster, ...}