from benchmarks.models import get_model
from benchmarks.policy import act, load_policy, make_observation

bundle = load_policy(get_model("smolvla"))
print("device:", bundle["device"])
print("dtype:", next(bundle["policy"].parameters()).dtype)

action = act(bundle, make_observation(bundle))
print("action shape:", tuple(action.shape))
