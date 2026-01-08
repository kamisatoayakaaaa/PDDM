import os
import torch

src = "distill5step/distill_epoch_010jy.pth"
dst = "distill5step/distill_epoch_010jy_pure.pth"

ckpt = torch.load(src, map_location="cpu")

if not isinstance(ckpt, dict):
    raise TypeError(f"Unexpected ckpt type: {type(ckpt)}")

if "student" not in ckpt:
    raise KeyError(f"'student' not found. keys={list(ckpt.keys())[:50]}")

state = ckpt["student"]

if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
    state = state["state_dict"]

if not isinstance(state, dict):
    raise TypeError(f"Unexpected student type: {type(state)}")

def strip_prefix(sd, prefix="module."):
    if not any(k.startswith(prefix) for k in sd.keys()):
        return sd
    return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in sd.items()}

state = strip_prefix(state, "module.")

os.makedirs(os.path.dirname(dst), exist_ok=True)
torch.save(state, dst)

print("Saved:", dst)
print("Num keys:", len(state))
print("Example keys:", list(state.keys())[:10])
