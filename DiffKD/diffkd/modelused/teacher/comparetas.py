from pathlib import Path
from collections import Counter, defaultdict
import re
import json
import torch

def unwrap(raw):
    sd = raw
    if isinstance(sd, dict):
        for k in ("state_dict", "model", "unet", "net", "ema"):
            if k in sd and isinstance(sd[k], dict):
                sd = sd[k]
                break
    if isinstance(sd, dict) and any(k.startswith("module.") for k in sd.keys()):
        sd = {k[7:]: v for k, v in sd.items()}
    return sd

def load_sd(p):
    try:
        raw = torch.load(str(p), map_location="cpu", weights_only=True)
    except TypeError:
        raw = torch.load(str(p), map_location="cpu")
    sd = unwrap(raw)
    if isinstance(sd, torch.nn.Module):
        sd = sd.state_dict()
    if not isinstance(sd, dict):
        raise TypeError(type(sd))
    return sd

def signature(sd):
    c = Counter()
    for k, v in sd.items():
        if torch.is_tensor(v):
            c[(tuple(v.shape), str(v.dtype))] += 1
    return c

def key_shape_map(sd):
    m = {}
    for k, v in sd.items():
        if torch.is_tensor(v):
            m[k] = (tuple(v.shape), str(v.dtype))
    return m

def main():
    teacher_ckpt = Path(r"C:\Users\17007\Desktop\PDDM\PDDM\DiffKD\diffkd\google\diffusion_pytorch_model_dmih_fix1x1_v2.pth")
    student_ckpt = Path(r"C:\Users\17007\Desktop\PDDM\PDDM\DiffKD\diffkd\pure_model\train2\best.pth")

    t_sd = load_sd(teacher_ckpt)
    s_sd = load_sd(student_ckpt)

    t_sig = signature(t_sd)
    s_sig = signature(s_sd)

    print("teacher tensors:", sum(1 for v in t_sd.values() if torch.is_tensor(v)))
    print("student tensors:", sum(1 for v in s_sd.values() if torch.is_tensor(v)))
    print("teacher unique shapes:", len(t_sig))
    print("student unique shapes:", len(s_sig))
    print("signature_equal:", t_sig == s_sig)

    if t_sig != s_sig:
        only_t = t_sig - s_sig
        only_s = s_sig - t_sig
        print("\n-- shapes only in teacher (up to 30) --")
        for i, (k, v) in enumerate(only_t.items()):
            if i >= 30: break
            print(k, v)
        print("\n-- shapes only in student (up to 30) --")
        for i, (k, v) in enumerate(only_s.items()):
            if i >= 30: break
            print(k, v)

    t_km = key_shape_map(t_sd)
    s_km = key_shape_map(s_sd)
    common = set(t_km.keys()) & set(s_km.keys())
    print("\ncommon keys:", len(common))
    if len(common) > 0:
        mism = []
        for k in common:
            if t_km[k] != s_km[k]:
                mism.append((k, t_km[k], s_km[k]))
        print("per-key shape mismatch in common keys:", len(mism))
        for k, a, b in mism[:40]:
            print(k, "teacher", a, "student", b)

if __name__ == "__main__":
    main()
