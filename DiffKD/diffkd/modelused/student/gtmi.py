from pathlib import Path
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

def _max_index(keys, prefix):
    r = re.compile(rf"^{re.escape(prefix)}\.(\d+)\.")
    mx = -1
    for k in keys:
        m = r.match(k)
        if m:
            mx = max(mx, int(m.group(1)))
    return mx

def _stage_block_max(keys, prefix, i):
    r = re.compile(rf"^{re.escape(prefix)}\.{i}\.block\.(\d+)\.")
    mx = -1
    for k in keys:
        m = r.match(k)
        if m:
            mx = max(mx, int(m.group(1)))
    return mx

def _stage_attn_max(keys, prefix, i):
    r = re.compile(rf"^{re.escape(prefix)}\.{i}\.attn\.(\d+)\.")
    mx = -1
    for k in keys:
        m = r.match(k)
        if m:
            mx = max(mx, int(m.group(1)))
    return mx

def _stage_channels(sd, prefix, i):
    for cand in (
        f"{prefix}.{i}.block.0.conv2.weight",
        f"{prefix}.{i}.block.0.conv1.weight",
    ):
        if cand in sd and torch.is_tensor(sd[cand]):
            return int(sd[cand].shape[0])
    r = re.compile(rf"^{re.escape(prefix)}\.{i}\.block\.\d+\.conv2\.weight$")
    for k, v in sd.items():
        if r.match(k) and torch.is_tensor(v):
            return int(v.shape[0])
    return None

def infer_struct(sd, ckpt_path: Path):
    keys = list(sd.keys())

    in_channels = None
    out_channels = None
    base_channels = None
    time_emb_dim = None

    if "conv_in.weight" in sd and torch.is_tensor(sd["conv_in.weight"]):
        base_channels = int(sd["conv_in.weight"].shape[0])
        in_channels = int(sd["conv_in.weight"].shape[1])

    if "conv_out.weight" in sd and torch.is_tensor(sd["conv_out.weight"]):
        out_channels = int(sd["conv_out.weight"].shape[0])

    if "temb.dense.0.weight" in sd and torch.is_tensor(sd["temb.dense.0.weight"]):
        time_emb_dim = int(sd["temb.dense.0.weight"].shape[0])

    has_mid = any(k.startswith("mid.") for k in keys)

    down_n = _max_index(keys, "down") + 1 if _max_index(keys, "down") >= 0 else 0
    up_n = _max_index(keys, "up") + 1 if _max_index(keys, "up") >= 0 else 0

    down = []
    for i in range(down_n):
        num_blocks = _stage_block_max(keys, "down", i) + 1 if _stage_block_max(keys, "down", i) >= 0 else 0
        num_attn = _stage_attn_max(keys, "down", i) + 1 if _stage_attn_max(keys, "down", i) >= 0 else 0
        down.append({
            "channels": _stage_channels(sd, "down", i),
            "num_blocks": int(num_blocks),
            "num_attn": int(num_attn),
            "has_downsample": any(k.startswith(f"down.{i}.downsample.") for k in keys),
            "has_upsample": False,
        })

    up = []
    for i in range(up_n):
        num_blocks = _stage_block_max(keys, "up", i) + 1 if _stage_block_max(keys, "up", i) >= 0 else 0
        num_attn = _stage_attn_max(keys, "up", i) + 1 if _stage_attn_max(keys, "up", i) >= 0 else 0
        up.append({
            "channels": _stage_channels(sd, "up", i),
            "num_blocks": int(num_blocks),
            "num_attn": int(num_attn),
            "has_downsample": False,
            "has_upsample": any(k.startswith(f"up.{i}.upsample.") for k in keys),
        })

    return {
        "ckpt": str(ckpt_path.resolve()),
        "in_channels": in_channels,
        "out_channels": out_channels,
        "base_channels": base_channels,
        "time_emb_dim": time_emb_dim,
        "num_down_stages": int(down_n),
        "num_up_stages": int(up_n),
        "has_mid": bool(has_mid),
        "down": down,
        "up": up,
    }

def save_yaml_like(obj, out_path: Path):
    try:
        import yaml
        out_path.write_text(
            yaml.safe_dump(obj, sort_keys=False, allow_unicode=True, default_flow_style=False),
            encoding="utf-8"
        )
    except Exception:
        out_path.with_suffix(".json").write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def main():
    ckpt = Path(r"C:\Users\17007\Desktop\PDDM\PDDM\DiffKD\diffkd\pure_model\train2\best.pth")
    out_yaml = ckpt.parent / "student_struct.yaml"

    try:
        raw = torch.load(str(ckpt), map_location="cpu", weights_only=True)
    except TypeError:
        raw = torch.load(str(ckpt), map_location="cpu")

    sd = unwrap(raw)
    if isinstance(sd, torch.nn.Module):
        sd = sd.state_dict()
    if not isinstance(sd, dict):
        raise TypeError(type(sd))

    y = infer_struct(sd, ckpt)
    save_yaml_like(y, out_yaml)
    print("saved:", out_yaml.resolve())

if __name__ == "__main__":
    main()
