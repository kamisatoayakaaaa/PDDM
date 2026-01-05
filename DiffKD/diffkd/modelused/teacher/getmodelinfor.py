import re
from pathlib import Path
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


def save_yaml(path, data):
    try:
        import yaml
        Path(path).write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    except Exception:
        def dump_obj(o, indent=0):
            sp = "  " * indent
            if isinstance(o, dict):
                s = ""
                for k, v in o.items():
                    if isinstance(v, (dict, list)):
                        s += f"{sp}{k}:\n{dump_obj(v, indent+1)}"
                    else:
                        s += f"{sp}{k}: {v}\n"
                return s
            if isinstance(o, list):
                s = ""
                for v in o:
                    if isinstance(v, (dict, list)):
                        s += f"{sp}-\n{dump_obj(v, indent+1)}"
                    else:
                        s += f"{sp}- {v}\n"
                return s
            return f"{sp}{o}\n"
        Path(path).write_text(dump_obj(data), encoding="utf-8")


def find_stage_ids(keys, prefix):
    ids = set()
    pat = re.compile(rf"^{re.escape(prefix)}\.(\d+)\.")
    for k in keys:
        m = pat.match(k)
        if m:
            ids.add(int(m.group(1)))
    return sorted(ids)


def stage_info(sd, prefix, i):
    keys = sd.keys()
    pat_block = re.compile(rf"^{re.escape(prefix)}\.{i}\.block\.(\d+)\.")
    pat_attn = re.compile(rf"^{re.escape(prefix)}\.{i}\.attn\.(\d+)\.")
    block_ids = set()
    attn_ids = set()
    for k in keys:
        m1 = pat_block.match(k)
        if m1:
            block_ids.add(int(m1.group(1)))
        m2 = pat_attn.match(k)
        if m2:
            attn_ids.add(int(m2.group(1)))

    ch = None
    for j in sorted(block_ids):
        w = sd.get(f"{prefix}.{i}.block.{j}.conv1.weight", None)
        if torch.is_tensor(w):
            ch = int(w.shape[0])
            break
    if ch is None:
        for j in sorted(attn_ids):
            w = sd.get(f"{prefix}.{i}.attn.{j}.q.weight", None)
            if torch.is_tensor(w):
                ch = int(w.shape[0])
                break

    has_downsample = any(k.startswith(f"{prefix}.{i}.downsample.") for k in keys)
    has_upsample = any(k.startswith(f"{prefix}.{i}.upsample.") for k in keys)

    return {
        "channels": ch,
        "num_blocks": (max(block_ids) + 1) if block_ids else 0,
        "num_attn": (max(attn_ids) + 1) if attn_ids else 0,
        "has_downsample": bool(has_downsample),
        "has_upsample": bool(has_upsample),
    }


def main():
    ckpt = Path(r"PDDM\DiffKD\diffkd\google\diffusion_pytorch_model_dmih_fix1x1_v2.pth").resolve()
    out_yaml = ckpt.parent / "teacher_struct.yaml"

    try:
        raw = torch.load(str(ckpt), map_location="cpu", weights_only=True)
    except TypeError:
        raw = torch.load(str(ckpt), map_location="cpu")

    sd = unwrap(raw)
    if isinstance(sd, torch.nn.Module):
        sd = sd.state_dict()
    if not isinstance(sd, dict):
        raise TypeError(type(sd))

    conv_in = sd.get("conv_in.weight", None)
    conv_out = sd.get("conv_out.weight", None)

    in_channels = int(conv_in.shape[1]) if torch.is_tensor(conv_in) else None
    base_channels = int(conv_in.shape[0]) if torch.is_tensor(conv_in) else None
    out_channels = int(conv_out.shape[0]) if torch.is_tensor(conv_out) else None

    temb_dim = None
    for k, v in sd.items():
        if k.endswith(".temb_proj.weight") and torch.is_tensor(v):
            temb_dim = int(v.shape[1])
            break

    keys = sd.keys()
    down_ids = find_stage_ids(keys, "down")
    up_ids = find_stage_ids(keys, "up")
    mid_like = any(k.startswith("mid.") or k.startswith("middle.") for k in keys)

    down = [stage_info(sd, "down", i) for i in down_ids]
    up = [stage_info(sd, "up", i) for i in up_ids]

    info = {
        "ckpt": str(ckpt),
        "in_channels": in_channels,
        "out_channels": out_channels,
        "base_channels": base_channels,
        "time_emb_dim": temb_dim,
        "num_down_stages": len(down_ids),
        "num_up_stages": len(up_ids),
        "has_mid": bool(mid_like),
        "down": down,
        "up": up,
    }

    save_yaml(out_yaml, info)

    print("saved yaml:", out_yaml)
    print("summary:", {k: info[k] for k in ["in_channels", "out_channels", "base_channels", "time_emb_dim", "num_down_stages", "num_up_stages", "has_mid"]})
    if down:
        print("down[0]:", down[0])
    if len(down) > 1:
        print("down[1]:", down[1])


if __name__ == "__main__":
    main()
