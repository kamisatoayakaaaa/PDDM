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


def main():
    ckpt = Path(r"PDDM\DiffKD\diffkd\pure_model\train2\best.pth")
    out_txt = ckpt.parent / "teacher_params.txt"

    try:
        raw = torch.load(str(ckpt), map_location="cpu", weights_only=True)
    except TypeError:
        raw = torch.load(str(ckpt), map_location="cpu")

    sd = unwrap(raw)

    if isinstance(sd, torch.nn.Module):
        items = list(sd.state_dict().items())
    elif isinstance(sd, dict):
        items = list(sd.items())
    else:
        raise TypeError(type(sd))

    items.sort(key=lambda x: x[0])

    total_numel = 0
    for _, v in items:
        if torch.is_tensor(v):
            total_numel += v.numel()

    prefixes = {}
    for k, v in items:
        p = k.split(".")[0]
        prefixes[p] = prefixes.get(p, 0) + (v.numel() if torch.is_tensor(v) else 0)

    top_prefixes = sorted(prefixes.items(), key=lambda x: x[1], reverse=True)[:20]

    lines = []
    lines.append(f"ckpt: {ckpt.resolve()}")
    lines.append(f"tensors: {sum(1 for _, v in items if torch.is_tensor(v))}")
    lines.append(f"total_numel: {total_numel}")
    lines.append("top_prefixes(numel):")
    for p, n in top_prefixes:
        lines.append(f"  {p}: {n}")

    lines.append("")
    lines.append("name\tshape\tdtype\tnumel")
    for k, v in items:
        if torch.is_tensor(v):
            lines.append(f"{k}\t{tuple(v.shape)}\t{v.dtype}\t{v.numel()}")
        else:
            lines.append(f"{k}\t<non-tensor>\t{type(v)}\t-")

    out_txt.write_text("\n".join(lines), encoding="utf-8")

    print("saved:", out_txt.resolve())
    print("summary:")
    print("\n".join(lines[:30]))
    print("\nfirst 50 params:")
    for i, (k, v) in enumerate(items[:50]):
        if torch.is_tensor(v):
            print(i, k, tuple(v.shape), v.dtype, v.numel())
        else:
            print(i, k, "<non-tensor>", type(v))


if __name__ == "__main__":
    main()
