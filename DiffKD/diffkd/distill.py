import argparse
import copy
import os
import re
import math
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from diffusers import DDPMScheduler
from tqdm import tqdm

from diffkd import DiffKD
from timestepconfig import DistillConfig
from models.diffusion import Model


HERE = Path(__file__).resolve().parent


def load_paths():
    import yaml
    p = HERE / "paths.yaml"
    return (yaml.safe_load(p.read_text(encoding="utf-8")) or {}) if p.exists() else {}


PATHS = load_paths()


def _ns(d):
    return SimpleNamespace(**{k: _ns(v) if isinstance(v, dict) else v for k, v in d.items()})


def to_minus_one_to_one(x):
    return x * 2 - 1


def unwrap_state_dict(raw):
    sd = raw
    if isinstance(sd, dict):
        for k in ("state_dict", "model", "unet", "net", "ema", "student", "teacher"):
            if k in sd and isinstance(sd[k], dict):
                sd = sd[k]
                break
    if isinstance(sd, dict) and any(k.startswith("module.") for k in sd.keys()):
        sd = {k[7:]: v for k, v in sd.items()}
    return sd


def build_loader(data_root, batch_size, num_workers, download=False):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(to_minus_one_to_one),
    ])
    dataset = datasets.CIFAR10(
        root=data_root,
        train=True,
        download=download,
        transform=transform,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )


def infer_cfg_from_sd(sd, image_size, num_train_timesteps):
    conv_in_w = sd["conv_in.weight"]
    conv_out_w = sd["conv_out.weight"]

    ch = int(conv_in_w.shape[0])
    in_channels = int(conv_in_w.shape[1])
    out_ch = int(conv_out_w.shape[0])

    down_pat = re.compile(r"^down\.(\d+)\.block\.(\d+)\.conv1\.weight$")
    attn_pat = re.compile(r"^down\.(\d+)\.attn\.(\d+)\.q\.weight$")

    levels = {}
    for k, v in sd.items():
        m = down_pat.match(k)
        if m:
            i = int(m.group(1))
            b = int(m.group(2))
            levels.setdefault(i, {"blocks": set(), "out0": None})
            levels[i]["blocks"].add(b)
            if b == 0:
                levels[i]["out0"] = int(v.shape[0])

    if not levels:
        raise ValueError("cannot infer cfg from state_dict (missing down.{i}.block.{j}.conv1.weight)")

    num_levels = len(levels)
    ch_mult = []
    num_res_blocks = None
    for i in range(num_levels):
        info = levels[i]
        if info["out0"] is None:
            raise ValueError(f"missing down.{i}.block.0.conv1.weight")
        ch_mult.append(int(info["out0"] // ch))
        nb = max(info["blocks"]) + 1
        num_res_blocks = nb if num_res_blocks is None else num_res_blocks

    attn_levels = set()
    for k in sd.keys():
        m = attn_pat.match(k)
        if m:
            attn_levels.add(int(m.group(1)))
    attn_resolutions = sorted({int(image_size // (2 ** i)) for i in attn_levels})

    resamp_with_conv = any(k.endswith("downsample.conv.weight") for k in sd.keys())

    cfg = {
        "model": {
            "type": "simple",
            "ch": ch,
            "out_ch": out_ch,
            "ch_mult": ch_mult,
            "num_res_blocks": int(num_res_blocks or 2),
            "attn_resolutions": attn_resolutions,
            "dropout": 0.0,
            "in_channels": in_channels,
            "resamp_with_conv": bool(resamp_with_conv),
        },
        "data": {"image_size": int(image_size)},
        "diffusion": {"num_diffusion_timesteps": int(num_train_timesteps)},
    }
    return _ns(cfg)


def load_sd(pth):
    try:
        raw = torch.load(str(pth), map_location="cpu", weights_only=True)
    except TypeError:
        raw = torch.load(str(pth), map_location="cpu")
    sd = unwrap_state_dict(raw)
    if not isinstance(sd, dict):
        raise TypeError(f"ckpt is not a state_dict dict: {type(sd)}")
    return sd


def build_model_from_ckpt(pth, cfg, device, strict=True):
    sd = load_sd(pth)
    model = Model(cfg).to(device)
    model.load_state_dict(sd, strict=strict)
    return model


def build_cfg(stage, lambda_trans, lambda_score, lr):
    if stage == 1:
        ts = [500]
    elif stage == 2:
        ts = [470, 485, 500, 515, 530]
    else:
        ts = []

    cfg = DistillConfig(
        distill_timesteps=ts,
        lambda_trans=lambda_trans,
        lambda_score=lambda_score,
    )
    if lr is not None:
        try:
            cfg.lr = lr
        except Exception:
            pass
    if not hasattr(cfg, "lr") or cfg.lr is None:
        cfg.lr = 1e-4
    return cfg


def make_parser():
    p = argparse.ArgumentParser()

    p.add_argument("--teacher_ckpt", type=str, default=PATHS.get("teacher_ckpt"))
    p.add_argument("--student_ckpt", type=str, default=PATHS.get("student_ckpt"))

    p.add_argument("--data_root", type=str, default=PATHS.get("data_root"))
    p.add_argument(
        "--download",
        action=argparse.BooleanOptionalAction,
        default=bool(PATHS.get("download", False)),
    )
    p.add_argument("--save_dir", type=str, default=PATHS.get("save_dir"))
    p.add_argument("--device", type=str, default=PATHS.get("device"))

    p.add_argument("--image_size", type=int, default=int(PATHS.get("image_size", 32)))
    p.add_argument("--num_train_timesteps", type=int, default=int(PATHS.get("num_train_timesteps", 1000)))

    p.add_argument("--epochs", type=int, default=int(PATHS.get("epochs", 1)))
    p.add_argument("--batch_size", type=int, default=int(PATHS.get("batch_size", 128)))
    p.add_argument("--num_workers", type=int, default=int(PATHS.get("num_workers", 4)))

    p.add_argument("--stage", type=int, choices=[1, 2, 3], default=int(PATHS.get("stage", 1)))
    p.add_argument("--lambda_trans", type=float, default=float(PATHS.get("lambda_trans", 1.0)))
    p.add_argument("--lambda_score", type=float, default=float(PATHS.get("lambda_score", 1.0)))
    p.add_argument("--lr", type=float, default=PATHS.get("lr", None))

    p.add_argument("--freeze_student", action="store_true", default=bool(PATHS.get("freeze_student", False)))
    return p


def main():
    args = make_parser().parse_args()

    if args.teacher_ckpt is None:
        raise ValueError("teacher_ckpt is required (paths.yaml or --teacher_ckpt)")
    if args.student_ckpt is None:
        raise ValueError("student_ckpt is required (paths.yaml or --student_ckpt)")
    if args.data_root is None:
        raise ValueError("data_root is required (paths.yaml or --data_root)")
    if args.save_dir is None:
        args.save_dir = "distill_ckpts"

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)

    scheduler = DDPMScheduler(num_train_timesteps=int(args.num_train_timesteps))

    teacher_sd = load_sd(args.teacher_ckpt)
    model_cfg = infer_cfg_from_sd(
        teacher_sd,
        image_size=int(args.image_size),
        num_train_timesteps=int(args.num_train_timesteps),
    )

    teacher_unet = Model(model_cfg).to(device)
    teacher_unet.load_state_dict(teacher_sd, strict=True)
    teacher_unet.eval()
    for p in teacher_unet.parameters():
        p.requires_grad_(False)

    student_sd = load_sd(args.student_ckpt)
    student = Model(model_cfg).to(device)
    student.load_state_dict(student_sd, strict=True)

    loader = build_loader(args.data_root, args.batch_size, args.num_workers, download=bool(args.download))
    cfg = build_cfg(args.stage, args.lambda_trans, args.lambda_score, args.lr)

    dummy_x = torch.randn(1, 3, int(args.image_size), int(args.image_size), device=device)
    dummy_t = torch.zeros(1, device=device, dtype=torch.long)
    with torch.no_grad():
        t_feat = teacher_unet(dummy_x, dummy_t)
        s_feat = student(dummy_x, dummy_t)

    student_channels = int(s_feat.shape[1])
    teacher_channels = int(t_feat.shape[1])

    distiller = DiffKD(
        student_channels=student_channels,
        teacher_channels=teacher_channels,
        kernel_size=3,
        inference_steps=5,
        num_train_timesteps=int(args.num_train_timesteps),
        use_ae=False,
        ae_channels=None,
        cfg=cfg,
    ).to(device)

    if args.freeze_student:
        for p in student.parameters():
            p.requires_grad_(False)
        student.eval()
    else:
        student.train()

    params = list(distiller.parameters())
    if not args.freeze_student:
        params += list(student.parameters())
    optimizer = torch.optim.Adam(params, lr=float(cfg.lr))

    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        distiller.train()
        student.eval() if args.freeze_student else student.train()

        total_loss = 0.0
        total_trans = 0.0
        total_ddim = 0.0
        total_rec = 0.0
        steps = 0

        pbar = tqdm(loader, desc=f"epoch {epoch}/{args.epochs}", dynamic_ncols=True)
        for x, _ in pbar:
            x = x.to(device)
            b = x.size(0)

            t = torch.randint(0, int(args.num_train_timesteps), (b,), device=device, dtype=torch.long)
            noise = torch.randn_like(x)
            xt = scheduler.add_noise(x, noise, t)

            with torch.no_grad():
                teacher_feat = teacher_unet(xt, t)

            if args.freeze_student:
                with torch.no_grad():
                    student_feat = student(xt, t)
            else:
                student_feat = student(xt, t)

            refined, t_feat_used, ddim_loss, rec_loss = distiller(student_feat, teacher_feat)

            trans_loss = F.mse_loss(refined, t_feat_used.detach())
            loss = cfg.lambda_trans * trans_loss + cfg.lambda_score * ddim_loss

            rec_val = 0.0
            if rec_loss is not None:
                loss = loss + rec_loss
                rec_val = float(rec_loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_val = float(loss.item())
            trans_val = float(trans_loss.item())
            ddim_val = float(ddim_loss.item())

            total_loss += loss_val
            total_trans += trans_val
            total_ddim += ddim_val
            total_rec += rec_val
            steps += 1

            pbar.set_postfix(loss=f"{loss_val:.4f}", trans=f"{trans_val:.4f}", ddim=f"{ddim_val:.4f}")

        mean_loss = total_loss / max(1, steps)
        mean_trans = total_trans / max(1, steps)
        mean_ddim = total_ddim / max(1, steps)
        mean_rec = total_rec / max(1, steps)

        print(f"epoch {epoch} loss={mean_loss:.4f} trans={mean_trans:.4f} ddim={mean_ddim:.4f} rec={mean_rec:.4f}")

        ckpt = {
            "epoch": epoch,
            "student_frozen": bool(args.freeze_student),
            "student": student.state_dict(),
            "distiller": distiller.state_dict(),
            "cfg": (cfg.__dict__ if hasattr(cfg, "__dict__") else str(cfg)),
            "num_train_timesteps": int(args.num_train_timesteps),
            "teacher_ckpt": str(Path(args.teacher_ckpt).resolve()),
            "student_ckpt": str(Path(args.student_ckpt).resolve()),
        }
        torch.save(ckpt, os.path.join(args.save_dir, f"distill_epoch_{epoch:03d}.pth"))


if __name__ == "__main__":
    main()
