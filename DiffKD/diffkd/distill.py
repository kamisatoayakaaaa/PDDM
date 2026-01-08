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
from torchvision import datasets, transforms, utils as tvu
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


def build_loader(data_root, batch_size, num_workers, train=True, download=False):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(to_minus_one_to_one),
    ])
    dataset = datasets.CIFAR10(
        root=data_root,
        train=train,
        download=download,
        transform=transform,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
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

def get_ts_from_cfg(cfg):
    ts = None

    # 1) dict/yaml.safe_load 形式
    if isinstance(cfg, dict):
        ts = cfg.get("stage", None)
        if ts is None and isinstance(cfg.get("distill", None), dict):
            ts = cfg["distill"].get("stage", None)

    # 2) OmegaConf / argparse-like 对象形式
    else:
        ts = getattr(cfg, "stage", None)
        if ts is None and hasattr(cfg, "distill"):
            ts = getattr(cfg.distill, "stage", None)

    if ts is None:
        raise ValueError("`stage` 未在 yaml 中配置，请写：stage: [500] 或 stage: [167,333,...]")

    if not isinstance(ts, (list, tuple)):
        raise ValueError(f"`stage` 必须是 list，例如 stage: [500]，但你现在是 {type(ts)}={ts}")

    ts = [int(x) for x in ts]
    if len(ts) == 0:
        raise ValueError("`stage` 列表为空：请至少提供一个时间步，例如 stage: [500]")

    return ts

def build_cfg(stage, lambda_trans, lambda_score, lr):
    ts = get_ts_from_cfg(cfg)   # 或 get_ts_from_cfg(config)，看你变量名
    print("stage(ts) =", ts)

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


def count_params(m: nn.Module):
    return sum(p.numel() for p in m.parameters())


def psnr_from_mse(mse: torch.Tensor, eps=1e-12):
    mse = torch.clamp(mse, min=eps)
    return (-10.0 * torch.log10(mse)).item()


def try_ssim(x01, y01):
    try:
        from pytorch_msssim import ssim
        return float(ssim(x01, y01, data_range=1.0, size_average=True).item())
    except Exception:
        return None


@torch.no_grad()
def x0_from_eps(scheduler: DDPMScheduler, xt, eps_pred, t):
    ac = scheduler.alphas_cumprod.to(device=xt.device, dtype=xt.dtype)
    a = ac[t].view(-1, 1, 1, 1)
    x0 = (xt - torch.sqrt(1.0 - a) * eps_pred) / torch.sqrt(a)
    return torch.clamp(x0, -1.0, 1.0)


@torch.no_grad()
def eval_denoise_quality(model, scheduler, x, t):
    noise = torch.randn_like(x)
    xt = scheduler.add_noise(x, noise, t)
    eps = model(xt, t)
    x0 = x0_from_eps(scheduler, xt, eps, t)

    x01 = torch.clamp((x + 1.0) * 0.5, 0.0, 1.0)
    x0_01 = torch.clamp((x0 + 1.0) * 0.5, 0.0, 1.0)

    mse = F.mse_loss(x0_01, x01)
    psnr = psnr_from_mse(mse.detach())
    ssim_val = try_ssim(x0_01, x01)
    return psnr, ssim_val


@torch.no_grad()
def sample_ddpm(model, scheduler, image_size, steps, init_noise, device):
    scheduler.set_timesteps(int(steps), device=device)
    x = init_noise.clone()
    n = x.shape[0]
    for tt in scheduler.timesteps:
        t_batch = tt.repeat(n)
        eps = model(x, t_batch)
        out = scheduler.step(eps, tt, x)
        x = out.prev_sample
    return torch.clamp(x, -1.0, 1.0)


@torch.no_grad()
def eval_gen_fidelity(student, teacher_unet, scheduler, image_size, steps, n, device):
    g = torch.Generator(device=device)
    g.manual_seed(1234)
    init = torch.randn((n, 3, image_size, image_size), generator=g, device=device)

    s_img = sample_ddpm(student, scheduler, image_size, steps, init, device)
    t_img = sample_ddpm(teacher_unet, scheduler, image_size, steps, init, device)

    s01 = torch.clamp((s_img + 1.0) * 0.5, 0.0, 1.0)
    t01 = torch.clamp((t_img + 1.0) * 0.5, 0.0, 1.0)

    mse = F.mse_loss(s01, t01)
    psnr = psnr_from_mse(mse.detach())
    ssim_val = try_ssim(s01, t01)
    return psnr, ssim_val, s01, t01


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

    p.add_argument("--stage", type=int, choices=[1, 2, 3], default=int(PATHS.get("stage", 2)))
    p.add_argument("--lambda_trans", type=float, default=float(PATHS.get("lambda_trans", 0.2)))
    p.add_argument("--lambda_score", type=float, default=float(PATHS.get("lambda_score", 0.05)))
    p.add_argument("--lambda_direct", type=float, default=float(PATHS.get("lambda_direct", 1.0)))
    p.add_argument("--lr", type=float, default=PATHS.get("lr", None))

    p.add_argument("--freeze_student", action="store_true", default=bool(PATHS.get("freeze_student", False)))

    p.add_argument("--eval_every", type=int, default=int(PATHS.get("eval_every", 1)))
    p.add_argument("--eval_bs", type=int, default=int(PATHS.get("eval_bs", 64)))
    p.add_argument("--eval_t", type=int, default=int(PATHS.get("eval_t", 500)))
    p.add_argument("--gen_steps", type=int, default=int(PATHS.get("gen_steps", 50)))
    p.add_argument("--gen_n", type=int, default=int(PATHS.get("gen_n", 4)))
    p.add_argument(
        "--save_samples",
        action=argparse.BooleanOptionalAction,
        default=bool(PATHS.get("save_samples", False)),
    )

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

    loader = build_loader(args.data_root, args.batch_size, args.num_workers, train=True, download=bool(args.download))
    eval_loader = build_loader(args.data_root, args.eval_bs, 0, train=False, download=bool(args.download))

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
    if args.save_samples:
        os.makedirs(os.path.join(args.save_dir, "samples"), exist_ok=True)

    print("device:", device)
    print("teacher_ckpt:", str(Path(args.teacher_ckpt).resolve()))
    print("student_ckpt:", str(Path(args.student_ckpt).resolve()))
    print("image_size:", int(args.image_size), "timesteps:", int(args.num_train_timesteps))
    print("cfg.model.ch:", int(model_cfg.model.ch), "ch_mult:", list(model_cfg.model.ch_mult))
    print("attn_resolutions:", list(getattr(model_cfg.model, "attn_resolutions", [])))
    print("teacher params:", count_params(teacher_unet), "student params:", count_params(student), "distiller params:", count_params(distiller))
    print("weights:", "lambda_direct=", float(args.lambda_direct), "lambda_refine=", float(cfg.lambda_trans), "lambda_ddim=", float(cfg.lambda_score))
    print("freeze_student:", bool(args.freeze_student))

    for epoch in range(1, args.epochs + 1):
        distiller.train()
        student.eval() if args.freeze_student else student.train()

        total_loss = 0.0
        total_direct = 0.0
        total_refine = 0.0
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

            direct_loss = F.mse_loss(student_feat, teacher_feat.detach())

            refined, t_feat_used, ddim_loss, rec_loss = distiller(student_feat, teacher_feat)

            refine_loss = F.mse_loss(refined, t_feat_used.detach())

            loss = float(args.lambda_direct) * direct_loss + float(cfg.lambda_trans) * refine_loss + float(cfg.lambda_score) * ddim_loss

            rec_val = 0.0
            if rec_loss is not None:
                loss = loss + rec_loss
                rec_val = float(rec_loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_val = float(loss.item())
            direct_val = float(direct_loss.item())
            refine_val = float(refine_loss.item())
            ddim_val = float(ddim_loss.item())

            total_loss += loss_val
            total_direct += direct_val
            total_refine += refine_val
            total_ddim += ddim_val
            total_rec += rec_val
            steps += 1

            pbar.set_postfix(
                loss=f"{loss_val:.4f}",
                direct=f"{direct_val:.4f}",
                refine=f"{refine_val:.4f}",
                ddim=f"{ddim_val:.4f}",
            )

        mean_loss = total_loss / max(1, steps)
        mean_direct = total_direct / max(1, steps)
        mean_refine = total_refine / max(1, steps)
        mean_ddim = total_ddim / max(1, steps)
        mean_rec = total_rec / max(1, steps)

        print(
            f"epoch {epoch} "
            f"loss={mean_loss:.4f} "
            f"direct={mean_direct:.4f} "
            f"refine={mean_refine:.4f} "
            f"ddim={mean_ddim:.4f} "
            f"rec={mean_rec:.4f}"
        )

        if args.eval_every > 0 and (epoch % int(args.eval_every) == 0):
            xb, _ = next(iter(eval_loader))
            xb = xb.to(device)
            b = xb.size(0)
            t_eval = int(args.eval_t)
            t_eval = max(0, min(t_eval, int(args.num_train_timesteps) - 1))
            tt = torch.full((b,), t_eval, device=device, dtype=torch.long)

            student.eval()
            s_psnr, s_ssim = eval_denoise_quality(student, scheduler, xb, tt)
            teacher_unet.eval()
            t_psnr, t_ssim = eval_denoise_quality(teacher_unet, scheduler, xb, tt)

            msg = f"[eval@t={t_eval}] denoise_psnr student={s_psnr:.3f}"
            if s_ssim is not None:
                msg += f" ssim={s_ssim:.5f}"
            msg += f" | teacher={t_psnr:.3f}"
            if t_ssim is not None:
                msg += f" ssim={t_ssim:.5f}"
            print(msg)

            g_psnr, g_ssim, s01, t01 = eval_gen_fidelity(
                student=student,
                teacher_unet=teacher_unet,
                scheduler=scheduler,
                image_size=int(args.image_size),
                steps=int(args.gen_steps),
                n=int(args.gen_n),
                device=device,
            )
            msg = f"[eval] gen_fidelity (student vs teacher) steps={int(args.gen_steps)} N={int(args.gen_n)} psnr={g_psnr:.3f}"
            if g_ssim is not None:
                msg += f" ssim={g_ssim:.5f}"
            print(msg)

            if args.save_samples:
                grid = torch.cat([t01, s01], dim=0)
                outp = os.path.join(args.save_dir, "samples", f"epoch_{epoch:03d}_T_then_S.png")
                tvu.save_image(grid, outp, nrow=int(args.gen_n), padding=2)

        ckpt = {
            "epoch": epoch,
            "student_frozen": bool(args.freeze_student),
            "student": student.state_dict(),
            "distiller": distiller.state_dict(),
            "cfg": (cfg.__dict__ if hasattr(cfg, "__dict__") else str(cfg)),
            "num_train_timesteps": int(args.num_train_timesteps),
            "teacher_ckpt": str(Path(args.teacher_ckpt).resolve()),
            "student_ckpt": str(Path(args.student_ckpt).resolve()),
            "lambda_direct": float(args.lambda_direct),
            "lambda_refine": float(cfg.lambda_trans),
            "lambda_ddim": float(cfg.lambda_score),
        }
        torch.save(ckpt, os.path.join(args.save_dir, f"distill_epoch_{epoch:03d}.pth"))

        if not args.freeze_student:
            student.train()


if __name__ == "__main__":
    main()
