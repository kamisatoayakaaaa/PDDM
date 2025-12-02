import os
import time
import math
import argparse
import yaml
import json
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.diffusion import Model


# ---- 全局 transform，避免在函数里定义 lambda 导致多进程无法 pickle ----
CIFAR_TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # x in [0,1] -> (x - 0.5) / 0.5 in [-1,1]
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


def dict2namespace(config_dict):
    namespace = argparse.Namespace()
    for key, value in config_dict.items():
        if isinstance(value, dict):
            value = dict2namespace(value)
        setattr(namespace, key, value)
    return namespace


def load_config(cfg_name):
    cfg_path = os.path.join("configs", cfg_name)
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg_ns = dict2namespace(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg_ns.device = device
    return cfg_ns


def get_next_run_dir(base_dir):
    os.makedirs(base_dir, exist_ok=True)
    existing = [d for d in os.listdir(base_dir)
                if d.startswith("train") and os.path.isdir(os.path.join(base_dir, d))]
    idx = 1
    while f"train{idx}" in existing:
        idx += 1
    run_dir = os.path.join(base_dir, f"train{idx}")
    os.makedirs(run_dir, exist_ok=False)
    return run_dir


def get_beta_schedule(beta_schedule, beta_start, beta_end, num_diffusion_timesteps):
    if beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def build_dataloader(data_root, batch_size, num_workers):
    # 注意：transform 用的是上面定义的全局 CIFAR_TRAIN_TRANSFORM，
    # 里面没有 lambda，不会触发 Windows 多进程 pickle 报错。
    train_set = datasets.CIFAR10(
        root=data_root,
        train=True,
        download=False,
        transform=CIFAR_TRAIN_TRANSFORM,
    )

    # 为了更稳一点，如果在 Windows 上且 num_workers > 0，可以强制改成 0
    if os.name == "nt" and num_workers > 0:
        print(f"[Info] Detected Windows. For safety, overriding num_workers={num_workers} -> 0")
        num_workers = 0

    loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    return loader


def train(args):
    config = load_config(args.config)
    device = config.device

    base_run_dir = os.path.join(os.path.dirname(__file__), "pure_model")
    run_dir = get_next_run_dir(base_run_dir)
    print(f"run_dir: {run_dir}")

    # 固定随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dataloader = build_dataloader(args.data_root, args.batch_size, args.num_workers)

    model = Model(config).to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    betas = get_beta_schedule(
        beta_schedule=config.diffusion.beta_schedule,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
        num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
    )
    betas = torch.from_numpy(betas).float().to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    num_timesteps = betas.shape[0]

    global_step = 0
    best_loss = float("inf")
    best_state = None
    history = []

    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        num_batches = 0
        start_time = time.time()

        for batch in dataloader:
            x, _ = batch
            x = x.to(device)

            noise = torch.randn_like(x)
            bsz = x.size(0)
            t = torch.randint(
                low=0,
                high=num_timesteps,
                size=(bsz,),
                device=device,
            )

            sqrt_ac = sqrt_alphas_cumprod[t].view(bsz, 1, 1, 1)
            sqrt_om = sqrt_one_minus_alphas_cumprod[t].view(bsz, 1, 1, 1)
            x_t = sqrt_ac * x + sqrt_om * noise

            noise_pred = model(x_t, t.float())
            loss = F.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip is not None and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            loss_val = loss.item()
            epoch_loss += loss_val
            num_batches += 1
            global_step += 1

        epoch_loss /= max(1, num_batches)
        elapsed = time.time() - start_time

        history.append(
            {
                "epoch": epoch,
                "loss": epoch_loss,
                "time_sec": elapsed,
            }
        )

        print(f"Epoch {epoch}/{args.epochs}  loss={epoch_loss:.6f}  time={elapsed:.2f}s")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            # 存 CPU 版 state dict，避免显存占用
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    best_path = os.path.join(run_dir, "best.pth")
    torch.save(best_state, best_path)

    log_path = os.path.join(run_dir, "train_log.md")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("# Diffusion base model training\n\n")
        f.write(f"- config: `{args.config}`\n")
        f.write(f"- data_root: `{args.data_root}`\n")
        f.write(f"- epochs: {args.epochs}\n")
        f.write(f"- batch_size: {args.batch_size}\n")
        f.write(f"- lr: {args.lr}\n")
        f.write(f"- best_loss: {best_loss:.6f}\n")
        f.write(f"- best_ckpt: `{best_path}`\n\n")
        f.write("## Loss history\n\n")
        f.write("```json\n")
        f.write(json.dumps(history, indent=2))
        f.write("\n```")

    print(f"Training done. Best ckpt saved to: {best_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="cifar10.yml")
    parser.add_argument("--data_root", type=str, default=r"C:\Users\17007\Desktop\PDDM\data")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    # Windows 下建议默认 0，有需要你可以改成 2/4 再试
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
