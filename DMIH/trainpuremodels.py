import os
import math
import argparse
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, groups=8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x, t_emb):
        h = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(h)
        time_emb = self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = h + time_emb
        h = self.norm2(h)
        h = self.act2(h)
        h = self.conv2(h)
        return h + self.skip(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, down=True):
        super().__init__()
        self.res1 = ResBlock(in_channels, out_channels, time_emb_dim)
        self.res2 = ResBlock(out_channels, out_channels, time_emb_dim)
        if down:
            self.downsample = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
        else:
            self.downsample = nn.Identity()

    def forward(self, x, t_emb):
        h = self.res1(x, t_emb)
        h = self.res2(h, t_emb)
        skip = h
        h = self.downsample(h)
        return h, skip


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, up=True):
        super().__init__()
        self.res1 = ResBlock(in_channels, out_channels, time_emb_dim)
        self.res2 = ResBlock(out_channels, out_channels, time_emb_dim)
        if up:
            self.upsample = nn.ConvTranspose2d(out_channels, out_channels, 4, stride=2, padding=1)
        else:
            self.upsample = nn.Identity()

    def forward(self, x, skip, t_emb):
        h = torch.cat([x, skip], dim=1)
        h = self.res1(h, t_emb)
        h = self.res2(h, t_emb)
        h = self.upsample(h)
        return h


class UNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=128, channel_mults=(1, 2, 2, 4), time_emb_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )
        self.conv_in = nn.Conv2d(in_channels, base_channels * channel_mults[0], 3, padding=1)
        self.downs = nn.ModuleList()
        in_ch = base_channels * channel_mults[0]
        self.skip_channels = []
        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            down = i != len(channel_mults) - 1
            self.downs.append(DownBlock(in_ch, out_ch, time_emb_dim, down=down))
            self.skip_channels.append(out_ch)
            in_ch = out_ch
        self.mid_block1 = ResBlock(in_ch, in_ch, time_emb_dim)
        self.mid_block2 = ResBlock(in_ch, in_ch, time_emb_dim)
        self.ups = nn.ModuleList()
        rev_skips = list(reversed(self.skip_channels))
        curr_ch = in_ch
        for i, skip_ch in enumerate(rev_skips):
            up = i != len(rev_skips) - 1
            block_in = curr_ch + skip_ch
            block_out = skip_ch
            self.ups.append(UpBlock(block_in, block_out, time_emb_dim, up=up))
            curr_ch = block_out
        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, curr_ch),
            nn.SiLU(),
            nn.Conv2d(curr_ch, in_channels, 3, padding=1),
        )

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        h = self.conv_in(x)
        skips = []
        for down in self.downs:
            h, skip = down(h, t_emb)
            skips.append(skip)
        h = self.mid_block1(h, t_emb)
        h = self.mid_block2(h, t_emb)
        for up in self.ups:
            skip = skips.pop()
            h = up(h, skip, t_emb)
        return self.conv_out(h)


def get_dataloader(data_root, batch_size, num_workers):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    dataset = datasets.CIFAR10(
        root=data_root,
        train=True,
        download=False,
        transform=transform,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return loader


def set_seed(seed):
    if seed is None:
        return
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_next_run_dir(base_dir):
    os.makedirs(base_dir, exist_ok=True)
    existing = [
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("train")
    ]
    max_idx = 0
    for d in existing:
        tail = d[5:]
        if tail.isdigit():
            max_idx = max(max_idx, int(tail))
    run_name = f"train{max_idx + 1}"
    run_dir = os.path.join(base_dir, run_name)
    os.makedirs(run_dir, exist_ok=False)
    return run_dir, run_name


def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dir, run_name = get_next_run_dir(args.out_root)
    print(f"run_dir: {run_dir}")

    model = UNet(
        in_channels=3,
        base_channels=args.base_channels,
        channel_mults=tuple(args.channel_mults),
        time_emb_dim=args.time_emb_dim,
    ).to(device)

    betas = torch.linspace(args.beta_start, args.beta_end, args.num_train_timesteps, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    dataloader = get_dataloader(args.data_root, args.batch_size, args.num_workers)

    best_loss = float("inf")
    global_step = 0
    epoch_losses = []

    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        t0 = time.time()
        for x, _ in dataloader:
            x = x.to(device)
            b = x.size(0)

            noise = torch.randn_like(x)
            timesteps = torch.randint(
                0,
                args.num_train_timesteps,
                (b,),
                device=device,
            ).long()

            alpha_bar = alphas_cumprod[timesteps]
            sqrt_alpha_bar = alpha_bar.sqrt().view(b, 1, 1, 1)
            sqrt_one_minus_alpha_bar = (1.0 - alpha_bar).sqrt().view(b, 1, 1, 1)
            x_t = sqrt_alpha_bar * x + sqrt_one_minus_alpha_bar * noise

            noise_pred = model(x_t, timesteps)
            loss = F.mse_loss(noise_pred, noise)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip is not None and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            global_step += 1
            epoch_loss += loss.item() * b

            if global_step % args.log_interval == 0:
                print(f"epoch {epoch} step {global_step} loss {loss.item():.4f}")

            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(
                    {"model": model.state_dict()},
                    os.path.join(run_dir, "best.pth"),
                )

        epoch_loss = epoch_loss / len(dataloader.dataset)
        epoch_losses.append(epoch_loss)
        t1 = time.time()
        print(f"epoch {epoch} mean_loss {epoch_loss:.4f} best_loss {best_loss:.4f} time {t1 - t0:.1f}s")

    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict()},
        os.path.join(run_dir, "last.pth"),
    )

    summary_path = os.path.join(run_dir, "summary.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# Pure CIFAR-10 Diffusion Training\n\n")
        f.write(f"- run_name: {run_name}\n")
        f.write(f"- run_dir: `{run_dir}`\n")
        f.write(f"- datetime: {datetime.now().isoformat(sep=' ', timespec='seconds')}\n")
        f.write(f"- device: {device}\n")
        f.write(f"- epochs: {args.epochs}\n")
        f.write(f"- batch_size: {args.batch_size}\n")
        f.write(f"- num_train_timesteps: {args.num_train_timesteps}\n")
        f.write(f"- beta_start: {args.beta_start}\n")
        f.write(f"- beta_end: {args.beta_end}\n")
        f.write(f"- base_channels: {args.base_channels}\n")
        f.write(f"- channel_mults: {list(args.channel_mults)}\n")
        f.write(f("- time_emb_dim: {}\n").format(args.time_emb_dim))
        f.write(f("- lr: {args.lr}\n"))
        f.write(f("- weight_decay: {args.weight_decay}\n"))
        f.write(f("- grad_clip: {args.grad_clip}\n"))
        f.write(f("- data_root: `{args.data_root}`\n"))
        f.write(f("\nBest step loss: {best_loss:.6f}\n\n"))
        f.write("## Epoch mean losses\n\n")
        f.write("| epoch | mean_loss |\n")
        f.write("| --- | --- |\n")
        for i, l in enumerate(epoch_losses):
            f.write(f"| {i} | {l:.6f} |\n")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--out_root", type=str, default="./pure_model")

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--num_train_timesteps", type=int, default=1000)
    parser.add_argument("--beta_start", type=float, default=0.0001)
    parser.add_argument("--beta_end", type=float, default=0.02)

    parser.add_argument("--base_channels", type=int, default=128)
    parser.add_argument("--channel_mults", nargs="+", type=int, default=[1, 2, 2, 4])
    parser.add_argument("--time_emb_dim", type=int, default=256)

    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--log_interval", type=int, default=100)

    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
