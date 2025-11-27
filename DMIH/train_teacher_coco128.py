import os, glob, math, argparse
from pathlib import Path
import json
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image
from tqdm import tqdm
from exp_logger import log_experiment


class Coco128Dataset(Dataset):
    def __init__(self, root, image_size):
        self.paths = []
        exts = ["*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"]
        for ext in exts:
            self.paths.extend(glob.glob(os.path.join(root, "**", ext), recursive=True))
        if not self.paths:
            raise RuntimeError(f"No images found under {root}")
        self.image_size = image_size
        self.transform = T.Compose([
            T.Resize(image_size),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, groups=8):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.block1 = nn.Sequential(
            nn.GroupNorm(groups, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )
        if in_channels != out_channels:
            self.res_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.res_conv = nn.Identity()

    def forward(self, x, t_emb):
        h = self.block1(x)
        t = self.time_mlp(t_emb)
        h = h + t[:, :, None, None]
        h = self.block2(h)
        return h + self.res_conv(x)


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 4, stride=2, padding=1)

    def forward(self, x, t_emb):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x, t_emb):
        x = nn.functional.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, image_channels=3, base_channels=64, channel_mults=(1, 2, 4, 8), time_emb_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )
        self.init_conv = nn.Conv2d(image_channels, base_channels, 3, padding=1)

        channels = [base_channels * m for m in channel_mults]
        self.channels = channels

        downs = []
        in_channels = base_channels
        for i, ch in enumerate(channels):
            downs.append(ResidualBlock(in_channels, ch, time_emb_dim))
            if i != len(channels) - 1:
                downs.append(Downsample(ch))
            in_channels = ch
        self.downs = nn.ModuleList(downs)

        self.mid_block1 = ResidualBlock(in_channels, in_channels, time_emb_dim)
        self.mid_block2 = ResidualBlock(in_channels, in_channels, time_emb_dim)

        ups = []
        in_channels = channels[-1]
        for i in reversed(range(len(channels))):
            out_ch = channels[i]
            ups.append(ResidualBlock(in_channels + out_ch, out_ch, time_emb_dim))
            if i != 0:
                ups.append(Upsample(out_ch))
            in_channels = out_ch
        self.ups = nn.ModuleList(ups)

        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, image_channels, 3, padding=1),
        )

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        x = self.init_conv(x)
        hiddens = []

        for layer in self.downs:
            if isinstance(layer, ResidualBlock):
                x = layer(x, t_emb)
                hiddens.append(x)
            else:
                x = layer(x, t_emb)

        x = self.mid_block1(x, t_emb)
        x = self.mid_block2(x, t_emb)

        for layer in self.ups:
            if isinstance(layer, ResidualBlock):
                skip = hiddens.pop()
                x = torch.cat([x, skip], dim=1)
                x = layer(x, t_emb)
            else:
                x = layer(x, t_emb)

        return self.final_conv(x)


class DiffusionTrainer:
    def __init__(self, model, image_size=256, channels=3, timesteps=1000, device="cuda"):
        self.model = model.to(device)
        self.image_size = image_size
        self.channels = channels
        self.timesteps = timesteps
        self.device = device

        betas = torch.linspace(1e-4, 0.02, timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

    def register_buffer(self, name, tensor):
        setattr(self, name, tensor.to(self.device))

    def sample_timesteps(self, batch_size):
        return torch.randint(0, self.timesteps, (batch_size,), device=self.device).long()

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_ac = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_om = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_ac * x_start + sqrt_om * noise, noise

    def p_losses(self, x_start, t):
        x_noisy, noise = self.q_sample(x_start, t)
        t = t.float()
        pred_noise = self.model(x_noisy, t)
        return nn.functional.mse_loss(pred_noise, noise)

    def train_epoch(self, dataloader, optimizer, epoch):
        self.model.train()
        losses = []
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for x in pbar:
            x = x.to(self.device)
            t = self.sample_timesteps(x.size(0))
            loss = self.p_losses(x, t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_val = loss.item()
            losses.append(loss_val)
            pbar.set_postfix({"loss": loss_val})
        avg_loss = float(sum(losses) / max(len(losses), 1))
        return avg_loss


def get_next_experiment_name(exp_root: Path, prefix: str = "experiment") -> str:
    exp_root.mkdir(parents=True, exist_ok=True)
    idx_max = 0
    for p in exp_root.iterdir():
        if p.is_dir() and p.name.startswith(prefix):
            suffix = p.name[len(prefix):]
            if suffix.isdigit():
                idx_max = max(idx_max, int(suffix))
    return f"{prefix}{idx_max + 1:02d}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--exp_name", type=str, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    base_dir = Path(__file__).resolve().parent
    exp_root = base_dir / "experiments"

    if args.exp_name is None:
        exp_name = get_next_experiment_name(exp_root, prefix="experiment")
    else:
        exp_name = args.exp_name

    exp_dir = exp_root / exp_name
    ckpt_dir = exp_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    dataset = Coco128Dataset(args.data_root, args.image_size)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )

    model = UNet(
        image_channels=3,
        base_channels=64,
        channel_mults=(1, 2, 4, 8),
        time_emb_dim=256,
    )
    trainer = DiffusionTrainer(
        model,
        image_size=args.image_size,
        channels=3,
        timesteps=args.timesteps,
        device=device,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    train_losses = []
    best_train = float("inf")
    best_path = ckpt_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        avg_loss = trainer.train_epoch(dataloader, optimizer, epoch)
        train_losses.append(avg_loss)

        if avg_loss < best_train:
            best_train = avg_loss
            ckpt = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "image_size": args.image_size,
                "timesteps": args.timesteps,
                "train_loss": avg_loss,
            }
            torch.save(ckpt, best_path)

    history = {
        "train_losses": train_losses,
        "best_train": best_train,
    }
    exp_dir.mkdir(parents=True, exist_ok=True)
    with open(exp_dir / "loss_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    plt.figure()
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label="train")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(exp_dir / "loss_curve.png")
    plt.close()

    config = {
        "data_root": args.data_root,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "timesteps": args.timesteps,
        "lr": args.lr,
        "exp_name": exp_name,
        "exp_dir": str(exp_dir),
        "ckpt_dir": str(ckpt_dir),
        "best_ckpt": str(best_path),
    }
    metrics = {
        "best_train_loss": best_train,
        "final_train_loss": train_losses[-1] if train_losses else None,
    }
    log_experiment(
        run_name="dmih_teacher_coco128",
        config=config,
        metrics=metrics,
        log_path="experiments.md",
    )


if __name__ == "__main__":
    main()
