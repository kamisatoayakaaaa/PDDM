import os, argparse, json
from pathlib import Path

import numpy as np
import torch
from torchvision.utils import save_image
import matplotlib.pyplot as plt

from train_teacher_coco128 import UNet, Coco128Dataset, DiffusionTrainer
from exp_logger import log_experiment


def make_sampling_buffers(trainer):
    device = trainer.device
    betas = trainer.betas
    alphas = 1.0 - betas
    alphas_cumprod = trainer.alphas_cumprod
    alphas_cumprod_prev = torch.cat(
        [torch.ones(1, device=device, dtype=alphas_cumprod.dtype), alphas_cumprod[:-1]],
        dim=0,
    )
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    sqrt_one_minus_alphas_cumprod = trainer.sqrt_one_minus_alphas_cumprod
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    return {
        "betas": betas,
        "sqrt_recip_alphas": sqrt_recip_alphas,
        "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod,
        "posterior_variance": posterior_variance,
    }


@torch.no_grad()
def p_sample(trainer, x, t, buf):
    betas_t = buf["betas"][t].view(-1, 1, 1, 1)
    sqrt_recip_alphas_t = buf["sqrt_recip_alphas"][t].view(-1, 1, 1, 1)
    sqrt_one_minus_alphas_cumprod_t = buf["sqrt_one_minus_alphas_cumprod"][t].view(
        -1, 1, 1, 1
    )
    eps_theta = trainer.model(x, t.float())
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t / sqrt_one_minus_alphas_cumprod_t * eps_theta
    )
    noise = torch.randn_like(x)
    nonzero_mask = (t != 0).float().view(-1, 1, 1, 1)
    posterior_var_t = buf["posterior_variance"][t].view(-1, 1, 1, 1)
    return model_mean + nonzero_mask * torch.sqrt(posterior_var_t) * noise


@torch.no_grad()
def sample(trainer, batch_size, channels=3, image_size=None):
    device = trainer.device
    if image_size is None:
        image_size = trainer.image_size
    x = torch.randn(batch_size, channels, image_size, image_size, device=device)
    buf = make_sampling_buffers(trainer)
    for step in reversed(range(trainer.timesteps)):
        t = torch.full((batch_size,), step, device=device, dtype=torch.long)
        x = p_sample(trainer, x, t, buf)
    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="eval/diffteacher")
    parser.add_argument("--num_pairs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.ckpt, map_location=device)
    image_size = ckpt.get("image_size", 256)
    timesteps = ckpt.get("timesteps", 8)

    dataset = Coco128Dataset(args.data_root, image_size)
    num_pairs = min(args.num_pairs, len(dataset))

    originals = []
    orig_paths = []
    for i in range(num_pairs):
        x = dataset[i]
        originals.append(x)
        orig_paths.append(dataset.paths[i])
    originals = torch.stack(originals, dim=0).to(device)

    model = UNet(
        image_channels=3,
        base_channels=64,
        channel_mults=(1, 2, 4, 8),
        time_emb_dim=256,
    )
    trainer = DiffusionTrainer(
        model,
        image_size=image_size,
        channels=3,
        timesteps=timesteps,
        device=device,
    )
    trainer.model.load_state_dict(ckpt["model"])
    trainer.model.eval()

    gen = sample(trainer, batch_size=num_pairs, channels=3, image_size=image_size)

    orig_vis = (originals.clamp(-1, 1) + 1) * 0.5
    gen_vis = (gen.clamp(-1, 1) + 1) * 0.5

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_image(orig_vis, out_dir / "originals_grid.png", nrow=min(4, num_pairs))
    save_image(gen_vis, out_dir / "generated_grid.png", nrow=min(4, num_pairs))

    nrows = num_pairs
    fig, axes = plt.subplots(nrows, 2, figsize=(6, 3 * nrows))
    if nrows == 1:
        axes = axes.reshape(1, 2)

    orig_np = orig_vis.cpu().permute(0, 2, 3, 1).numpy()
    gen_np = gen_vis.cpu().permute(0, 2, 3, 1).numpy()

    for i in range(nrows):
        axes[i, 0].imshow(orig_np[i])
        axes[i, 0].axis("off")
        axes[i, 0].set_title("original")
        axes[i, 1].imshow(gen_np[i])
        axes[i, 1].axis("off")
        axes[i, 1].set_title("generated")

    plt.tight_layout()
    fig.savefig(out_dir / "pairs_matplotlib.png", dpi=200)
    plt.close(fig)

    for i in range(num_pairs):
        pair = torch.stack([orig_vis[i], gen_vis[i]], dim=0)
        save_image(pair, out_dir / f"pair_{i:04d}.png", nrow=2)

    np.savez(
        out_dir / "eval_pairs.npz",
        originals=orig_vis.cpu().numpy(),
        generated=gen_vis.cpu().numpy(),
        paths=np.array(orig_paths),
    )

    meta = {
        "ckpt": args.ckpt,
        "data_root": args.data_root,
        "out_dir": str(out_dir),
        "num_pairs": num_pairs,
        "image_size": image_size,
        "timesteps": timesteps,
        "seed": args.seed,
        "orig_paths": orig_paths,
    }
    with open(out_dir / "eval_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    config = {
        "ckpt": args.ckpt,
        "data_root": args.data_root,
        "out_dir": str(out_dir),
        "num_pairs": num_pairs,
        "image_size": image_size,
        "timesteps": timesteps,
        "seed": args.seed,
    }
    metrics = {}
    log_experiment(
        run_name="dmih_teacher_coco128_eval",
        config=config,
        metrics=metrics,
        log_path="experiments.md",
    )


if __name__ == "__main__":
    main()
