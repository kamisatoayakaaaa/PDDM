#蒸馏从此开始，直接使用python train1.py
import argparse
import copy

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from diffusers import DDPMPipeline

from diffkd import DiffKD
from timestepconfig import DistillConfig

def to_minus_one_to_one(x):
    return x * 2 - 1


def build_loader(data_root, batch_size, num_workers):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(to_minus_one_to_one),  # 用顶层函数，能被 pickle
    ])
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
    )
    return loader

def build_teacher(device):
    pipe = DDPMPipeline.from_pretrained("google/ddpm-cifar10-32")
    pipe.to(device)
    pipe.unet.eval()
    return pipe.unet, pipe.scheduler


def build_student(student_ckpt, teacher_unet: nn.Module, device):
    obj = torch.load(student_ckpt, map_location=device)

    if isinstance(obj, nn.Module):
        print("Loaded student as full nn.Module from", student_ckpt)
        student = obj.to(device)
    elif isinstance(obj, dict):
        print("Loaded student checkpoint as state_dict from", student_ckpt)
        print("Using a copy of teacher UNet as student architecture.")
        student = copy.deepcopy(teacher_unet).to(device)
        missing, unexpected = student.load_state_dict(obj, strict=False)
        if missing or unexpected:
            print(f"Loaded student checkpoint with {len(missing)} missing and {len(unexpected)} unexpected keys (strict=False).")
            # 如果你连这句话也不想要，直接整段删掉就行

    else:
        raise TypeError(f"Unsupported checkpoint type: {type(obj)}")

    return student


def build_cfg(stage, lambda_trans, lambda_score, lr):
    if stage == 1:
        ts = [500]
    elif stage == 2:
        ts = [470, 485, 500, 515, 530]
    else:
        ts = []  # 空则在 DiffKD 里 fallback 到 tau 全时间轴

    cfg = DistillConfig(
        distill_timesteps=ts,
        lambda_trans=lambda_trans,
        lambda_score=lambda_score,
    )
    if lr is not None:
        cfg.lr = lr
    return cfg


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data_root",
        type=str,
        default=r"C:\Users\17007\Desktop\PDDM\data",
        help="CIFAR-10 父目录（下面有 cifar-10-batches-py）",
    )
    p.add_argument(
        "--student_ckpt",
        type=str,
        default=r"pure_model\train2\best.pth",
        help="学生模型 ckpt 路径（可以是 full model 或 state_dict）",
    )
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--stage", type=int, choices=[1, 2, 3], default=1)
    p.add_argument("--lambda_trans", type=float, default=1.0)
    p.add_argument("--lambda_score", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--freeze_student", action="store_true")
    p.add_argument("--save_dir", type=str, default="distill_ckpts")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)

    loader = build_loader(args.data_root, args.batch_size, args.num_workers)
    teacher_unet, scheduler = build_teacher(device)
    student = build_student(args.student_ckpt, teacher_unet, device)
    cfg = build_cfg(args.stage, args.lambda_trans, args.lambda_score, args.lr)

    c, h, w = 3, 32, 32
    dummy_x = torch.randn(1, c, h, w, device=device)
    dummy_t = torch.zeros(1, device=device, dtype=torch.long)

    with torch.no_grad():
        t_out = teacher_unet(dummy_x, dummy_t)
        t_feat = t_out.sample if hasattr(t_out, "sample") else t_out

        # 对 student 也做同样处理
        s_out = student(dummy_x, dummy_t)  # 既然是 UNet2DModel，就用 int 时间步
        s_feat = s_out.sample if hasattr(s_out, "sample") else s_out

    student_channels = s_feat.shape[1]
    teacher_channels = t_feat.shape[1]


    num_train_timesteps = getattr(scheduler, "num_train_timesteps", None)
    if num_train_timesteps is None and hasattr(scheduler, "config"):
        num_train_timesteps = getattr(scheduler.config, "num_train_timesteps")
    num_train_timesteps = int(num_train_timesteps)

    distiller = DiffKD(
        student_channels=student_channels,
        teacher_channels=teacher_channels,
        kernel_size=3,
        inference_steps=5,
        num_train_timesteps=num_train_timesteps,
        use_ae=False,
        ae_channels=None,
        cfg=cfg,
    ).to(device)

    for p in teacher_unet.parameters():
        p.requires_grad_(False)
    teacher_unet.eval()

    if args.freeze_student:
        for p in student.parameters():
            p.requires_grad_(False)
        student.eval()
    else:
        student.train()

    params = list(distiller.parameters())
    if not args.freeze_student:
        params += list(student.parameters())

    optimizer = torch.optim.Adam(params, lr=cfg.lr)

    if args.save_dir is not None:
        import os
        os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        distiller.train()
        if args.freeze_student:
            student.eval()
        else:
            student.train()

        total_loss = 0.0
        total_trans = 0.0
        total_ddim = 0.0
        total_rec = 0.0
        steps = 0

        for x, _ in loader:
            x = x.to(device)
            b = x.size(0)

            t = torch.randint(
                0,
                num_train_timesteps,
                (b,),
                device=device,
                dtype=torch.long,
            )
            noise = torch.randn_like(x)
            xt = scheduler.add_noise(x, noise, t)

            with torch.no_grad():
                t_out = teacher_unet(xt, t)
                teacher_feat = t_out.sample if hasattr(t_out, "sample") else t_out

            if args.freeze_student:
                with torch.no_grad():
                    s_out = student(xt, t)   # UNet2DModel 用 long timestep
                    student_feat = s_out.sample if hasattr(s_out, "sample") else s_out
            else:
                s_out = student(xt, t)
                student_feat = s_out.sample if hasattr(s_out, "sample") else s_out


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

            total_loss += float(loss.item())
            total_trans += float(trans_loss.item())
            total_ddim += float(ddim_loss.item())
            total_rec += rec_val
            steps += 1

        mean_loss = total_loss / steps
        mean_trans = total_trans / steps
        mean_ddim = total_ddim / steps
        mean_rec = total_rec / steps

        print(
            f"epoch {epoch} "
            f"loss={mean_loss:.4f} "
            f"trans={mean_trans:.4f} "
            f"ddim={mean_ddim:.4f} "
            f"rec={mean_rec:.4f}"
        )

        if args.save_dir is not None:
            ckpt = {
                "epoch": epoch,
                "student_frozen": args.freeze_student,
                "student": student.state_dict(),
                "distiller": distiller.state_dict(),
                "cfg": cfg,
                "num_train_timesteps": num_train_timesteps,
            }
            torch.save(
                ckpt,
                f"{args.save_dir}/distill_epoch_{epoch:03d}.pth",
            )


if __name__ == "__main__":
    main()
