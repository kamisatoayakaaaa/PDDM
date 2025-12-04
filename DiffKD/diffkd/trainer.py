import os
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from diffkd import DiffKD
from timestepconfig import DistillConfig


class DistillTrainer:
    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        noise_scheduler,
        cfg: Optional[DistillConfig] = None,
        device: Optional[torch.device] = None,
        freeze_teacher: bool = True,
        freeze_student: bool = True,
        input_shape: Tuple[int, int, int] = (3, 32, 32),
        inference_steps: int = 5,
        use_ae: bool = False,
        ae_channels: Optional[int] = None,
        num_train_timesteps: Optional[int] = None,
        save_dir: Optional[str] = None,
    ) -> None:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.teacher = teacher.to(device)
        self.student = student.to(device)
        self.noise_scheduler = noise_scheduler
        self.cfg = cfg or DistillConfig()

        if freeze_teacher:
            for p in self.teacher.parameters():
                p.requires_grad_(False)
            self.teacher.eval()

        self.freeze_student = freeze_student
        if freeze_student:
            for p in self.student.parameters():
                p.requires_grad_(False)
            self.student.eval()

        if num_train_timesteps is not None:
            self.num_train_timesteps = int(num_train_timesteps)
        else:
            n = getattr(self.noise_scheduler, "num_train_timesteps", None)
            if n is None and hasattr(self.noise_scheduler, "config"):
                n = getattr(self.noise_scheduler.config, "num_train_timesteps", None)
            if n is None:
                raise ValueError("noise_scheduler must provide num_train_timesteps or config.num_train_timesteps")
            self.num_train_timesteps = int(n)

        c, h, w = input_shape
        dummy_x = torch.randn(1, c, h, w, device=device)
        dummy_t = torch.zeros(1, device=device, dtype=torch.long)

        with torch.no_grad():
            t_out = self.teacher(dummy_x, dummy_t)
            if hasattr(t_out, "sample"):
                t_feat = t_out.sample
            else:
                t_feat = t_out
            s_out = self.student(dummy_x, dummy_t.float())

        student_channels = int(s_out.shape[1])
        teacher_channels = int(t_feat.shape[1])

        self.distiller = DiffKD(
            student_channels=student_channels,
            teacher_channels=teacher_channels,
            kernel_size=3,
            inference_steps=inference_steps,
            num_train_timesteps=self.num_train_timesteps,
            use_ae=use_ae,
            ae_channels=ae_channels,
            cfg=self.cfg,
        ).to(device)

        self.save_dir = Path(save_dir) if save_dir is not None else None
        if self.save_dir is not None:
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def _forward_teacher(self, x, t_long):
        with torch.no_grad():
            out = self.teacher(x, t_long)
            if hasattr(out, "sample"):
                return out.sample
            return out

    def _forward_student(self, x, t_long):
        if self.freeze_student:
            with torch.no_grad():
                return self.student(x, t_long.float())
        return self.student(x, t_long.float())

    def train_one_epoch(self, loader: DataLoader, optimizer: torch.optim.Optimizer):
        self.distiller.train()
        if self.freeze_student:
            self.student.eval()
        else:
            self.student.train()

        total_loss = 0.0
        total_trans = 0.0
        total_ddim = 0.0
        total_rec = 0.0
        steps = 0

        for x, _ in loader:
            x = x.to(self.device)
            b = x.size(0)

            t = torch.randint(
                0,
                self.num_train_timesteps,
                (b,),
                device=self.device,
                dtype=torch.long,
            )
            noise = torch.randn_like(x)
            xt = self.noise_scheduler.add_noise(x, noise, t)

            teacher_feat = self._forward_teacher(xt, t)
            student_feat = self._forward_student(xt, t)

            refined, t_feat, ddim_loss, rec_loss = self.distiller(student_feat, teacher_feat)

            trans_loss = F.mse_loss(refined, t_feat.detach())
            loss = self.cfg.lambda_trans * trans_loss + self.cfg.lambda_score * ddim_loss
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

        return {
            "loss": mean_loss,
            "trans_loss": mean_trans,
            "ddim_loss": mean_ddim,
            "rec_loss": mean_rec,
        }

    def fit(self, loader: DataLoader, epochs: int, lr: float = 1e-4):
        params = list(self.distiller.parameters())
        if not self.freeze_student:
            params += list(self.student.parameters())
        optimizer = torch.optim.Adam(params, lr=lr)

        history = []
        for epoch in range(1, epochs + 1):
            metrics = self.train_one_epoch(loader, optimizer)
            history.append(metrics)

            print(
                f"epoch {epoch} "
                f"loss={metrics['loss']:.4f} "
                f"trans={metrics['trans_loss']:.4f} "
                f"ddim={metrics['ddim_loss']:.4f} "
                f"rec={metrics['rec_loss']:.4f}"
            )

            if self.save_dir is not None:
                ckpt = {
                    "epoch": epoch,
                    "student_frozen": self.freeze_student,
                    "student": self.student.state_dict(),
                    "distiller": self.distiller.state_dict(),
                    "cfg": self.cfg,
                    "num_train_timesteps": self.num_train_timesteps,
                }
                torch.save(ckpt, self.save_dir / f"distill_epoch_{epoch:03d}.pth")

        return history
