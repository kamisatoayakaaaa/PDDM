import torch
from torch import nn
import torch.nn.functional as F

from .diffkd_modules import DiffusionModel, NoiseAdapter, AutoEncoder, DDIMPipeline
from .scheduling_ddim import DDIMScheduler
from .timestepconfig import DistillConfig


class DiffKD(nn.Module):
    def __init__(
            self,
            student_channels,
            teacher_channels,
            kernel_size=3,
            inference_steps=5,
            num_train_timesteps=1000,
            use_ae=False,
            ae_channels=None,
            cfg: DistillConfig | None = None,
    ):
        super().__init__()
        self.use_ae = use_ae
        self.diffusion_inference_steps = inference_steps

        # 如果外面没传 cfg，这里直接用默认 DistillConfig
        self.cfg = cfg if cfg is not None else DistillConfig()

        if use_ae:
            if ae_channels is None:
                ae_channels = teacher_channels // 2
            self.ae = AutoEncoder(teacher_channels, ae_channels)
            teacher_channels = ae_channels

        self.trans = nn.Conv2d(student_channels, teacher_channels, 1)
        self.model = DiffusionModel(channels_in=teacher_channels, kernel_size=kernel_size)

        self.scheduler = DDIMScheduler(
            num_train_timesteps=num_train_timesteps,
            clip_sample=False,
            beta_schedule="linear",
        )
        self.noise_adapter = NoiseAdapter(teacher_channels, kernel_size)
        self.pipeline = DDIMPipeline(self.model, self.scheduler, self.noise_adapter)
        self.proj = nn.Sequential(
            nn.Conv2d(teacher_channels, teacher_channels, 1),
            nn.BatchNorm2d(teacher_channels),
        )

        max_t = self.scheduler.config.num_train_timesteps

        # 1）如果 cfg.distill_timesteps 有值，用它
        # 2）否则就自动用 cfg.tau
        focus = None
        if self.cfg.distill_timesteps is not None and len(self.cfg.distill_timesteps) > 0:
            focus = self.cfg.distill_timesteps
        elif self.cfg.tau is not None and len(self.cfg.tau) > 0:
            focus = self.cfg.tau

        if focus is not None:
            cleaned = []
            for t in focus:
                t_int = int(t)
                if 0 <= t_int < int(max_t):
                    cleaned.append(t_int)
            if len(cleaned) > 0:
                self.valid_timesteps = torch.tensor(cleaned, dtype=torch.long)
            else:
                self.valid_timesteps = None
        else:
            self.valid_timesteps = None

    def forward(self, student_feat, teacher_feat):
        student_feat = self.trans(student_feat)

        if self.use_ae:
            hidden_t_feat, rec_t_feat = self.ae(teacher_feat)
            rec_loss = F.mse_loss(teacher_feat, rec_t_feat)
            teacher_feat = hidden_t_feat.detach()
        else:
            rec_loss = None

        refined_feat = self.pipeline(
            batch_size=student_feat.shape[0],
            device=student_feat.device,
            dtype=student_feat.dtype,
            shape=student_feat.shape[1:],
            feat=student_feat,
            num_inference_steps=self.diffusion_inference_steps,
            proj=self.proj,
        )
        refined_feat = self.proj(refined_feat)

        ddim_loss = self.ddim_loss(teacher_feat)
        return refined_feat, teacher_feat, ddim_loss, rec_loss

    def ddim_loss(self, gt_feat):
        noise = torch.randn(gt_feat.shape, device=gt_feat.device)
        bs = gt_feat.shape[0]

        if self.valid_timesteps is None:
            max_t = self.scheduler.config.num_train_timesteps
            timesteps = torch.randint(
                0,
                int(max_t),
                (bs,),
                device=gt_feat.device,
            ).long()
        else:
            vt = self.valid_timesteps.to(gt_feat.device)
            idx = torch.randint(0, vt.shape[0], (bs,), device=gt_feat.device)
            timesteps = vt[idx]

        noisy_images = self.scheduler.add_noise(gt_feat, noise, timesteps)
        noise_pred = self.model(noisy_images, timesteps)
        loss = F.mse_loss(noise_pred, noise)
        return loss
