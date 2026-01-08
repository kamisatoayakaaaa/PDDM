import os
import time
import math
import sys
import numpy as np
import torch
import torch.utils.data as data
from tqdm import tqdm
from models.diffusion import Model
from functions import get_optimizer
from functions.losses_hiding import hiding_loss
from datasets import get_dataset, data_transform, inverse_data_transform
from functions.ckpt_util import get_ckpt_path
import torchvision.utils as tvu
import torchvision.transforms as T
from PIL import Image
from torch.optim import AdamW
import torch.nn as nn
import copy
from .ext_acc import cal_ext_acc
from .md_fidelity import cal_md_fidelity
from diffusers import DDPMPipeline


class Diffusion(object):
    def __init__(self, args, config, device=None, secret_img_pth=""):
        self.args = args
        self.config = config
        self.param_ratio = config.hiding.sparsity
        self.top_n_layers = config.hiding.n_layers
        self.lora_lr = config.hiding.lora_lr
        self.lbd = config.hiding.lbd
        self.ts = config.hiding.ts
        self.loraplus_lr_ratio = config.hiding.loraplus_lr_ratio
        self.lora_dim = config.hiding.rank
        self.locon_dim = config.hiding.rank
        self.secret_img_pth = secret_img_pth
        self.psl_iters = config.hiding.select_iters
        self.psl_lr = config.hiding.select_lr
        self.key = config.hiding.seed
        self.hf_model_id = getattr(self.args, "hf_model_id", None)
        self.base_ckpt = getattr(args, "base_ckpt", "")
        self.use_hf_teacher = self.hf_model_id is not None
        if not os.path.exists(self.args.output_folder):
            os.makedirs(self.args.output_folder)
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device
        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()
        transform = T.Compose([
            T.Resize((config.data.image_size, config.data.image_size)),
            T.ToTensor()
        ])
        images=[]
        for pth in self.secret_img_pth:
            images.append(
                data_transform(self.config, transform(Image.open(pth))).unsqueeze(dim=0)
            )
        self.secret_img = torch.cat(images, dim=0)
        torch.manual_seed(self.key)
        self.zs = torch.randn_like(self.secret_img, memory_format=torch.contiguous_format).to(self.device)
        torch.save(self.zs, os.path.join(self.args.output_folder, "zs.pt"))

        # ---------------------------
        # 选择基座 checkpoint 的逻辑
        # 优先级：
        # 1) 用户通过 --base_ckpt 提供的纯净模型
        # 2) （可选）huggingface 模型 id
        # 3) 退回原来的 ema_cifar10 / ema_lsun_xxx ckpt
        # ---------------------------
        self.ckpt = None

        # 1) 用户自定义 base_ckpt（你现在就是用这个）
        self.base_ckpt = getattr(self.args, "base_ckpt", None)
        if self.base_ckpt is not None:
            print(f"[Diffusion] Using user-provided base_ckpt: {self.base_ckpt}")
            self.ckpt = self.base_ckpt

        # 2) （可选）支持 hf_model_id，将来如果又想从 diffusers 拉模型，可以在这里处理
        #    现在你用的是自己训练的 pure_model，所以这里不会进
        elif getattr(self.args, "hf_model_id", None) is not None:
            from diffusers import DDPMPipeline
            self.hf_model_id = self.args.hf_model_id
            print(f"[Diffusion] Using HuggingFace model as base: {self.hf_model_id}")
            pipe = DDPMPipeline.from_pretrained(self.hf_model_id)
            # 如果之后要用 HF 的 UNet，你可以加一个 self.hf_unet = pipe.unet 之类的
            # 目前你的流程还是用本地 Model(config)，所以这里只是占位

        # 3) 否则走回原来的 ema_xxx 逻辑
        else:
            if self.config.data.dataset == "CIFAR10":
                name = "cifar10"
            elif self.config.data.dataset == "LSUN":
                name = f"lsun_{self.config.data.category}"
            else:
                raise ValueError(f"Unknown dataset: {self.config.data.dataset}")
            self.ckpt = get_ckpt_path(f"ema_{name}")
            print(f"[Diffusion] Using default ema checkpoint: {self.ckpt}")



    def param_select(self):            
        args, config = self.args, self.config

        dataset, _  = get_dataset(args, config)
        train_loader = data.DataLoader(
            dataset,
            batch_size=self.config.hiding.n_secrets,
            shuffle=False,
            num_workers=config.data.num_workers,
        )

        # 如果指定了 hf_model_id，则用 HuggingFace 的 UNet 作为基座
        if self.hf_model_id is not None:
            # 这里会顺带把 self.betas / self.num_timesteps 换成 HF scheduler 的设置
            model = self._load_hf_unet_and_scheduler()
            model = model.to(self.device)

            # param_select 里原本有一个 model_ref，但这里只用于 sensitivity 计算中的参考；
            # 对 HF 分支我们就简单用一个 frozen 的 deep copy
            model_ref = copy.deepcopy(model).to(self.device)
            model_ref.eval()
            for p in model_ref.parameters():
                p.requires_grad_(False)

            # 优化器简单用 Adam，学习率沿用原来的 select_lr
            optimizer = torch.optim.Adam(model.parameters(), lr=self.psl_lr)

        else:
            # 原始 DMIH 路线：用本地 Model(config) + ema ckpt
            model = Model(config).to(self.device)
            model_ref = Model(config).to(self.device)

            optimizer = get_optimizer(self.config, model.parameters())

            states = torch.load(self.ckpt, map_location=self.device)
            model.load_state_dict(states)
            model_ref.load_state_dict(states)

        # 两种分支都共用一套 grad_dict 逻辑
        grad_dict = {name: 0. for name, _ in model.named_parameters()}


        for name, param in model_ref.named_parameters():
            param.requires_grad = False

        data_iterator = iter(train_loader)

        start = time.time()
        print("Calculating sensitivity...")
        pbar = tqdm(range(self.psl_iters), desc="Sensitivity", dynamic_ncols=True)
        for _ in pbar:
            try:
                (x, y) = next(data_iterator)
            except StopIteration:
                data_iterator = iter(train_loader)
                (x, y) = next(data_iterator)

            x = x.to(self.device)
            x = data_transform(self.config, x)

            bs = x.shape[0]

            x_tar = self.secret_img.to(self.device)[:bs]
            e_fixed = self.zs.to(self.device)[:bs]

            t = torch.randint(low=0, high=self.num_timesteps, size=(bs,), device=self.device)

            b = self.betas.to(self.device)
            t_fixed = torch.ones_like(t, device=self.device) * self.ts
            t_fixed = t_fixed + self.config.hiding.ts_interval * torch.arange(bs, device=self.device)

            loss = hiding_loss(
                model=model,
                model_ref=model_ref,
                x0=x,
                t=t,
                b=b,
                x_tar=x_tar,
                t_fixed=t_fixed,
                e_fixed=e_fixed,
                lbd=self.lbd,
            )

            loss_value = float(loss.item())
            if not math.isfinite(loss_value):
                print("Loss is {}, stopping...".format(loss_value))
                sys.exit(1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_dict[name] += (param.grad ** 2).detach()

            pbar.set_postfix(loss=f"{loss_value:.4f}")


        end = time.time()
        hours, rem = divmod(end-start, 3600)
        minutes, seconds = divmod(rem, 60)
        print("Time cost for sensitivity calculation: {:0>2} hours {:0>2} minutes {:05.2f} seconds!".format(int(hours),int(minutes),seconds))

        grad_shapes = {}
        grad_shapes_int = {}

        grad_skip_kwd_list = []

        for key in grad_dict.keys():
            if not any(kwd in key for kwd in grad_skip_kwd_list):
                grad_shapes[key] = grad_dict[key].shape
                grad_shapes_int[key] = np.cumprod(list(grad_dict[key].shape))[-1]

        large_tensor = torch.cat([grad_dict[key].flatten() for key in grad_shapes.keys()])

        grad_sum_dict = {}
        param_num = self.param_ratio
        all_param_num = torch.ones_like(large_tensor).sum()

        values, indexes = large_tensor.topk(math.ceil(param_num * all_param_num))

        tmp_large_tensor = torch.zeros_like(large_tensor, device='cuda')
        tmp_large_tensor[indexes] = 1.

        tmp_large_tensor_list = tmp_large_tensor.split([shape for shape in grad_shapes_int.values()])

        unstructured_param_num = 0
        unstructured_name_shapes = {}
        unstructured_name_shapes_int = {}
        unstructured_grad_mask = {}

        for i, key in enumerate(grad_shapes.keys()):
            grad_sum = tmp_large_tensor_list[i].view(grad_shapes[key]).sum()
            grad_sum_dict[key] = grad_sum

            unstructured_param_num += grad_sum.item()
            unstructured_name_shapes[key] = tmp_large_tensor_list[i].view(grad_shapes[key]).shape
            unstructured_name_shapes_int[key] = np.cumprod(list(grad_dict[key].shape))[-1]
            unstructured_grad_mask[key] = tmp_large_tensor_list[i].view(grad_shapes[key])

        res = {'unstructured_name_shapes': unstructured_name_shapes,
                'unstructured_name_shapes_int': unstructured_name_shapes_int,
                'unstructured_params': unstructured_param_num,
                'unstructured_indexes': torch.nonzero(torch.cat(
                    [unstructured_grad_mask[key].flatten() for key in
                    unstructured_grad_mask.keys()])).squeeze(
                    -1) if unstructured_param_num != 0 else torch.zeros(0).long(),

                }
        torch.save(res, os.path.join(self.args.output_folder, 'param_req_{}.pth'.format(self.param_ratio)))
        del res

        return self.param_ratio, os.path.join(self.args.output_folder, 'param_req_{}.pth'.format(self.param_ratio))

    def train(self):
        args, config = self.args, self.config

        dataset, _ = get_dataset(args, config)
        train_loader = data.DataLoader(
            dataset,
            batch_size=self.config.hiding.n_secrets,
            shuffle=False,
            num_workers=config.data.num_workers,
        )

        # 🔹 和 param_select 一样：如果有 hf_model_id，就用 HuggingFace 的 UNet
        if getattr(self, "hf_model_id", None) is not None:
            # 这里会同步 self.betas 和 self.num_timesteps
            model = self._load_hf_unet_and_scheduler().to(self.device)
            model_ref = copy.deepcopy(model).to(self.device)
            model_ref.eval()
            for p in model_ref.parameters():
                p.requires_grad_(False)
        else:
            # 原始 DMIH 路线：本地 Model(config) + ema ckpt
            model = Model(config).to(self.device)
            model_ref = Model(config).to(self.device)

        sensitivity_path = os.path.join(self.args.output_folder, 'param_req_{}.pth'.format(self.param_ratio))
        param_info = torch.load(sensitivity_path, map_location='cpu')

        unstructured_name_shapes = param_info['unstructured_name_shapes']
        unstructured_indexes = param_info['unstructured_indexes']
        grad_mask = torch.cat([torch.zeros(unstructured_name_shapes[key]).flatten() for key in unstructured_name_shapes.keys()])
        grad_mask[unstructured_indexes] = 1.
        grad_mask = grad_mask.split([np.cumprod(list(shape))[-1] for shape in unstructured_name_shapes.values()])
        grad_mask = {k: mask.view(v) for mask, (k, v) in zip(grad_mask, unstructured_name_shapes.items())}

        layer_name_list = []
        layer_sens_n_list = []
        layer_count=0
        for name, param in model.named_parameters():
            assert 'module.' not in str(name)
            layer_count+=1
            n_sens_params = torch.sum(grad_mask[name])
            layer_sens_n_list.append(n_sens_params.numpy())
            layer_name_list.append(name.rsplit('.', 1)[0])
        
        K = self.top_n_layers
        test_list = layer_sens_n_list
        temp = reversed(sorted(test_list)[-K:])
        res = []
        sens_layer = []
        for ele in temp:
            res.append((test_list.index(ele), ele))
            sens_layer.append(layer_name_list[test_list.index(ele)])

        
        # 🔹 只有在“本地模型”模式下才用原来的 ema ckpt
        if getattr(self, "hf_model_id", None) is None:
            states = torch.load(self.ckpt, map_location=self.device)
            model.load_state_dict(states)
            model_ref.load_state_dict(states)

        # model_ref 始终是冻结的参考模型
        for name, param in model_ref.named_parameters():
            param.requires_grad = False

        # 主模型的底座参数也先全部冻结，后面只训练 LoRA
        for name, param in model.named_parameters():
            param.requires_grad = False


        from locon.locon_kohya import create_network
        lora_network = create_network(unet=model, sens_module=sens_layer, network_dim=self.lora_dim, conv_dim=self.locon_dim)
        lora_network.apply_to()

        optimizer = create_loraplus_optimizer(opt_model=lora_network, optimizer_cls=AdamW, lr=self.lora_lr, loraplus_lr_ratio=self.loraplus_lr_ratio)

        data_iterator = iter(train_loader)
        best_loss = float("inf")
        best_lora_state = None

        total_steps = self.config.hiding.n_iters * self.config.hiding.n_secrets
        pbar = tqdm(range(total_steps), desc="Hiding", dynamic_ncols=True)

        for i in pbar:
            try:
                (x, y) = next(data_iterator)
            except StopIteration:
                data_iterator = iter(train_loader)
                (x, y) = next(data_iterator)

            x = x.to(self.device)
            x = data_transform(self.config, x)
            bs = x.shape[0]

            x_tar = self.secret_img.to(self.device)[:bs]
            e_fixed = self.zs.to(self.device)[:bs]

            t = torch.randint(low=0, high=self.num_timesteps, size=(bs,), device=self.device)

            b = self.betas
            t_fixed = torch.ones_like(t, device=self.device) * self.ts
            t_fixed = t_fixed + self.config.hiding.ts_interval * torch.arange(bs, device=self.device)

            lora_network.apply_to()

            loss = hiding_loss(
                model=model,
                model_ref=model_ref,
                x0=x,
                t=t,
                b=b,
                x_tar=x_tar,
                t_fixed=t_fixed,
                e_fixed=e_fixed,
                lbd=self.lbd,
            )

            loss_val = float(loss.item())
            if loss_val < best_loss:
                best_loss = loss_val
                best_lora_state = {k: v.detach().cpu().clone() for k, v in lora_network.state_dict().items()}

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=f"{loss_val:.4f}", best=f"{best_loss:.4f}")

        if best_lora_state is None:
            best_lora_state = {k: v.detach().cpu().clone() for k, v in lora_network.state_dict().items()}

        if getattr(self, "hf_model_id", None) is None:
            model_export = Model(config).to(self.device)
            states = torch.load(self.ckpt, map_location=self.device)
            model_export.load_state_dict(states, strict=True)
        else:
            model_export = copy.deepcopy(model).to(self.device)

        merge_locon(model_export, best_lora_state, sens_layer)

        states = [model_export.state_dict(), optimizer.state_dict()]
        md_save_pth = os.path.join(self.args.output_folder, "ckpt")
        if not os.path.exists(md_save_pth):
            os.makedirs(md_save_pth)
        torch.save(states, os.path.join(md_save_pth, "ckpt_best_loss.pth"))

        fd_p, fd_s, fd_l, fd_d = self.extract_secret(model_export)

        stego_sample_dir = self.sample_avd(model_export)

        if not os.path.exists(os.path.join(self.args.output_folder, "pretrained_sample_avd")):
            ref_sample_dir = self.sample_avd(model_ref, pretrained=True)
        else:
            ref_sample_dir = os.path.join(self.args.output_folder, "pretrained_sample_avd")

        sc_p, sc_s, sc_l, sc_d = cal_md_fidelity(ref_sample_dir, stego_sample_dir)

        return fd_p, fd_s, fd_l, fd_d, sc_p, sc_s, sc_l, sc_d


    def sample(self):
        model = Model(self.config)
        if not self.args.use_pretrained:
            md_save_pth = os.path.join(self.args.output_folder,"ckpt")
            states_pth = os.path.join(md_save_pth, "ckpt_best_loss.pth")
            states = torch.load(states_pth, map_location=self.config.device)
            print("Loading state dict from: ", states_pth)

            model = model.to(self.device)
            model.load_state_dict(states[0], strict=True)

        else:
            print("Sampling with the pretrained DDPM model...")
            model.load_state_dict(torch.load(self.ckpt, map_location=self.device))
            model.to(self.device)

        self.extract_secret(model)
        self.sample_avd(model, pretrained=self.args.use_pretrained)
        self.sample_fid(model, pretrained=self.args.use_pretrained)
    
    def extract_secret(self, model):
        image_folder = os.path.join(self.args.output_folder, 'extracted')
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)
        print("Extracting secret image to folder: ", image_folder)

        xt = self.zs

        t = torch.randint(
            low=0, high=self.num_timesteps, size=(self.config.hiding.n_secrets,)
        ).to(self.device)

        t_fixed = torch.ones_like(t).to(self.device) * self.ts
        t_fixed = t_fixed + self.config.hiding.ts_interval * torch.tensor(list(range(t_fixed.shape[0]))).to(self.device)

        assert xt.shape[0] == t_fixed.shape[0]

        model.eval()
        at_fixed = compute_alpha(self.betas, t_fixed.long())
        output = model(xt, t_fixed.float())
        e = output
        x0_from_e = (1.0 / at_fixed).sqrt() * xt - (1.0 / at_fixed - 1).sqrt() * e
        x0_from_e = torch.clamp(x0_from_e, -1, 1)
        bd_clean = x0_from_e
        out_img = bd_clean.clone()
        out_img = out_img.detach().cpu()

        assert self.config.hiding.n_secrets == out_img.shape[0]
        assert self.config.hiding.n_secrets == len(self.secret_img_pth)

        fd_ps = fd_ss = fd_ls = fd_ds = 0.0
        for n in range(self.config.hiding.n_secrets):
            out_img_ = out_img[n].unsqueeze(dim=0)
            out_img_ = inverse_data_transform_(out_img_)
            ext_img_save_pth = os.path.join(image_folder, f"extracted_{n}.png")
            tvu.save_image(out_img_, ext_img_save_pth)

            fd_p, fd_s, fd_l, fd_d = cal_ext_acc(self.secret_img_pth[n], ext_img_save_pth)
            fd_ps+=fd_p
            fd_ss+=fd_s
            fd_ls+=fd_l
            fd_ds+=fd_d

        return fd_ps/self.config.hiding.n_secrets, fd_ss/self.config.hiding.n_secrets, fd_ls/self.config.hiding.n_secrets, fd_ds/self.config.hiding.n_secrets

    def sample_avd(self, model, pretrained=False):
        config = self.config
        if pretrained:
            image_folder = os.path.join(self.args.output_folder, 'pretrained_sample_avd')
            print("Sampling from pretrained model...")
            print("Samples saving to folder: ", image_folder)
        else:
            image_folder = os.path.join(self.args.output_folder, 'stego_sample_avd')
            print("Samples saving to folder: ", image_folder)

        if not os.path.exists(image_folder):
            os.makedirs(image_folder)

        img_id = 0
        total_n_samples = 100
        sampling_batch_size = 50
        n_rounds = (total_n_samples - img_id) // sampling_batch_size

        with torch.no_grad():
            torch.manual_seed(42)
            for round in tqdm(
                range(n_rounds), desc="Generating image samples."
            ):
                n = sampling_batch_size
                x = torch.randn(
                    n,
                    config.data.channels,
                    config.data.image_size,
                    config.data.image_size,
                    device=self.device,
                )
                x = self.sample_image(x, model)
                x = inverse_data_transform(config, x)

                for i in range(n):
                    tvu.save_image(
                        x[i], os.path.join(image_folder, f"{img_id}.png")
                    )
                    img_id += 1
        return image_folder
    
    def sample_fid(self, model, pretrained=False):
        config = self.config
        if pretrained:
            image_folder = os.path.join(self.args.output_folder, 'pretrained_sample_fid')
            print("Sampling from pretrained model...")
            print("Samples saving to folder: ", image_folder)
        else:
            image_folder = os.path.join(self.args.output_folder, 'stego_sample_fid')
            print("Samples saving to folder: ", image_folder)
        self.args.eta = 1

        if not os.path.exists(image_folder):
            os.makedirs(image_folder)

        img_id = 0
        total_n_samples = 50000
        sampling_batch_size = 50
        n_rounds = (total_n_samples - img_id) // config.sampling.batch_size

        with torch.no_grad():
            for _ in tqdm(
                range(n_rounds), desc="Generating image samples."
            ):
                x = torch.randn(
                    sampling_batch_size,
                    config.data.channels,
                    config.data.image_size,
                    config.data.image_size,
                    device=self.device,
                )

                x = self.sample_image(x, model)
                x = inverse_data_transform(config, x)

                for i in range(sampling_batch_size):
                    tvu.save_image(
                        x[i], os.path.join(image_folder, f"{img_id}.png")
                    )
                    img_id += 1

    def sample_image(self, x, model, last=True):
        try:
            skip = self.args.skip
        except Exception:
            skip = 1

        if self.args.sample_type == "ddpm_noisy":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
                
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising_tmp import ddpm_steps

            x = ddpm_steps(x, seq, model, self.betas)
        else:
            raise NotImplementedError

        if last:
            x = x[0][-1]

        return x

    def _load_hf_unet_and_scheduler(self):
        """
        用 HuggingFace 的 DDPMPipeline 加载一个预训练扩散模型，
        并且包装成一个“返回 Tensor 的普通 UNet”，
        这样就可以直接喂给 hiding_loss / LoRA / sample_image 使用。
        """
        assert self.hf_model_id is not None, "hf_model_id 为空，不能加载 HF 模型。"

        # 1. 加载 HF pipeline
        pipe = DDPMPipeline.from_pretrained(self.hf_model_id)
        pipe.to(self.device)

        # 2. 用 HF 的 scheduler 同步 betas 和步数
        scheduler = pipe.scheduler
        self.betas = scheduler.betas.to(self.device)

        # 注意：新版 diffusers 推荐用 config.num_train_timesteps
        num_train_timesteps = getattr(scheduler.config, "num_train_timesteps", None)
        if num_train_timesteps is None:
            # 兼容旧版本
            num_train_timesteps = scheduler.num_train_timesteps
        self.num_timesteps = int(num_train_timesteps)

        # 3. 包一层 wrapper，让 forward 返回 Tensor
        class HFUNetWrapper(nn.Module):
            def __init__(self, unet):
                super().__init__()
                self.unet = unet

            def forward(self, x, t):
                # 确保 t 是张量，并在正确的 device 上
                if not torch.is_tensor(t):
                    t = torch.tensor([t], device=x.device, dtype=torch.long).repeat(x.size(0))
                else:
                    t = t.to(device=x.device)

                out = self.unet(x, t)
                # HuggingFace UNet2DModel 返回 UNet2DOutput(sample=...)
                if hasattr(out, "sample"):
                    return out.sample
                return out

        # 返回一个“看起来”跟你原来的 Model 一样的 UNet
        return HFUNetWrapper(pipe.unet)

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a


def inverse_data_transform_(X):
    X = (X + 1.0) / 2.0

    return torch.clamp(X, 0.0, 1.0)


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    if beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def merge_conv(weight_a: nn.Parameter, weight_b: nn.Parameter, device = 'cuda'):
    rank, in_ch, kernel_size, k_ = weight_a.shape
    out_ch, rank_, _, _ = weight_b.shape
    assert rank == rank_ and kernel_size == k_
    
    wa = weight_a.to(device)
    wb = weight_b.to(device)
    
    if device == 'cpu':
        wa = wa.float()
        wb = wb.float()
    
    merged = wb.reshape(out_ch, -1) @ wa.reshape(rank, -1)
    weight = merged.reshape(out_ch, in_ch, kernel_size, kernel_size)
    del wb, wa
    return weight


def merge_linear(weight_a: nn.Parameter, weight_b: nn.Parameter, device = 'cuda'):
    rank, in_ch = weight_a.shape
    out_ch, rank_ = weight_b.shape
    assert rank == rank_
    
    wa = weight_a.to(device)
    wb = weight_b.to(device)
    
    if device == 'cpu':
        wa = wa.float()
        wb = wb.float()
    
    weight = wb @ wa
    del wb, wa
    return weight


def merge_locon(base_model, locon_state_dict, target_replace_modules, scale: float = 1.0, device = 'cuda'):
    def merge(root_module: torch.nn.Module, target_replace_modules):  
        for name, module in list(root_module.named_modules()):
            if module.__class__.__name__ not in {'Linear', 'Conv2d'}:
                continue

            if name in target_replace_modules:

                lora_name = 'lora' + '.' + name
                lora_name = lora_name.replace('.', '_')
                
                down = locon_state_dict[f'{lora_name}.lora_down.weight'].float()
                up = locon_state_dict[f'{lora_name}.lora_up.weight'].float()
                alpha = locon_state_dict[f'{lora_name}.alpha'].float()
                rank = down.shape[0]
                
                if module.__class__.__name__ == 'Conv2d':
                    delta = merge_conv(down, up, device)
                    module.weight.requires_grad_(False)
                    # module.weight += (alpha.to(device)/rank * scale * delta).cuda()
                    module.weight += (alpha.to(device)/math.sqrt(rank) * scale * delta).cuda()
                    
                    del delta
                    # print(str(name), ' Merged.')
                elif module.__class__.__name__ == 'Linear':
                    delta = merge_linear(down, up, device)
                    module.weight.requires_grad_(False)
                    # module.weight += (alpha.to(device)/rank * scale * delta).cuda()
                    module.weight += (alpha.to(device)/math.sqrt(rank) * scale * delta).cuda()
                    del delta

    merge(base_model, target_replace_modules)


def create_loraplus_optimizer(opt_model, optimizer_cls, lr, loraplus_lr_ratio, weight_decay=0.0):
    param_groups = {
        "groupA": {},
        "groupB": {},
    }

    for name, param in opt_model.named_parameters():
        if not param.requires_grad or "org_module" in name:
            continue

        elif "lora_up" in name:
            param_groups["groupB"][name] = param

        elif "lora_down" in name:
            param_groups["groupA"][name] = param

        else:
            continue

    assigned_param_groups = ""
    for group in param_groups:
        assigned_param_groups += f"{group}\n {list(param_groups[group].keys())}\n\n"

    optimizer_grouped_parameters = [
        {
            "params": list(param_groups["groupA"].values()),
            "weight_decay": weight_decay,
            "lr": lr,
        },
        {
            "params": list(param_groups["groupB"].values()),
            "weight_decay": weight_decay,
            "lr": lr * loraplus_lr_ratio,
        },
    ]

    optimizer = optimizer_cls(optimizer_grouped_parameters, lr=lr, weight_decay=weight_decay)
    return optimizer