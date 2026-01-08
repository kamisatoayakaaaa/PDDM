import argparse
import yaml
import sys
import os
import glob
import torch
import random
from tqdm import tqdm
from runners.diffusion_hiding import Diffusion

torch.set_printoptions(sci_mode=False)


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])
    parser.add_argument("--config", type=str, default="cifar10.yml")
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--image_folder", type=str, default="images")
    parser.add_argument("--output_folder", type=str, default="output")
    parser.add_argument("--use_pretrained", action="store_true")
    parser.add_argument("--sample_type", type=str, default="ddpm_noisy")
    parser.add_argument("--skip_type", type=str, default="uniform")
    parser.add_argument("--timesteps", type=int, default=1000, help="number of steps involved in ddpm sampling")
    parser.add_argument("--eta", type=float, default=1, help="eta used to control the variances of sigma")
    parser.add_argument(
        "--hf_model_id",
        type=str,
        default=None,
        help="(可选) HuggingFace 预训练扩散模型 id，例如 'google/ddpm-cifar10-32'",
    )
    parser.add_argument(
        "--base_ckpt",
        type=str,
        default="",
        help="本地 diffusion 基座模型的 ckpt 路径（.pth），若为空则使用默认/官方权重",
    )

    args = parser.parse_args()
    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    new_config.device = device
    torch.backends.cudnn.benchmark = True

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config = parse_args_and_config()
    secret_imgs = sorted(glob.glob(os.path.join(args.image_folder, "*.png")))

    n_secrets_set = len(secret_imgs) // int(config.hiding.n_secrets)

    psnr_f = ssim_f = lpips_f = dists_f = psnr_s = ssim_s = lpips_s = dists_s = 0.0

    it = zip(*(iter(secret_imgs),) * int(config.hiding.n_secrets))
    pbar = tqdm(
        it,
        total=n_secrets_set,
        desc="Sampling" if args.sample else "Training",
        dynamic_ncols=True,
    )

    for i, secret_imgs_pth in enumerate(pbar, start=1):
        runner = Diffusion(args, config, secret_img_pth=secret_imgs_pth)
        if args.sample:
            runner.sample()
        else:
            runner.param_select()
            def _f(x):
                return float(x.detach().cpu().item()) if torch.is_tensor(x) else float(x)

            fd_p, fd_s, fd_l, fd_d, sc_p, sc_s, sc_l, sc_d = runner.train()

            psnr_f += _f(fd_p)
            ssim_f += _f(fd_s)
            lpips_f += _f(fd_l)
            dists_f += _f(fd_d)

            psnr_s += _f(sc_p)
            ssim_s += _f(sc_s)
            lpips_s += _f(sc_l)
            dists_s += _f(sc_d)


            pbar.set_postfix(
                ext_psnr=f"{psnr_f / i:.3f}",
                ext_ssim=f"{ssim_f / i:.3f}",
                fid_psnr=f"{psnr_s / i:.3f}",
                fid_ssim=f"{ssim_s / i:.3f}",
            )

    if n_secrets_set > 0 and not args.sample:
        print(
            "Average Extraction Accuracy: PSNR={}, SSIM={}, LPIPS={}, DISTS={}.".format(
                psnr_f / n_secrets_set,
                ssim_f / n_secrets_set,
                lpips_f / n_secrets_set,
                dists_f / n_secrets_set,
            )
        )
        print(
            "Average Model Fidelity: PSNR={}, SSIM={}, LPIPS={}, DISTS={}.".format(
                psnr_s / n_secrets_set,
                ssim_s / n_secrets_set,
                lpips_s / n_secrets_set,
                dists_s / n_secrets_set,
            )
        )
    elif n_secrets_set == 0:
        print("No secret images found (or not enough images to form one set).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
