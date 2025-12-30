import torch

SRC = r"CIFAR-10googlemodel\diffusion_pytorch_model_dmih_fix1x1.pth"
DST = r"CIFAR-10googlemodel\diffusion_pytorch_model_dmih_fix1x1_v2.pth"

def safe_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")

sd = safe_load(SRC)

fixed = 0
targets = ("q.weight", "k.weight", "v.weight", "proj_out.weight")

for k, v in list(sd.items()):
    if (
        (".attn." in k or ".attn_" in k)   # ✅ 同时覆盖 attn. 和 attn_1
        and k.endswith(targets)
        and isinstance(v, torch.Tensor)
        and v.ndim == 2
    ):
        sd[k] = v.unsqueeze(-1).unsqueeze(-1)
        fixed += 1

torch.save(sd, DST)
print("saved:", DST)
print("fixed:", fixed)
print("mid.attn_1.q.weight shape:", sd["mid.attn_1.q.weight"].shape)
