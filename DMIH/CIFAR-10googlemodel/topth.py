import torch
from safetensors.torch import load_file

sd = load_file("diffusion_pytorch_model.safetensors")
torch.save(sd, "diffusion_pytorch_model.pth")
print("saved diffusion_pytorch_model.pth, keys =", len(sd))
