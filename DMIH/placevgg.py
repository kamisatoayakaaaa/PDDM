import argparse
import hashlib
import os
import shutil

def sha256_prefix(path, n=8, chunk=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()[:n], h.hexdigest()

def get_torch_checkpoints_dir():
    torch_home = os.environ.get("TORCH_HOME")
    if torch_home:
        return os.path.join(torch_home, "hub", "checkpoints")

    try:
        import torch
        hub_dir = torch.hub.get_dir()
        return os.path.join(hub_dir, "checkpoints")
    except Exception:
        return os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub", "checkpoints")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=str, required=True)
    ap.add_argument("--dst_dir", type=str, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    src = os.path.abspath(os.path.expanduser(args.src))
    if not os.path.isfile(src):
        raise FileNotFoundError(f"src not found: {src}")

    dst_dir = args.dst_dir
    if dst_dir is None:
        dst_dir = get_torch_checkpoints_dir()
    dst_dir = os.path.abspath(os.path.expanduser(dst_dir))
    os.makedirs(dst_dir, exist_ok=True)

    dst = os.path.join(dst_dir, "vgg16-397923af.pth")
    if os.path.exists(dst) and not args.force:
        print("already exists:", dst)
    else:
        shutil.copy2(src, dst)
        print("copied to:", dst)

    pref, full = sha256_prefix(dst)
    ok = (pref == "397923af")
    print("sha256:", full)
    print("starts_with_397923af:", ok)
    if not ok:
        raise RuntimeError("hash prefix mismatch: this file is NOT the official vgg16-397923af.pth (or was corrupted during copy).")

if __name__ == "__main__":
    main()
