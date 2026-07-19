"""下载 LingBot-Vision 模型到 models/ 目录。

用法:
    python download_models.py              # 默认下载 large
    python download_models.py --variant base
    python download_models.py --variant giant
"""
import argparse
import shutil
from pathlib import Path

VARIANTS = {
    "small":  "robbyant/lingbot-vision-vit-small",
    "base":   "robbyant/lingbot-vision-vit-base",
    "large":  "robbyant/lingbot-vision-vit-large",
    "giant":  "robbyant/lingbot-vision-vit-giant",
}
LOCAL_NAMES = {
    "small":  "lbotv_vit_small.pt",
    "base":   "lbotv_vit_base.pt",
    "large":  "lbotv_vit_large.pt",
    "giant":  "lbotv_vit_giant.pt",
}


def main():
    ap = argparse.ArgumentParser(description="下载 LingBot-Vision 模型到 models/")
    ap.add_argument("--variant", default="large", choices=list(VARIANTS))
    ap.add_argument("--models-dir", default="models")
    args = ap.parse_args()

    repo_id = VARIANTS[args.variant]
    local_name = LOCAL_NAMES[args.variant]
    out_dir = Path(args.models_dir)
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / local_name

    if out_path.exists():
        size_mb = out_path.stat().st_size / 1024 / 1024
        print(f"[skip] {out_path} 已存在 ({size_mb:.1f} MB)")
        return

    print(f"[download] {repo_id} -> {out_path}")
    from huggingface_hub import hf_hub_download
    cached = hf_hub_download(repo_id=repo_id, filename="model.pt")
    shutil.copy(cached, out_path)
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"[done] {out_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()