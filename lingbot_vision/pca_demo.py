"""Run a LingBot-Vision backbone and save PCA patch-token visualizations."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from .loader import DTYPE_MAP, extract_patch_tokens, load_backbone, load_backbone_state, load_config
from .preprocess import iter_image_paths, load_image


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _label(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(out, text, (7, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _pca_rgb(patch_tokens, h, w):
    x = patch_tokens[0].detach().float().cpu().numpy()
    x = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    rgb = x @ vt[:3].T
    rgb = rgb.reshape(h, w, 3)
    lo = np.percentile(rgb, 1, axis=(0, 1), keepdims=True)
    hi = np.percentile(rgb, 99, axis=(0, 1), keepdims=True)
    rgb = (rgb - lo) / np.maximum(hi - lo, 1e-6)
    rgb = np.clip(rgb, 0, 1)
    return (rgb * 255).astype(np.uint8)


def _save_rgb(path, img_rgb):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))


def run(args):
    dtype = DTYPE_MAP[args.dtype]
    cfg = load_config(args.config_file)
    ckpt = load_backbone_state(args.ckpt)
    backbone, _ = load_backbone(cfg, ckpt, device=args.device, dtype=dtype)
    del ckpt

    paths = list(iter_image_paths(args.input))
    if args.max_images:
        paths = paths[: args.max_images]
    if not paths:
        raise FileNotFoundError(f"no images found under {args.input}")
    print(f"[pca_demo] images={len(paths)} size={args.size} dtype={args.dtype} out={args.out}")

    out_dir = Path(args.out)
    for idx, path in enumerate(paths):
        img_norm, img_rgb, (H, W) = load_image(path, size=args.size, patch_size=backbone.patch_size, mode=args.mode)
        patch_tokens, (h, w) = extract_patch_tokens(backbone, img_norm, args.device, dtype)
        pca = _pca_rgb(patch_tokens, h, w)
        pca_up = cv2.resize(pca, (W, H), interpolation=cv2.INTER_NEAREST)
        panel = np.concatenate([_label(img_rgb, f"input {H}x{W}"), _label(pca_up, f"patch PCA {h}x{w}")], axis=1)
        stem = Path(path).stem
        _save_rgb(out_dir / f"{stem}_pca.png", pca_up)
        _save_rgb(out_dir / f"{stem}_panel.png", panel)
        print(f"[pca_demo] {idx + 1}/{len(paths)} {path} -> {out_dir / (stem + '_panel.png')}")
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()


def build_parser():
    ap = argparse.ArgumentParser(prog="python -m lingbot_vision.pca_demo")
    ap.add_argument("--config-file", default=str(Path(__file__).parent / "configs" / "lbot_vision_vitl.yaml"))
    ap.add_argument("--ckpt", required=True, help="pure backbone .pt path, e.g. /path/to/model.pt")
    ap.add_argument("--input", default=str(_repo_root() / "examples" / "example.png"))
    ap.add_argument("--out", default=str(_repo_root() / "outputs" / "pca_demo"))
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument(
        "--mode",
        default="square",
        choices=["square", "shortest"],
        help=(
            "square: resize to size x size (does not preserve aspect ratio); "
            "shortest: resize the shortest side to size, then center-crop a size x size square"
        ),
    )
    ap.add_argument("--dtype", default="bf16", choices=sorted(DTYPE_MAP))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-images", type=int, default=0)
    return ap


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
