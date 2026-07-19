"""LingBot-Vision 视频涂抹跟踪 demo.

第一帧鼠标涂抹目标对象 → 逐帧 cosine token 匹配 → 输出带 mask 叠加的 mp4。

用法:
    python demo_video_scribble.py --video 你的视频.mp4 --output demo_output.mp4
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from lingbot_vision import extract_patch_tokens, load_pretrained_backbone
from lingbot_vision.preprocess import IMAGENET_MEAN, IMAGENET_STD


def parse_args():
    ap = argparse.ArgumentParser(description="LingBot-Vision 视频涂抹跟踪 demo")
    ap.add_argument("--video", required=True, help="输入视频路径")
    ap.add_argument("--model", default="models/", help="模型路径（默认 models/）")
    ap.add_argument("--variant", default="small", help="模型变体（默认 small）")
    ap.add_argument("--output", default="outputs/output.mp4", help="输出 mp4 路径")
    ap.add_argument("--size", type=int, default=512, help="处理尺寸（默认 512）")
    ap.add_argument("--threshold", type=float, default=0.85, help="相似度阈值（默认 0.85）")
    ap.add_argument("--brush", type=int, default=30, help="画笔粗细像素（默认 15）")
    ap.add_argument("--device", default="auto", help="设备（auto/cuda/cpu，默认 auto）")
    ap.add_argument("--dtype", default="auto", choices=["auto", "fp32", "bf16", "fp16"])
    ap.add_argument("--no-preview", action="store_true", help="禁用实时预览窗口（无头/远程环境使用）")
    return ap.parse_args()


def _snap(size, patch_size):
    """对齐到 patch_size 的倍数，与 lingbot_vision.preprocess._snap 一致。"""
    return max(patch_size, (size // patch_size) * patch_size)


def frame_to_norm(frame_bgr, size, patch_size, device):
    """BGR 帧 → ImageNet 归一化张量 [1, 3, size, size]。

    复用 lingbot_vision.preprocess 的 IMAGENET_MEAN / IMAGENET_STD 常量，
    逻辑等价于 load_image 但从 cv2 BGR numpy 输入而非 PIL 文件路径。
    """
    size = _snap(size, patch_size)
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    img_t = torch.from_numpy(resized.astype(np.float32) / 255.0)
    img_t = img_t.permute(2, 0, 1).unsqueeze(0)
    img_norm = (img_t - IMAGENET_MEAN) / IMAGENET_STD
    return img_norm, size


# ---------------------------------------------------------------------------
# 鼠标涂抹交互
# ---------------------------------------------------------------------------

def select_scribble(frame_bgr, size, brush):
    """弹窗显示第一帧（按原视频比例），鼠标涂抹目标，Enter 确认。

    显示画布按原视频宽高比缩放（长边 = size），mask 内部保持 size×size 正方形
    以对齐 backbone 的 patch 网格。鼠标坐标自动映射到 mask 空间。
    返回 [size, size] uint8 涂抹 mask（255=涂抹, 0=背景）。
    """
    H0, W0 = frame_bgr.shape[:2]
    # 显示画布：长边 = size，短边按原视频比例
    if W0 >= H0:
        disp_w, disp_h = size, max(1, int(size * H0 / W0))
    else:
        disp_h, disp_w = size, max(1, int(size * W0 / H0))
    base = cv2.resize(frame_bgr, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)
    mask = np.zeros((size, size), dtype=np.uint8)
    drawing = [False]
    last_pt = [None]

    # 显示坐标 → mask 坐标（mask 是 size×size 正方形）
    sx = size / disp_w
    sy = size / disp_h

    def to_mask(pt):
        return (int(pt[0] * sx), int(pt[1] * sy))

    def render():
        # mask 半透明叠加到显示画布，保证看到的红色 = 实际选中的 patch 区域
        mask_disp = cv2.resize(mask, (disp_w, disp_h), interpolation=cv2.INTER_NEAREST)
        overlay = base.copy()
        red = np.array([0, 0, 255], dtype=np.uint8)
        blend = (overlay.astype(np.float32) * 0.5 + red.astype(np.float32) * 0.5).astype(np.uint8)
        overlay[mask_disp > 0] = blend[mask_disp > 0]
        return overlay

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing[0] = True
            mp = to_mask((x, y))
            last_pt[0] = mp
            cv2.circle(mask, mp, max(1, brush // 2), 255, -1)
        elif event == cv2.EVENT_MOUSEMOVE and drawing[0]:
            mp = to_mask((x, y))
            if last_pt[0] is not None:
                cv2.line(mask, last_pt[0], mp, 255, brush)
            last_pt[0] = mp
        elif event == cv2.EVENT_LBUTTONUP:
            drawing[0] = False
            last_pt[0] = None

    win = "scribble - 鼠标涂抹目标, Enter 确认, ESC 退出"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, on_mouse)
    print("[demo] 在弹窗里用鼠标涂抹要跟踪的对象，涂好后按 Enter 确认，ESC 退出")
    while True:
        cv2.imshow(win, render())
        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # Enter
            break
        if key == 27:  # ESC
            cv2.destroyWindow(win)
            raise SystemExit("用户取消")
    cv2.destroyWindow(win)

    if mask.sum() == 0:
        raise SystemExit("涂抹为空，请重新运行并涂抹目标")
    return mask


# ---------------------------------------------------------------------------
# Query 构建
# ---------------------------------------------------------------------------

def build_query_tokens(backbone, frame_bgr, scribble_mask, size, patch_size, device, dtype):
    """涂抹 mask 下采样到 patch 级 → 提取选中 patch 的 token → [Nq, C]。"""
    h = w = size // patch_size
    # 像素级 mask [size, size] → patch 级 [h, w]，每个 patch 内涂抹像素占比 > 0.5 即选中
    mask_down = cv2.resize(scribble_mask, (w, h), interpolation=cv2.INTER_AREA)
    patch_selected = (mask_down > 127).reshape(-1)  # [h*w]

    img_norm, _ = frame_to_norm(frame_bgr, size, patch_size, device)
    patch_tokens, _ = extract_patch_tokens(backbone, img_norm, device, dtype)  # [1, h*w, C]
    tokens = patch_tokens[0]  # [h*w, C]
    query = tokens[patch_selected]  # [Nq, C]
    print(f"[demo] 选中 {int(patch_selected.sum())} 个 patch 作为 query，dim={tokens.shape[-1]}")
    return query


# ---------------------------------------------------------------------------
# 匹配
# ---------------------------------------------------------------------------

def match_tokens(query_tokens, frame_tokens, h, w, threshold):
    """cosine sim → max → 阈值化 → [h, w] bool。"""
    q = torch.nn.functional.normalize(query_tokens, dim=-1)   # [Nq, C]
    k = torch.nn.functional.normalize(frame_tokens, dim=-1)   # [N, C]
    sim = k @ q.T                                               # [N, Nq]
    max_sim, _ = sim.max(dim=1)                                 # [N]
    patch_mask = (max_sim > threshold).cpu().numpy().reshape(h, w)
    return patch_mask


# ---------------------------------------------------------------------------
# 渲染
# ---------------------------------------------------------------------------

def draw_overlay(frame_bgr, patch_mask):
    """patch mask 上采样到原帧尺寸 → 红色半透明填充 + 轮廓 → 叠加帧。"""
    H, W = frame_bgr.shape[:2]
    mask_up = cv2.resize(
        patch_mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST
    )
    # 形态学膨胀平滑 patch 边缘锯齿
    kernel = np.ones((5, 5), np.uint8)
    mask_up = cv2.dilate(mask_up, kernel, iterations=2)

    overlay = frame_bgr.copy()
    red = np.array([0, 0, 255], dtype=np.uint8)  # BGR 红
    overlay[mask_up == 1] = (
        overlay[mask_up == 1].astype(np.float32) * 0.5 + red.astype(np.float32) * 0.5
    ).astype(np.uint8)

    # 轮廓线
    contours, _ = cv2.findContours(mask_up, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2)
    return overlay


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # 设备与 dtype 自动检测（与 test.py 逻辑一致）
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if args.dtype == "auto":
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    else:
        dtype_map = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
        dtype = dtype_map[args.dtype]
    print(f"[demo] device={device} dtype={dtype}")

    print(f"[demo] 加载模型 {args.model} variant={args.variant} ...")
    backbone, embed_dim = load_pretrained_backbone(
        args.model,
        variant=args.variant,
        device=device,
        dtype=dtype,
    )
    patch_size = backbone.patch_size
    size = _snap(args.size, patch_size)
    print(f"[demo] patch_size={patch_size} size={size} embed_dim={embed_dim}")

    # 打开视频
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"无法打开视频: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 读第一帧
    ok, first_frame = cap.read()
    if not ok:
        raise SystemExit("无法读取第一帧")

    # 涂抹交互
    scribble_mask = select_scribble(first_frame, size, args.brush)

    # 构建 query tokens
    query_tokens = build_query_tokens(
        backbone, first_frame, scribble_mask, size, patch_size, device, dtype
    )

    # 准备输出
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    H, W = first_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H))
    if not writer.isOpened():
        raise SystemExit(f"无法创建输出视频: {out_path}")

    # 逐帧处理（从第一帧开始）
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        img_norm, _ = frame_to_norm(frame, size, patch_size, device)
        patch_tokens, (h, w) = extract_patch_tokens(backbone, img_norm, device, dtype)
        patch_mask = match_tokens(query_tokens, patch_tokens[0], h, w, args.threshold)
        out_frame = draw_overlay(frame, patch_mask)
        writer.write(out_frame)

        # 实时预览：处理一帧显示一帧，ESC 可中断（已处理帧仍会保存）
        if not args.no_preview:
            cv2.imshow("lingbot-vision demo - 处理中, ESC 中断", out_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                print("[demo] 用户中断，保存已处理帧 ...")
                break

        if frame_idx % 10 == 0 or frame_idx == 1:
            print(f"[demo] 处理 {frame_idx}/{total if total > 0 else '?'} 帧")
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    writer.release()
    cap.release()
    if not args.no_preview:
        cv2.destroyAllWindows()
    print(f"[demo] 完成，共 {frame_idx} 帧 → {out_path}")


if __name__ == "__main__":
    main()