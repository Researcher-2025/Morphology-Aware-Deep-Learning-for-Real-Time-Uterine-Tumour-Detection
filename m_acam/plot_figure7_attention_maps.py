"""
Figure 7: Attention map visualization (2×2) — Grad-CAM on segmentation decoder features,
pseudo–radiologist gaze from Gaussian-smoothed GT mask (development), optional ROI circle on (a).

Section 5.13 style layout:
  (a) Round / (b) Elongated — native grayscale + title with mask eccentricity (and optional size).
  (c) Calcified — same as other panels (pick a calcified case in your data).
  (d) Grad-CAM jet overlay + caption with Pearson r and IoU vs pseudo-gaze.

Requires: checkpoint, dataset, four dataset indices (test split by default).

Usage:
  python -m m_acam.plot_figure7_attention_maps \\
    --checkpoint checkpoints/best_model.pt \\
    --dataset-root Dataset \\
    --indices 0,12,3,0 \\
    --out checkpoints/figures/figure7_attention.png
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from m_acam.dataset import UterineFibroidDataset, mask_eccentricity
from m_acam.model import MACAM
from m_acam.utils import set_global_seed

try:
    from scipy.stats import pearsonr
except Exception:  # pragma: no cover
    pearsonr = None


def _minmax(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = x.astype(np.float64)
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < eps:
        return np.zeros_like(x, dtype=np.float64)
    return (x - lo) / (hi - lo + eps)


def seg_grad_cam(
    model: MACAM,
    image: torch.Tensor,
    mask: torch.Tensor,
) -> np.ndarray:
    """Grad-CAM w.r.t. decoder features for segmentation BCE loss (full-res logits vs mask)."""
    model.eval()
    image = image.detach()
    mask = mask.detach()

    model.zero_grad(set_to_none=True)
    out = model(image)
    feat = out["decoder_feat"]
    feat.retain_grad()
    logits = out["seg_logits"]
    loss = F.binary_cross_entropy_with_logits(logits, mask)
    loss.backward()

    grad = feat.grad
    if grad is None:
        raise RuntimeError("Grad-CAM: decoder_feat.grad is None")

    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = (weights * feat).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = cam.squeeze(0).squeeze(0)
    cam = cam.detach().cpu().float().numpy()
    cam = _minmax(cam).astype(np.float32)
    h, w = int(image.shape[2]), int(image.shape[3])
    cam_up = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
    return cam_up.astype(np.float32)


def pseudo_gaze_from_mask(mask_hw: np.ndarray, sigma: float = 12.0) -> np.ndarray:
    """Smooth binary mask → pseudo fixation map [0, 1]."""
    m = (mask_hw > 0.5).astype(np.float32)
    if m.max() < 0.5:
        return np.zeros_like(m, dtype=np.float32)
    k = int(max(3, round(6 * sigma / 20)))
    if k % 2 == 0:
        k += 1
    blur = cv2.GaussianBlur(m, (k, k), sigmaX=sigma, sigmaY=sigma)
    return _minmax(blur).astype(np.float32)


def mask_principal_axes_px(mask: np.ndarray) -> Tuple[float, float]:
    """Approximate major × minor diameter in pixels (ellipse fit)."""
    ys, xs = np.where(mask > 0.5)
    if len(xs) < 5:
        return 0.0, 0.0
    pts = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=0)
    mean = pts.mean(axis=1, keepdims=True)
    c = np.cov(pts - mean)
    ev = np.linalg.eigvalsh(c)
    ev = np.sort(np.clip(ev, 1e-6, None))
    # 4*sqrt(lambda) ~ axis length for uniform ellipse (order-of-magnitude for caption)
    a = 4.0 * math.sqrt(ev[1])
    b = 4.0 * math.sqrt(ev[0])
    return float(max(a, b)), float(min(a, b))


def draw_roi_circle(img: np.ndarray, mask: np.ndarray, color: float = 1.0, thickness: int = 2) -> np.ndarray:
    """White circle at mask centroid; radius from equivalent disk area."""
    out = img.copy()
    ys, xs = np.where(mask > 0.5)
    if len(xs) == 0:
        return out
    cy = float(ys.mean())
    cx = float(xs.mean())
    area = float((mask > 0.5).sum())
    r = math.sqrt(max(area, 1.0) / math.pi) * 0.95
    h, w = out.shape[:2]
    cv2.circle(out, (int(round(cx)), int(round(cy))), int(round(r)), color, thickness=thickness, lineType=cv2.LINE_AA)
    return np.clip(out, 0.0, 1.0)


def gaze_iou(cam01: np.ndarray, gaze01: np.ndarray, thr: float = 0.5) -> float:
    a = (cam01 >= thr) & (gaze01 >= thr)
    b = (cam01 >= thr) | (gaze01 >= thr)
    inter = float(a.sum())
    union = float(b.sum())
    return inter / max(union, 1e-8)


def gaze_correlation(cam01: np.ndarray, gaze01: np.ndarray) -> float:
    a = cam01.reshape(-1).astype(np.float64)
    b = gaze01.reshape(-1).astype(np.float64)
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return float("nan")
    if pearsonr is not None:
        try:
            r, _ = pearsonr(a, b)
            return float(r)
        except Exception:
            return float("nan")
    c = np.corrcoef(a, b)[0, 1]
    return float(c) if np.isfinite(c) else float("nan")


def tensor_image_to_gray01(img_b1hw: torch.Tensor) -> np.ndarray:
    return img_b1hw.squeeze().detach().cpu().numpy().astype(np.float32)


def load_model(ckpt: str, device: torch.device) -> MACAM:
    m = MACAM(pretrained=False, num_det_classes=1).to(device)
    w = torch.load(ckpt, map_location=device)
    m.load_state_dict(w["model"], strict=False)
    m.eval()
    return m


def build_caption_a_b(
    tag: str,
    ecc: float,
    mask: np.ndarray,
    size_cm: float,
) -> str:
    if tag.lower() == "a":
        if size_cm > 0:
            return f"(a) Round fibroid (e={ecc:.2f}, {size_cm:.1f}cm)"
        return f"(a) Round fibroid (e={ecc:.2f})"
    maj, minr = mask_principal_axes_px(mask)
    if maj > 1 and minr > 1:
        return f"(b) Elongated fibroid (e={ecc:.2f}, {maj:.1f}×{minr:.1f}px)"
    return f"(b) Elongated fibroid (e={ecc:.2f})"


def overlay_jet(gray01: np.ndarray, heat01: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    try:
        cmap = plt.get_cmap("jet")
    except Exception:
        cmap = plt.cm.jet
    h = _minmax(heat01)
    rgba = cmap(h)
    rgb = rgba[:, :, :3].astype(np.float32)
    g = np.stack([gray01, gray01, gray01], axis=-1)
    out = (1.0 - alpha) * g + alpha * rgb
    return np.clip(out, 0.0, 1.0)


def plot_figure7(
    model: MACAM,
    ds: UterineFibroidDataset,
    indices: Tuple[int, int, int, int],
    out_path: Path,
    device: torch.device,
    circle_on_a: bool = True,
    dpi: int = 200,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 9.5))
    tags = ["a", "b", "c", "d"]

    for ax, idx, letter in zip(axes.flat, indices, tags):
        sample = ds[int(idx)]
        img = sample["image"].unsqueeze(0).to(device)
        mask = sample["mask"].unsqueeze(0).to(device)
        gray = tensor_image_to_gray01(img)
        m_np = mask.squeeze().cpu().numpy().astype(np.float32)

        ecc = mask_eccentricity((m_np > 0.5).astype(np.uint8))
        morph = ds.samples[int(idx)].get("morphology", {}) or {}
        size_cm = float(morph.get("size_cm", 0.0) or 0.0)

        if letter == "a":
            cap = build_caption_a_b("a", ecc, m_np, size_cm)
            disp = draw_roi_circle(gray, m_np) if circle_on_a else gray
            ax.imshow(disp, cmap="gray", vmin=0, vmax=1)
            ax.set_title(cap, fontsize=10)
        elif letter == "b":
            cap = build_caption_a_b("b", ecc, m_np, size_cm)
            ax.imshow(gray, cmap="gray", vmin=0, vmax=1)
            ax.set_title(cap, fontsize=10)
        elif letter == "c":
            ax.imshow(gray, cmap="gray", vmin=0, vmax=1)
            ax.set_title("(c) Calcified fibroid", fontsize=10)
        else:
            with torch.set_grad_enabled(True):
                cam = seg_grad_cam(model, img, mask)
            gaze = pseudo_gaze_from_mask(m_np)
            cam_n = _minmax(cam).astype(np.float32)
            r = gaze_correlation(cam_n, gaze)
            iou = gaze_iou(cam_n, gaze)
            ov = overlay_jet(gray, cam)
            ax.imshow(ov)
            r_txt = f"{r:.2f}" if np.isfinite(r) else "nan"
            ax.set_title(
                rf"(d) Pseudo–radiologist gaze vs Grad-CAM ($r={r_txt}$, IoU={100.0 * iou:.1f}\%)",
                fontsize=9,
            )

        ax.axis("off")

    fig.suptitle("Figure 7: Attention Map Visualization — 4 panels", fontsize=12, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Figure 7 Grad-CAM + pseudo-gaze panels")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--dataset-root", type=str, default="Dataset")
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--indices", type=str, default="0,0,0,0", help="Comma-separated four dataset indices (a,b,c,d)")
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--test-ratio", type=float, default=0.1)
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--out", type=str, default="checkpoints/figures/figure7_attention_maps.png")
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--no-circle-a", action="store_true", help="Disable ROI circle on panel (a)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parts = [int(x.strip()) for x in args.indices.split(",")]
    if len(parts) != 4:
        raise SystemExit("--indices must have exactly four integers: a,b,c,d")
    idx_tuple = (parts[0], parts[1], parts[2], parts[3])

    ds = UterineFibroidDataset.from_voc_folder(
        dataset_root=args.dataset_root,
        split=args.split,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        image_size=args.image_size,
        train=False,
        apply_augmentation=False,
    )
    for i in idx_tuple:
        if i < 0 or i >= len(ds):
            raise SystemExit(f"Index {i} out of range for dataset length {len(ds)}")

    model = load_model(args.checkpoint, device)
    plot_figure7(
        model,
        ds,
        idx_tuple,
        Path(args.out),
        device,
        circle_on_a=not args.no_circle_a,
        dpi=args.dpi,
    )
    print(f"Wrote {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
