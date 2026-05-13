"""Save one dataset sample: preprocessed image + mask (+ optional overlay) for QA."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from m_acam.dataset import UterineFibroidDataset, mask_eccentricity


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Inspect one training/val/test sample after dataset preprocessing")
    p.add_argument("--dataset-root", type=str, default="Dataset")
    p.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--test-ratio", type=float, default=0.1)
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--save", type=str, default="", help="Optional path to save PNG (e.g. checkpoints/debug/sample.png)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    train_mode = args.split == "train"
    ds = UterineFibroidDataset.from_voc_folder(
        dataset_root=args.dataset_root,
        split=args.split,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=42,
        image_size=args.image_size,
        train=train_mode,
        apply_augmentation=train_mode,
    )
    if args.index < 0 or args.index >= len(ds):
        raise SystemExit(f"index {args.index} out of range [0, {len(ds) - 1}]")

    s = ds[args.index]
    img = s["image"].squeeze().numpy()
    mask = s["mask"].squeeze().numpy()
    ecc = float(s["eccentricity"].item())
    ecc_mask = mask_eccentricity((mask > 0.5).astype(np.uint8))
    morph = ds.samples[args.index].get("morphology", {})

    print("=== Preprocessed sample ===")
    print(f"image_id:     {s['image_id']}")
    print(f"image shape:  {tuple(s['image'].shape)}  dtype={s['image'].dtype}  min={img.min():.4f} max={img.max():.4f}")
    print(f"mask shape:   {tuple(s['mask'].shape)}  fg_pixels={(mask > 0.5).sum()}")
    print(f"eccentricity (tensor): {ecc:.4f}  (from mask recompute): {ecc_mask:.4f}")
    print(f"morphology:   {morph}")
    print(f"shape_label:  {int(s['shape_label'].item())}")
    print(f"figo_label:   {int(s['figo_label'].item())}")
    print(f"size_cm:      {float(s['size_cm'].item()):.4f}")

    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        ax[0].imshow(img, cmap="gray", vmin=0, vmax=1)
        ax[0].set_title("Preprocessed image")
        ax[1].imshow(mask, cmap="gray", vmin=0, vmax=1)
        ax[1].set_title("Mask (from bbox)")
        ov = np.stack([img, img, img], axis=-1)
        ov = np.clip(ov + 0.35 * np.stack([mask, mask * 0.2, mask * 0.2], axis=-1), 0, 1)
        ax[2].imshow(ov)
        ax[2].set_title("Overlay")
        for a in ax:
            a.axis("off")
        fig.suptitle(str(s["image_id"]), fontsize=10)
        fig.tight_layout()
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out.resolve()}")


if __name__ == "__main__":
    main()
