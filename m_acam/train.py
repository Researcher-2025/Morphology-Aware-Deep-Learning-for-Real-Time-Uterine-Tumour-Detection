import argparse
import csv
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from tqdm import tqdm

from m_acam.dataset import UterineFibroidDataset, collate_fn, write_split_files
from m_acam.loss import MultiTaskLoss
from m_acam.model import MACAM
from m_acam.utils import AverageMeter, EarlyStopping, dice_coefficient, seed_worker, set_global_seed, to_device


def segmentation_pixel_accuracy(seg_logits: torch.Tensor, seg_target: torch.Tensor) -> float:
    pred = (torch.sigmoid(seg_logits) > 0.5).float()
    target = (seg_target > 0.5).float()
    correct = (pred == target).float().mean()
    return float(correct.item())


def build_loaders(dataset_root: str, val_ratio: float, test_ratio: float, batch_size: int = 16, num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:
    train_ds = UterineFibroidDataset.from_voc_folder(
        dataset_root=dataset_root,
        split="train",
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=42,
        train=True,
        apply_augmentation=True,
    )
    val_ds = UterineFibroidDataset.from_voc_folder(
        dataset_root=dataset_root,
        split="val",
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=42,
        train=False,
        apply_augmentation=False,
    )
    g = torch.Generator()
    g.manual_seed(42)
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
        generator=g,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
        generator=g,
    )
    return train_loader, val_loader


def figo_weights_from_dataset(dataset: UterineFibroidDataset) -> torch.Tensor:
    labels = [int(r.get("morphology", {}).get("figo", 0)) for r in dataset.samples]
    counts = np.bincount(np.array(labels, dtype=np.int64), minlength=8).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float32)


def set_trainable_params(model: MACAM, epoch: int, base_lr: float) -> Dict:
    if epoch <= 10:
        for p in model.stem.parameters():
            p.requires_grad = False
        for layer in [model.layer1, model.layer2, model.layer3, model.layer4]:
            for p in layer.parameters():
                p.requires_grad = False
        param_groups = [
            {"params": model.seg_head.parameters(), "lr": base_lr},
            {"params": model.det_head.parameters(), "lr": base_lr},
            {"params": model.morph_head.parameters(), "lr": base_lr},
        ]
    else:
        for p in model.parameters():
            p.requires_grad = True
        backbone_params = list(model.stem.parameters()) + list(model.layer1.parameters()) + list(model.layer2.parameters())
        decoder_params = list(model.layer3.parameters()) + list(model.layer4.parameters()) + list(model.seg_head.parameters())
        head_params = list(model.det_head.parameters()) + list(model.morph_head.parameters())
        param_groups = [
            {"params": backbone_params, "lr": base_lr * 0.1},
            {"params": decoder_params, "lr": base_lr * 0.5},
            {"params": head_params, "lr": base_lr * 1.0},
        ]
    return param_groups


def evaluate(
    model: MACAM,
    loader: DataLoader,
    criterion: MultiTaskLoss,
    device: torch.device,
    max_batches: int = None,
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    loss_meter = AverageMeter()
    dice_meter = AverageMeter()
    acc_meter = AverageMeter()
    with torch.no_grad():
        for i, batch in enumerate(loader, start=1):
            batch = to_device(batch, device)
            outputs = model(batch["image"])
            loss, loss_dict = criterion(outputs, batch)
            seg_probs = torch.sigmoid(outputs["seg_logits"])
            dice = dice_coefficient(seg_probs, batch["mask"])
            acc = segmentation_pixel_accuracy(outputs["seg_logits"], batch["mask"])
            loss_meter.update(loss.item(), n=batch["image"].size(0))
            dice_meter.update(dice.item(), n=batch["image"].size(0))
            acc_meter.update(acc, n=batch["image"].size(0))
            if max_batches is not None and i >= max_batches:
                break
    return dice_meter.avg, {"val_loss": loss_meter.avg, "val_dice": dice_meter.avg, "val_accuracy": acc_meter.avg}


def train(args: argparse.Namespace) -> None:
    set_global_seed(42)
    cv2.setNumThreads(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    write_split_files(args.dataset_root, str(Path(args.out_dir) / "splits"), val_ratio=args.val_ratio, test_ratio=args.test_ratio, seed=42)

    train_loader, val_loader = build_loaders(
        args.dataset_root,
        args.val_ratio,
        args.test_ratio,
        args.batch_size,
        args.num_workers,
    )
    figo_weights = figo_weights_from_dataset(train_loader.dataset).to(device)
    model = MACAM(pretrained=True, num_det_classes=1).to(device)
    criterion = MultiTaskLoss(anchors=model.anchors, figo_class_weights=figo_weights, image_size=512).to(device)

    optimizer = AdamW(set_trainable_params(model, 1, args.lr), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=1, eta_min=1e-6)
    patience = args.early_stop_patience if args.epochs > 1 else 10**9
    early_stopper = EarlyStopping(patience=patience)

    train_len = len(train_loader)
    train_steps_eff = train_len if args.max_train_batches is None else min(train_len, args.max_train_batches)
    val_len = len(val_loader)
    val_steps_eff = val_len if args.max_val_batches is None else min(val_len, args.max_val_batches)

    global_step = 0
    history_path = Path(args.out_dir) / "training_history.csv"
    with open(history_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_accuracy", "val_accuracy", "train_loss", "val_loss", "val_dice"])

    for epoch in range(1, args.epochs + 1):
        if epoch == 11:
            optimizer = AdamW(set_trainable_params(model, epoch, args.lr), lr=args.lr, weight_decay=1e-4)
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=1, eta_min=1e-6)

        model.train()
        loss_meter = AverageMeter()
        train_acc_meter = AverageMeter()
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False, total=train_steps_eff)
        for step, batch in enumerate(pbar, start=1):
            batch = to_device(batch, device)
            outputs = model(batch["image"])
            loss, loss_dict = criterion(outputs, batch)
            train_acc = segmentation_pixel_accuracy(outputs["seg_logits"], batch["mask"])
            loss = loss / args.accumulation_steps
            loss.backward()

            if step % args.accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            global_step += 1
            scheduler.step(epoch - 1 + step / max(train_steps_eff, 1))
            loss_meter.update(loss_dict["total"], n=batch["image"].size(0))
            train_acc_meter.update(train_acc, n=batch["image"].size(0))
            pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}", "lr": f"{optimizer.param_groups[-1]['lr']:.6e}"})
            if args.max_train_batches is not None and step >= args.max_train_batches:
                break

        if step % args.accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        val_dice, val_metrics = evaluate(
            model, val_loader, criterion, device, max_batches=args.max_val_batches
        )
        improved = early_stopper.step(val_dice)

        if epoch % 5 == 0:
            ckpt_path = Path(args.out_dir) / f"checkpoint_epoch_{epoch}.pt"
            torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict()}, ckpt_path)

        if improved:
            best_path = Path(args.out_dir) / "best_model.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "val_dice": val_dice,
                    "metrics": val_metrics,
                },
                best_path,
            )

        with open(history_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    epoch,
                    round(train_acc_meter.avg * 100.0, 4),
                    round(val_metrics["val_accuracy"] * 100.0, 4),
                    round(loss_meter.avg, 6),
                    round(val_metrics["val_loss"], 6),
                    round(val_dice, 6),
                ]
            )

        print(
            f"Epoch {epoch}: train_acc={train_acc_meter.avg*100.0:.2f}% | "
            f"val_acc={val_metrics['val_accuracy']*100.0:.2f}% | "
            f"train_loss={loss_meter.avg:.4f} | val_loss={val_metrics['val_loss']:.4f} | val_dice={val_dice:.4f}"
        )
        if early_stopper.should_stop:
            print(f"Early stopping at epoch {epoch} (patience={early_stopper.patience}).")
            break


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("M-ACAM Training")
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="Dataset",
        help="Root dataset folder containing Annotations/Annotations and JPEGImages/JPEGImages.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test split ratio (held out from training).")
    parser.add_argument("--out-dir", type=str, default="./checkpoints", help="Output directory for checkpoints.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--accumulation-steps", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help="Cap training steps per epoch (smoke test / time budget). None = full epoch.",
    )
    parser.add_argument(
        "--max-val-batches",
        type=int,
        default=None,
        help="Cap validation batches per epoch (faster val). None = full val set.",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=20,
        help="Early stopping patience on val Dice. Use a large value with --epochs 1.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
