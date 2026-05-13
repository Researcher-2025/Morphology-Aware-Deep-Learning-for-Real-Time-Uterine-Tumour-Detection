import math
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import SimpleITK as sitk
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


def compute_snr_db(img: np.ndarray) -> float:
    signal = float(np.mean(img))
    noise = float(np.std(img) + 1e-8)
    return 20.0 * np.log10((signal + 1e-8) / noise)


def minmax_normalize(img: np.ndarray) -> np.ndarray:
    vmin = float(img.min())
    vmax = float(img.max())
    if vmax - vmin < 1e-8:
        return np.zeros_like(img, dtype=np.float32)
    return ((img - vmin) / (vmax - vmin)).astype(np.float32)


def mask_eccentricity(mask: np.ndarray) -> float:
    ys, xs = np.where(mask > 0)
    if len(xs) < 5:
        return 0.0
    coords = np.stack([xs, ys], axis=1).astype(np.float32)
    cov = np.cov(coords.T)
    eigvals = np.sort(np.linalg.eigvalsh(cov))
    if eigvals[-1] <= 1e-8:
        return 0.0
    ratio = float(eigvals[0] / (eigvals[-1] + 1e-8))
    ecc = math.sqrt(max(0.0, 1.0 - ratio))
    return float(np.clip(ecc, 0.0, 1.0))


def bbox_from_mask(mask: np.ndarray) -> Optional[List[float]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x1, x2 = float(xs.min()), float(xs.max())
    y1, y2 = float(ys.min()), float(ys.max())
    return [x1, y1, x2, y2]


def read_medical_image(path: str) -> np.ndarray:
    path_obj = Path(path)
    if path_obj.suffix.lower() in [".nii", ".nii.gz", ".mhd", ".nrrd"]:
        img = sitk.ReadImage(str(path_obj))
        arr = sitk.GetArrayFromImage(img)
        if arr.ndim == 3:
            arr = arr[arr.shape[0] // 2]
        return arr.astype(np.float32)

    img = cv2.imread(str(path_obj), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.astype(np.float32)


class UterineFibroidDataset(Dataset):
    def __init__(
        self,
        samples: List[Dict],
        image_size: int = 512,
        train: bool = True,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid: Tuple[int, int] = (8, 8),
        apply_augmentation: bool = True,
    ) -> None:
        self.image_size = image_size
        self.train = train
        self.apply_augmentation = apply_augmentation and train
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid = clahe_tile_grid
        self.clahe = None
        self.samples = samples

    @classmethod
    def from_voc_folder(
        cls,
        dataset_root: str,
        split: str,
        val_ratio: float = 0.2,
        test_ratio: float = 0.1,
        seed: int = 42,
        image_size: int = 512,
        train: bool = True,
        apply_augmentation: bool = True,
    ) -> "UterineFibroidDataset":
        all_samples = load_voc_samples(dataset_root)
        split_indices = create_splits(len(all_samples), val_ratio=val_ratio, test_ratio=test_ratio, seed=seed)
        if split not in split_indices:
            raise ValueError(f"Unknown split: {split}. Expected one of {list(split_indices.keys())}.")
        selected_idx = split_indices[split]
        split_samples = [all_samples[i] for i in selected_idx]
        return cls(
            samples=split_samples,
            image_size=image_size,
            train=train,
            apply_augmentation=apply_augmentation,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        if self.clahe is None:
            self.clahe = cv2.createCLAHE(
                clipLimit=self.clahe_clip_limit,
                tileGridSize=self.clahe_tile_grid,
            )
        img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        img = np.clip(img, 0, None)
        if img.max() > 255:
            img = (255.0 * img / (img.max() + 1e-8)).astype(np.uint8)
        else:
            img = img.astype(np.uint8)

        img = self.clahe.apply(img)
        img = minmax_normalize(img)
        snr = compute_snr_db(img)
        if snr < 15.0:
            img_u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
            img_u8 = cv2.bilateralFilter(img_u8, d=5, sigmaColor=50, sigmaSpace=50)
            img = minmax_normalize(img_u8)
        return img

    def _elastic_deform(self, img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h, w = img.shape[:2]
        alpha = 18.0
        sigma = 5.0
        dx = cv2.GaussianBlur((np.random.rand(h, w).astype(np.float32) * 2 - 1), (0, 0), sigma) * alpha
        dy = cv2.GaussianBlur((np.random.rand(h, w).astype(np.float32) * 2 - 1), (0, 0), sigma) * alpha
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (x + dx).astype(np.float32)
        map_y = (y + dy).astype(np.float32)
        img_d = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        mask_d = cv2.remap(mask, map_x, map_y, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101)
        return img_d, mask_d

    def _random_affine(self, img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h, w = img.shape[:2]
        angle = random.uniform(-15.0, 15.0)
        scale = random.uniform(0.9, 1.1)
        tx = random.uniform(-0.1 * w, 0.1 * w)
        ty = random.uniform(-0.1 * h, 0.1 * h)
        center = (w / 2.0, h / 2.0)
        m = cv2.getRotationMatrix2D(center, angle, scale)
        m[0, 2] += tx
        m[1, 2] += ty
        img_t = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        mask_t = cv2.warpAffine(mask, m, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
        return img_t, mask_t

    def _intensity_aug(self, img: np.ndarray) -> np.ndarray:
        alpha = random.uniform(0.85, 1.15)
        beta = random.uniform(-0.10, 0.10)
        img = np.clip(img * alpha + beta, 0.0, 1.0)

        if random.random() < 0.5:
            noise = np.random.normal(0, 0.02, size=img.shape).astype(np.float32)
            img = np.clip(img + noise, 0.0, 1.0)

        gamma = random.uniform(0.8, 1.2)
        img = np.power(np.clip(img, 1e-6, 1.0), gamma)
        return np.clip(img, 0.0, 1.0)

    def augment(self, img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        original_area = float((mask > 0).sum())

        if random.random() < 0.5:
            img = cv2.flip(img, 1)
            mask = cv2.flip(mask, 1)
        if random.random() < 0.9:
            img, mask = self._random_affine(img, mask)
        if random.random() < 0.5:
            img, mask = self._elastic_deform(img, mask)

        img = self._intensity_aug(img)
        new_area = float((mask > 0).sum())
        if original_area > 0 and (new_area / original_area) < 0.7:
            return img, np.zeros_like(mask, dtype=np.uint8)
        return img, (mask > 0).astype(np.uint8)

    def __getitem__(self, idx: int) -> Dict:
        rec = self.samples[idx]
        image = read_medical_image(rec["image"])
        if "mask" in rec and rec["mask"]:
            mask = read_medical_image(rec["mask"]).astype(np.uint8)
        else:
            mask = np.zeros_like(image, dtype=np.uint8)
            box = rec.get("bbox", None)
            if box is not None:
                x1, y1, x2, y2 = [int(v) for v in box]
                x1 = np.clip(x1, 0, mask.shape[1] - 1)
                x2 = np.clip(x2, 0, mask.shape[1] - 1)
                y1 = np.clip(y1, 0, mask.shape[0] - 1)
                y2 = np.clip(y2, 0, mask.shape[0] - 1)
                mask[y1 : y2 + 1, x1 : x2 + 1] = 1

        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        image = self.preprocess(image)
        mask = (mask > 0).astype(np.uint8)

        if self.apply_augmentation:
            image, mask = self.augment(image, mask)

        bbox = bbox_from_mask(mask)
        if bbox is None:
            bbox = [0.0, 0.0, 0.0, 0.0]
            has_obj = 0.0
        else:
            has_obj = 1.0

        eccentricity = mask_eccentricity(mask)

        morphology = rec.get("morphology", {})
        stem = rec.get("image_id", Path(rec["image"]).stem)
        sample = {
            "image_id": stem,
            "image": torch.from_numpy(image).unsqueeze(0).float(),
            "mask": torch.from_numpy(mask).unsqueeze(0).float(),
            "boxes": torch.tensor([bbox], dtype=torch.float32),
            "labels": torch.tensor([1 if has_obj > 0 else 0], dtype=torch.long),
            "has_obj": torch.tensor([has_obj], dtype=torch.float32),
            "shape_label": torch.tensor(morphology.get("shape", 0), dtype=torch.long),
            "size_cm": torch.tensor(float(morphology.get("size_cm", 0.0)), dtype=torch.float32),
            "figo_label": torch.tensor(morphology.get("figo", 0), dtype=torch.long),
            "eccentricity": torch.tensor(eccentricity, dtype=torch.float32),
        }
        return sample


def parse_voc_xml(xml_path: Path, image_dir: Path) -> Optional[Dict]:
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    filename = root.findtext("filename")
    if filename is None:
        return None

    image_path = image_dir / filename
    if not image_path.exists():
        return None

    objects = root.findall("object")
    if not objects:
        return {
            "image": str(image_path),
            "image_id": image_path.stem,
            "bbox": None,
            "morphology": {"shape": 0, "size_cm": 0.0, "figo": 0},
        }

    # Use largest fibroid region as primary target for this implementation.
    best_box = None
    best_area = -1.0
    for obj in objects:
        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        xmin = float(bnd.findtext("xmin", "0"))
        ymin = float(bnd.findtext("ymin", "0"))
        xmax = float(bnd.findtext("xmax", "0"))
        ymax = float(bnd.findtext("ymax", "0"))
        area = max(0.0, xmax - xmin) * max(0.0, ymax - ymin)
        if area > best_area:
            best_area = area
            best_box = [xmin, ymin, xmax, ymax]

    return {
        "image": str(image_path),
        "image_id": image_path.stem,
        "bbox": best_box,
        "morphology": {"shape": 0, "size_cm": 0.0, "figo": 0},
    }


def load_voc_samples(dataset_root: str) -> List[Dict]:
    root = Path(dataset_root)
    ann_dir = root / "Annotations" / "Annotations"
    img_dir = root / "JPEGImages" / "JPEGImages"
    xml_files = sorted(ann_dir.glob("*.xml"))
    if not xml_files:
        raise FileNotFoundError(f"No XML annotation files found in: {ann_dir}")
    all_samples = []
    for xml_file in xml_files:
        rec = parse_voc_xml(xml_file, img_dir)
        if rec is not None:
            all_samples.append(rec)
    return all_samples


def create_splits(total_count: int, val_ratio: float = 0.2, test_ratio: float = 0.1, seed: int = 42) -> Dict[str, List[int]]:
    if total_count <= 0:
        return {"train": [], "val": [], "test": []}
    indices = list(range(total_count))
    train_val_idx, test_idx = train_test_split(indices, test_size=test_ratio, random_state=seed, shuffle=True)
    effective_val_ratio = val_ratio / max(1.0 - test_ratio, 1e-8)
    effective_val_ratio = float(np.clip(effective_val_ratio, 1e-6, 0.999999))
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=effective_val_ratio,
        random_state=seed,
        shuffle=True,
    )
    return {"train": train_idx, "val": val_idx, "test": test_idx}


def write_split_files(dataset_root: str, out_dir: str, val_ratio: float = 0.2, test_ratio: float = 0.1, seed: int = 42) -> None:
    samples = load_voc_samples(dataset_root)
    splits = create_splits(len(samples), val_ratio=val_ratio, test_ratio=test_ratio, seed=seed)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    for split_name, idxs in splits.items():
        with open(out_path / f"{split_name}.txt", "w", encoding="utf-8") as f:
            for idx in idxs:
                image_id = samples[idx].get("image_id", Path(samples[idx]["image"]).stem)
                f.write(f"{image_id}\n")


def collate_fn(batch: List[Dict]) -> Dict:
    keys = batch[0].keys()
    out = {}
    for k in keys:
        if k in ["boxes", "labels", "has_obj", "image_id"]:
            out[k] = [b[k] for b in batch]
        else:
            out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out
