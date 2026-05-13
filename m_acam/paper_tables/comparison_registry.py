"""
Register models for Table-style comparison reports.

Extend this file for your paper: add a callable that returns an instantiated
torch.nn.Module for each baseline you actually implement. If `builder` is
None, the report will skip that row with a warning (until you wire it).

Keep names/types/sources aligned with your manuscript citations.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional

import torch.nn as nn

from m_acam.model import MACAM


@dataclass(frozen=True)
class ModelComparisonSpec:
    key: str
    display_name: str
    task_type: str
    source: str
    builder: Optional[Callable[[], nn.Module]]


def _macam() -> nn.Module:
    return MACAM(pretrained=False, num_det_classes=1)


def _efficientnet_b0_baseline() -> nn.Module:
    """ImageNet EfficientNet-B0 backbone (classification). Your seg head adds params."""
    from torchvision.models import efficientnet_b0

    return efficientnet_b0(weights=None)


# ---------------------------------------------------------------------------
# Edit below: set `builder` to your implemented baseline, or None if not yet.
# Example:
#   def _my_unet() -> nn.Module:
#       from baselines.unet import UNet
#       return UNet(in_channels=1, num_classes=1)
# ---------------------------------------------------------------------------

DEFAULT_COMPARISON_REGISTRY: List[ModelComparisonSpec] = [
    ModelComparisonSpec(
        key="yolov3",
        display_name="YOLOv3 [1], [7]",
        task_type="Detection only",
        source="Huo et al. implementation",
        builder=None,
    ),
    ModelComparisonSpec(
        key="efficientnet_b0",
        display_name="EfficientNet-B0 [22], [28]",
        task_type="Classification + Segmentation",
        source="Xi & Wang implementation (use your repo's class if different)",
        builder=_efficientnet_b0_baseline,
    ),
    ModelComparisonSpec(
        key="unet",
        display_name="U-Net [17]",
        task_type="Segmentation",
        source="Standard implementation",
        builder=None,
    ),
    ModelComparisonSpec(
        key="transunet",
        display_name="TransUNet [4]",
        task_type="Segmentation",
        source="Official repository",
        builder=None,
    ),
    ModelComparisonSpec(
        key="swin_unet",
        display_name="Swin-Unet [3]",
        task_type="Segmentation",
        source="Official repository",
        builder=None,
    ),
    ModelComparisonSpec(
        key="macam",
        display_name="M-ACAM (Ours)",
        task_type="Multi-task",
        source="This work",
        builder=_macam,
    ),
]
