from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


class MorphologyAwareAttentionGate(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.ecc_mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )
        self.round_path = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )
        self.elongated_path = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(1, 7), padding=(0, 3), bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=(7, 1), padding=(3, 0), bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )
        self.gamma = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        desc = F.adaptive_avg_pool2d(x, 1).flatten(1)
        e_hat = self.ecc_mlp(desc)
        e_map = e_hat.view(-1, 1, 1, 1)
        round_feat = self.round_path(x) * (1.0 - e_map)
        elongated_feat = self.elongated_path(x) * e_map
        fused = round_feat + elongated_feat
        out = x + self.gamma * fused
        return out, e_hat


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.up4 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        self.gate3 = MorphologyAwareAttentionGate(1024)
        self.dec3 = ConvBlock(2048, 1024)

        self.up3 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.gate2 = MorphologyAwareAttentionGate(512)
        self.dec2 = ConvBlock(1024, 512)

        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.gate1 = MorphologyAwareAttentionGate(256)
        self.dec1 = ConvBlock(512, 256)

        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec0 = ConvBlock(128, 128)
        self.seg_head = nn.Conv2d(128, 1, kernel_size=1)

    def forward(self, c1: torch.Tensor, c2: torch.Tensor, c3: torch.Tensor, c4: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.up4(c4)
        if x.shape[-2:] != c3.shape[-2:]:
            x = F.interpolate(x, size=c3.shape[-2:], mode="bilinear", align_corners=False)
        g3, e3 = self.gate3(c3)
        x = self.dec3(torch.cat([x, g3], dim=1))

        x = self.up3(x)
        if x.shape[-2:] != c2.shape[-2:]:
            x = F.interpolate(x, size=c2.shape[-2:], mode="bilinear", align_corners=False)
        g2, e2 = self.gate2(c2)
        x = self.dec2(torch.cat([x, g2], dim=1))

        x = self.up2(x)
        if x.shape[-2:] != c1.shape[-2:]:
            x = F.interpolate(x, size=c1.shape[-2:], mode="bilinear", align_corners=False)
        g1, e1 = self.gate1(c1)
        x = self.dec1(torch.cat([x, g1], dim=1))

        x = self.up1(x)
        x = self.dec0(x)
        seg_logits = self.seg_head(x)
        e_hat = torch.stack([e1.squeeze(1), e2.squeeze(1), e3.squeeze(1)], dim=1).mean(dim=1, keepdim=True)
        return {"seg_logits": seg_logits, "decoder_feat": x, "e_hat": e_hat}


class YOLODecoder(nn.Module):
    def __init__(self, in_channels: List[int], num_classes: int = 1, anchors_per_scale: int = 3) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.anchors_per_scale = anchors_per_scale
        out_c = anchors_per_scale * (5 + num_classes)
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(ch, ch // 2, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(ch // 2),
                    nn.LeakyReLU(0.1, inplace=True),
                    nn.Conv2d(ch // 2, out_c, kernel_size=1),
                )
                for ch in in_channels
            ]
        )

    def forward(self, c2: torch.Tensor, c3: torch.Tensor, c4: torch.Tensor) -> List[torch.Tensor]:
        return [h(f) for h, f in zip(self.heads, [c2, c3, c4])]


class MorphologyHead(nn.Module):
    def __init__(self, in_channels: int = 128, shape_classes: int = 3, figo_classes: int = 8) -> None:
        super().__init__()
        feat_dim = in_channels * 2
        self.shared = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
        )
        self.shape_fc = nn.Linear(256, shape_classes)
        self.size_fc = nn.Linear(256, 1)
        self.figo_fc = nn.Linear(256, figo_classes)
        self.ecc_fc = nn.Sequential(nn.Linear(256, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        gap = F.adaptive_avg_pool2d(x, 1).flatten(1)
        gmp = F.adaptive_max_pool2d(x, 1).flatten(1)
        feat = self.shared(torch.cat([gap, gmp], dim=1))
        return {
            "shape_logits": self.shape_fc(feat),
            "size_pred": self.size_fc(feat),
            "figo_logits": self.figo_fc(feat),
            "ecc_pred": self.ecc_fc(feat),
        }


class MACAM(nn.Module):
    def __init__(self, pretrained: bool = True, num_det_classes: int = 1) -> None:
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        base = resnet50(weights=weights)
        self._adapt_first_conv(base)
        self._dilate_layer4(base)

        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

        self.det_head = YOLODecoder([512, 1024, 2048], num_classes=num_det_classes, anchors_per_scale=3)
        self.seg_head = UNetDecoder()
        self.morph_head = MorphologyHead(128, shape_classes=3, figo_classes=8)

        self.anchors = {
            "p3": torch.tensor([[32, 32], [48, 24], [64, 64]], dtype=torch.float32),
            "p4": torch.tensor([[96, 48], [128, 128], [64, 64]], dtype=torch.float32),
            "p5": torch.tensor([[128, 64], [192, 96], [128, 128]], dtype=torch.float32),
        }

    @staticmethod
    def _adapt_first_conv(base: nn.Module) -> None:
        w = base.conv1.weight.data
        new_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        new_conv.weight.data = w.mean(dim=1, keepdim=True)
        base.conv1 = new_conv

    @staticmethod
    def _dilate_layer4(base: nn.Module) -> None:
        for i, block in enumerate(base.layer4):
            if i == 0:
                block.conv2.stride = (1, 1)
                if block.downsample is not None:
                    block.downsample[0].stride = (1, 1)
            block.conv2.dilation = (2, 2)
            block.conv2.padding = (2, 2)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        input_size = x.shape[-2:]
        x = self.stem(x)
        c1 = self.layer1(x)  # 128 x 128 x 256
        c2 = self.layer2(c1)  # 64 x 64 x 512
        c3 = self.layer3(c2)  # 32 x 32 x 1024
        c4 = self.layer4(c3)  # 16 x 16 x 2048

        det_outputs = self.det_head(c2, c3, c4)
        seg = self.seg_head(c1, c2, c3, c4)
        morph = self.morph_head(seg["decoder_feat"])
        return {
            "det_outputs": det_outputs,
            "seg_logits": F.interpolate(seg["seg_logits"], size=input_size, mode="bilinear", align_corners=False),
            "decoder_feat": seg["decoder_feat"],
            "shape_logits": morph["shape_logits"],
            "size_pred": morph["size_pred"],
            "figo_logits": morph["figo_logits"],
            "ecc_pred": morph["ecc_pred"],
            "e_hat": seg["e_hat"],
            "features": {"c1": c1, "c2": c2, "c3": c3, "c4": c4},
        }


def non_max_suppression(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.45) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)
    x1, y1, x2, y2 = boxes.unbind(-1)
    areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break
        xx1 = torch.maximum(x1[i], x1[order[1:]])
        yy1 = torch.maximum(y1[i], y1[order[1:]])
        xx2 = torch.minimum(x2[i], x2[order[1:]])
        yy2 = torch.minimum(y2[i], y2[order[1:]])
        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-8)
        order = order[1:][iou <= iou_threshold]
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)
