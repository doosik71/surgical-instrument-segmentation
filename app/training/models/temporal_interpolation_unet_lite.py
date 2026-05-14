"""Lightweight temporal interpolation segmentation model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(slots=True)
class EncoderFeatures:
    """Feature pyramid emitted by one encoder branch."""

    stage1: torch.Tensor
    stage2: torch.Tensor
    stage3: torch.Tensor


class ConvNormAct(nn.Sequential):
    """A compact Conv-BN-ReLU block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
    ) -> None:
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class DepthwiseSeparableBlock(nn.Module):
    """Depthwise separable convolution block with optional residual."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.depthwise = ConvNormAct(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=stride,
            groups=in_channels,
        )
        self.pointwise = ConvNormAct(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
        )
        self.projection = None
        if stride != 1 or in_channels != out_channels:
            self.projection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the block."""
        identity = x
        x = self.depthwise(x)
        x = self.pointwise(x)
        if self.projection is not None:
            identity = self.projection(identity)
        x = x + identity
        return self.activation(x)


class EncoderStage(nn.Sequential):
    """One downsampling encoder stage."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            DepthwiseSeparableBlock(in_channels, out_channels, stride=2),
            DepthwiseSeparableBlock(out_channels, out_channels, stride=1),
        )


class SharedRgbEncoder(nn.Module):
    """Shared encoder for previous and current RGB frames."""

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        self.stage1 = EncoderStage(in_channels, 24)
        self.stage2 = EncoderStage(24, 40)
        self.stage3 = EncoderStage(40, 64)

    def forward(self, x: torch.Tensor) -> EncoderFeatures:
        """Encode an RGB image."""
        stage1 = self.stage1(x)
        stage2 = self.stage2(stage1)
        stage3 = self.stage3(stage2)
        return EncoderFeatures(stage1=stage1, stage2=stage2, stage3=stage3)


class MaskEncoder(nn.Module):
    """Dedicated encoder for the previous segmentation mask."""

    def __init__(self, in_channels: int = 1) -> None:
        super().__init__()
        self.stage1 = EncoderStage(in_channels, 12)
        self.stage2 = EncoderStage(12, 24)
        self.stage3 = EncoderStage(24, 32)

    def forward(self, x: torch.Tensor) -> EncoderFeatures:
        """Encode a binary mask."""
        stage1 = self.stage1(x)
        stage2 = self.stage2(stage1)
        stage3 = self.stage3(stage2)
        return EncoderFeatures(stage1=stage1, stage2=stage2, stage3=stage3)


class FusionBlock(nn.Sequential):
    """Compress concatenated branch features into one fused tensor."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            ConvNormAct(in_channels, out_channels, kernel_size=1),
            DepthwiseSeparableBlock(out_channels, out_channels, stride=1),
        )


class DecoderBlock(nn.Module):
    """Upsample, fuse skip features, and refine."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.fuse = nn.Sequential(
            DepthwiseSeparableBlock(in_channels + skip_channels, out_channels, stride=1),
            DepthwiseSeparableBlock(out_channels, out_channels, stride=1),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        """Decode one stage."""
        x = self.upsample(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.fuse(x)


class TemporalInterpolationUNetLite(nn.Module):
    """Three-branch lightweight temporal interpolation network."""

    def __init__(self) -> None:
        super().__init__()
        self.rgb_encoder = SharedRgbEncoder(in_channels=3)
        self.mask_encoder = MaskEncoder(in_channels=1)

        self.fuse_stage1 = FusionBlock(in_channels=24 + 24 + 12, out_channels=32)
        self.fuse_stage2 = FusionBlock(in_channels=40 + 40 + 24, out_channels=48)
        self.fuse_stage3 = FusionBlock(in_channels=64 + 64 + 32, out_channels=72)

        self.bottleneck = nn.Sequential(
            DepthwiseSeparableBlock(72, 96, stride=1),
            DepthwiseSeparableBlock(96, 96, stride=1),
        )

        self.decoder1 = DecoderBlock(in_channels=96, skip_channels=48, out_channels=64)
        self.decoder2 = DecoderBlock(in_channels=64, skip_channels=32, out_channels=40)
        self.decoder3 = DecoderBlock(in_channels=40, skip_channels=0, out_channels=24)
        self.refine = nn.Sequential(
            ConvNormAct(24, 16, kernel_size=3),
            nn.Conv2d(16, 1, kernel_size=1),
        )

    def forward(
        self,
        previous_rgb: torch.Tensor,
        previous_mask: torch.Tensor,
        current_rgb: torch.Tensor,
    ) -> torch.Tensor:
        """Predict current-frame logits from temporal context."""
        prev_features = self.rgb_encoder(previous_rgb)
        curr_features = self.rgb_encoder(current_rgb)
        mask_features = self.mask_encoder(previous_mask)

        fused_stage1 = self.fuse_stage1(
            torch.cat([prev_features.stage1, curr_features.stage1, mask_features.stage1], dim=1)
        )
        fused_stage2 = self.fuse_stage2(
            torch.cat([prev_features.stage2, curr_features.stage2, mask_features.stage2], dim=1)
        )
        fused_stage3 = self.fuse_stage3(
            torch.cat([prev_features.stage3, curr_features.stage3, mask_features.stage3], dim=1)
        )

        x = self.bottleneck(fused_stage3)
        x = self.decoder1(x, fused_stage2)
        x = self.decoder2(x, fused_stage1)
        x = self.decoder3(x, None)
        return self.refine(x)
