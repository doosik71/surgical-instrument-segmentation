"""Loss functions for segmentation training."""

from app.training.losses.segmentation_losses import DiceLoss, SegmentationLoss

__all__ = ["DiceLoss", "SegmentationLoss"]
