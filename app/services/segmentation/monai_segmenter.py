"""MONAI segmentation service backed by a local GPU model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.config.settings import AppSettings
from app.domain.models import FrameResult
from app.services.geometry import extract_tool_geometries_from_mask
from app.services.runtime.device import get_device_status


@dataclass(slots=True)
class LoadedModelInfo:
    """Metadata about the loaded MONAI model."""

    repo_id: str
    filename: str
    device: str
    weights_path: Path


class MonaiToolSegmenter:
    """GPU-first MONAI segmenter using a locally downloaded model file."""

    def __init__(
        self,
        settings: AppSettings,
        input_size: tuple[int, int] = (480, 736),
        mask_threshold: float = 0.5,
        min_component_area: int = 400,
        min_contour_area: int = 400,
    ) -> None:
        self.settings = settings
        self.device_status = get_device_status(require_gpu=settings.require_gpu)
        self.input_size = input_size
        self.mask_threshold = mask_threshold
        self.min_component_area = min_component_area
        self.min_contour_area = min_contour_area
        self.model_info: LoadedModelInfo | None = None
        self.model = None
        self.device = None
        self.torch = None
        self.transform = None
        self._pil_image_cls = None

    def load(self) -> LoadedModelInfo:
        """Load the MONAI model from the local model directory."""
        if not self.device_status.ready:
            raise RuntimeError(self.device_status.reason or "GPU runtime is not ready")

        model_path = self.settings.local_model_path
        if not model_path.exists():
            raise FileNotFoundError(f"Local model file not found: {model_path}")

        import torch
        import torchvision.transforms as transforms
        from monai.networks.nets import FlexibleUNet
        from PIL import Image

        self.torch = torch
        self.device = torch.device(self.device_status.device_label)
        self.transform = transforms.Compose(
            [
                transforms.Resize(self.input_size),
                transforms.ToTensor(),
            ]
        )

        model = FlexibleUNet(
            in_channels=3,
            out_channels=2,
            backbone="efficientnet-b2",
            spatial_dims=2,
            pretrained=False,
            is_pad=False,
            pre_conv=None,
        )

        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()

        self.model = model
        self._pil_image_cls = Image
        self.model_info = LoadedModelInfo(
            repo_id=self.settings.model_repo_id,
            filename=self.settings.model_filename,
            device=self.device_status.device_label,
            weights_path=model_path,
        )
        return self.model_info

    def _ensure_loaded(self) -> None:
        """Load the model lazily before inference."""
        if self.model is None:
            self.load()

    @staticmethod
    def _normalize_image(image: np.ndarray) -> np.ndarray:
        """Normalize an image array into a 3-channel uint8 image."""
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.ndim == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Expected an image with shape (H, W), (H, W, 3), or (H, W, 4)")

        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        return image

    def segment_mask(self, image: np.ndarray) -> np.ndarray:
        """Run segmentation and return a binary foreground mask."""
        self._ensure_loaded()

        normalized_image = self._normalize_image(image)
        image_height, image_width = normalized_image.shape[:2]

        # Use the provided image directly if it matches input_size, otherwise transform
        if (image_height, image_width) == self.input_size:
            import torch
            input_tensor = (
                torch.from_numpy(normalized_image.transpose(2, 0, 1))
                .float()
                .div(255.0)
                .unsqueeze(0)
                .to(self.device)
            )
        else:
            input_tensor = self.transform(self._pil_image_cls.fromarray(normalized_image)).unsqueeze(0).to(self.device)

        with self.torch.no_grad():
            output_tensor = self.model(input_tensor)
            probabilities = self.torch.softmax(output_tensor, dim=1)
            foreground = probabilities[:, 1, :, :]

        mask_tensor = foreground[0].detach().cpu().numpy()

        # Only resize back if the input image was NOT already at the target size
        if (image_height, image_width) != self.input_size:
            mask_tensor = cv2.resize(
                mask_tensor,
                (image_width, image_height),
                interpolation=cv2.INTER_LINEAR,
            )

        return (mask_tensor >= self.mask_threshold).astype(np.uint8)

    def analyze_image(
        self,
        image: np.ndarray,
        original_image_size: tuple[int, int] | None = None,
        mapping: tuple[float, tuple[int, int]] | None = None,
    ) -> FrameResult:
        """Run segmentation and contour-based geometry extraction on one image."""
        binary_mask = self.segment_mask(image)
        return extract_tool_geometries_from_mask(
            mask=binary_mask,
            image_size=image.shape[:2],
            original_image_size=original_image_size,
            mapping=mapping,
            min_component_area=self.min_component_area,
            min_contour_area=self.min_contour_area,
        )
