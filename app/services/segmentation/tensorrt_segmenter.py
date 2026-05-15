"""TensorRT segmentation service backed by a local engine file."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.config.settings import AppSettings
from app.domain.models import FrameResult
from app.services.geometry import extract_tool_geometries_from_mask
from app.services.runtime.device import get_device_status
from app.services.segmentation.base import LoadedModelInfo


class TensorRTToolSegmenter:
    """GPU-first TensorRT segmenter using a serialized local engine."""

    def __init__(
        self,
        settings: AppSettings,
        engine_path: Path | None = None,
        input_size: tuple[int, int] = (480, 736),
        mask_threshold: float = 0.5,
        min_component_area: int = 400,
        min_contour_area: int = 400,
    ) -> None:
        self.settings = settings
        self.engine_path = engine_path or settings.local_trt_model_path
        self.device_status = get_device_status(require_gpu=settings.require_gpu)
        self.input_size = input_size
        self.mask_threshold = mask_threshold
        self.min_component_area = min_component_area
        self.min_contour_area = min_contour_area
        self.model_info: LoadedModelInfo | None = None
        self.torch = None
        self.trt = None
        self.device = None
        self.engine = None
        self.context = None
        self.input_name: str | None = None
        self.output_name: str | None = None
        self.output_dtype = None

    def load(self) -> LoadedModelInfo:
        """Load the TensorRT engine and execution context."""
        if not self.device_status.ready:
            raise RuntimeError(self.device_status.reason or "GPU runtime is not ready")
        if not self.engine_path.exists():
            raise FileNotFoundError(f"TensorRT engine file not found: {self.engine_path}")

        try:
            import tensorrt as trt
            import torch
        except ImportError as error:
            raise RuntimeError(
                "TensorRT runtime is not available. Install the matching 'tensorrt' Python package first."
            ) from error

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(self.engine_path.read_bytes())
        if engine is None:
            raise RuntimeError(
                "Failed to deserialize the TensorRT engine. "
                f"Rebuild {self.engine_path} with the installed TensorRT runtime ({trt.__version__})."
            )

        context = engine.create_execution_context()
        if context is None:
            raise RuntimeError("Failed to create a TensorRT execution context.")

        self.trt = trt
        self.torch = torch
        self.device = torch.device(self.device_status.device_label)
        self.engine = engine
        self.context = context
        self.input_name = self._find_tensor_name(trt.TensorIOMode.INPUT)
        self.output_name = self._find_tensor_name(trt.TensorIOMode.OUTPUT)
        self.output_dtype = torch.from_numpy(
            np.empty((), dtype=trt.nptype(self.engine.get_tensor_dtype(self.output_name)))
        ).dtype
        self.model_info = LoadedModelInfo(
            runtime="trt",
            device=self.device_status.device_label,
            weights_path=self.engine_path,
            filename=self.engine_path.name,
        )
        return self.model_info

    def _ensure_loaded(self) -> None:
        if self.engine is None or self.context is None:
            self.load()

    def _find_tensor_name(self, mode) -> str:
        for index in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(index)
            if self.engine.get_tensor_mode(tensor_name) == mode:
                return tensor_name
        raise RuntimeError(f"Could not find a tensor with mode {mode}")

    @staticmethod
    def _normalize_image(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.ndim == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Expected an image with shape (H, W), (H, W, 3), or (H, W, 4)")

        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        return image

    def _prepare_input_tensor(self, image: np.ndarray):
        normalized = self._normalize_image(image)
        image_height, image_width = normalized.shape[:2]
        if (image_height, image_width) != self.input_size:
            raise ValueError(
                f"TensorRT segmenter expects image size {self.input_size}, got {(image_height, image_width)}"
            )
        return (
            self.torch.from_numpy(normalized.transpose(2, 0, 1))
            .float()
            .div(255.0)
            .unsqueeze(0)
            .contiguous()
            .to(self.device)
        )

    def _infer_logits(self, input_tensor):
        input_shape = tuple(int(dim) for dim in input_tensor.shape)
        if not self.context.set_input_shape(self.input_name, input_shape):
            raise RuntimeError(f"Failed to set TensorRT input shape to {input_shape}")

        output_shape = tuple(int(dim) for dim in self.context.get_tensor_shape(self.output_name))
        if any(dim < 0 for dim in output_shape):
            raise RuntimeError(f"TensorRT returned an unresolved output shape: {output_shape}")

        output_tensor = self.torch.empty(output_shape, device=self.device, dtype=self.output_dtype)
        self.context.set_tensor_address(self.input_name, int(input_tensor.data_ptr()))
        self.context.set_tensor_address(self.output_name, int(output_tensor.data_ptr()))
        stream = self.torch.cuda.current_stream(device=self.device)
        if not self.context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT execute_async_v3 failed.")
        return output_tensor

    def segment_mask(self, image: np.ndarray) -> np.ndarray:
        """Run TensorRT inference and return a binary foreground mask."""
        self._ensure_loaded()
        input_tensor = self._prepare_input_tensor(image)

        logits = self._infer_logits(input_tensor)
        probabilities = self.torch.softmax(logits, dim=1)
        foreground = probabilities[:, 1, :, :]
        mask_tensor = foreground[0].detach().cpu().numpy()
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
