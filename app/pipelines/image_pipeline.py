"""Still-image processing pipeline."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.config.settings import AppSettings
from app.domain.models import FrameInput, ImageFolderBatch, MediaKind, ProcessedFrame
from app.services.geometry import resize_with_padding
from app.services.segmentation import MonaiToolSegmenter, Segmenter


SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


class ImagePipeline:
    """Pipeline for single images and sequential folder processing."""

    def __init__(
        self,
        settings: AppSettings,
        segmenter: Segmenter | None = None,
    ) -> None:
        self.settings = settings
        self.segmenter = segmenter or MonaiToolSegmenter(settings)

    @staticmethod
    def _read_image_bgr(path: Path) -> np.ndarray:
        """Read an image file as BGR."""
        image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Failed to load image: {path}")
        return image_bgr

    @staticmethod
    def list_image_paths(folder_path: str | Path) -> list[Path]:
        """List supported image files in a folder in deterministic order."""
        folder = Path(folder_path)
        if not folder.is_dir():
            raise NotADirectoryError(f"Not a folder: {folder}")

        image_paths = [
            path
            for path in sorted(folder.iterdir())
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ]
        return image_paths

    def process_array(
        self,
        image_bgr: np.ndarray,
        source_path: str | Path,
        kind: MediaKind = MediaKind.IMAGE,
        frame_index: int = 0,
        sequence_index: int | None = None,
        sequence_length: int | None = None,
    ) -> ProcessedFrame:
        """Process one in-memory BGR image with performance optimization."""
        source = Path(source_path)
        original_size = image_bgr.shape[:2]

        # Performance Optimization: Resize and pad to segmenter's input size
        processed_bgr, scale, pad = resize_with_padding(image_bgr, self.segmenter.input_size)
        image_rgb = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB)

        frame = FrameInput(
            kind=kind,
            source_path=source,
            image_rgb=image_rgb,
            original_image_size=original_size,
            processing_metadata=(scale, pad),
            frame_index=frame_index,
            sequence_index=sequence_index,
            sequence_length=sequence_length,
        )
        result = self.segmenter.analyze_image(
            image_rgb, 
            original_image_size=original_size, 
            mapping=(scale, pad)
        )
        return ProcessedFrame(frame=frame, result=result)

    def process_image(self, image_path: str | Path) -> ProcessedFrame:
        """Process one image file."""
        path = Path(image_path)
        image_bgr = self._read_image_bgr(path)
        return self.process_array(image_bgr=image_bgr, source_path=path, kind=MediaKind.IMAGE)

    def iter_folder(self, folder_path: str | Path) -> list[ProcessedFrame]:
        """Process all supported images in a folder sequentially."""
        image_paths = self.list_image_paths(folder_path)
        items: list[ProcessedFrame] = []

        for index, path in enumerate(image_paths):
            image_bgr = self._read_image_bgr(path)
            items.append(
                self.process_array(
                    image_bgr=image_bgr,
                    source_path=path,
                    kind=MediaKind.IMAGE_FOLDER,
                    frame_index=index,
                    sequence_index=index,
                    sequence_length=len(image_paths),
                )
            )

        return items

    def process_folder(self, folder_path: str | Path) -> ImageFolderBatch:
        """Process a folder and return batch results."""
        folder = Path(folder_path)
        items = self.iter_folder(folder)
        return ImageFolderBatch(folder_path=folder, items=items)
