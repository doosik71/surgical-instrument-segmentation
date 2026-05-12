"""Still-image processing pipeline."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.config.settings import AppSettings
from app.domain.models import FrameInput, ImageFolderBatch, MediaKind, ProcessedFrame
from app.services.segmentation.monai_segmenter import MonaiToolSegmenter


SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


class ImagePipeline:
    """Pipeline for single images and sequential folder processing."""

    def __init__(
        self,
        settings: AppSettings,
        segmenter: MonaiToolSegmenter | None = None,
    ) -> None:
        self.settings = settings
        self.segmenter = segmenter or MonaiToolSegmenter(settings)

    @staticmethod
    def _read_image_rgb(path: Path) -> np.ndarray:
        """Read an image file and convert it to RGB."""
        image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Failed to load image: {path}")
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

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
        image_rgb: np.ndarray,
        source_path: str | Path,
        kind: MediaKind = MediaKind.IMAGE,
        frame_index: int = 0,
        sequence_index: int | None = None,
        sequence_length: int | None = None,
    ) -> ProcessedFrame:
        """Process one in-memory RGB image."""
        source = Path(source_path)
        frame = FrameInput(
            kind=kind,
            source_path=source,
            image_rgb=image_rgb,
            frame_index=frame_index,
            sequence_index=sequence_index,
            sequence_length=sequence_length,
        )
        result = self.segmenter.analyze_image(image_rgb)
        return ProcessedFrame(frame=frame, result=result)

    def process_image(self, image_path: str | Path) -> ProcessedFrame:
        """Process one image file."""
        path = Path(image_path)
        image_rgb = self._read_image_rgb(path)
        return self.process_array(image_rgb=image_rgb, source_path=path, kind=MediaKind.IMAGE)

    def iter_folder(self, folder_path: str | Path) -> list[ProcessedFrame]:
        """Process all supported images in a folder sequentially."""
        image_paths = self.list_image_paths(folder_path)
        items: list[ProcessedFrame] = []

        for index, path in enumerate(image_paths):
            image_rgb = self._read_image_rgb(path)
            items.append(
                self.process_array(
                    image_rgb=image_rgb,
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
