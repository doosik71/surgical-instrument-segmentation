"""On-the-fly temporal dataset built directly from videos."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from app.services.geometry import resize_with_padding


@dataclass(frozen=True, slots=True)
class TemporalSampleRef:
    """Reference to one temporal training sample."""

    video_path: Path
    previous_frame_index: int
    current_frame_index: int


def split_video_paths(
    video_paths: list[Path],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> tuple[list[Path], list[Path], list[Path]]:
    """Split sorted videos deterministically into train/val/test sets."""
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be in [0, 1)")
    if not 0.0 <= test_ratio < 1.0:
        raise ValueError("test_ratio must be in [0, 1)")
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be < 1.0")

    sorted_paths = sorted(video_paths)
    total = len(sorted_paths)
    if total == 0:
        return [], [], []

    test_count = int(math.floor(total * test_ratio))
    val_count = int(math.floor(total * val_ratio))
    train_count = total - val_count - test_count

    if train_count <= 0:
        raise ValueError("Not enough videos to create a non-empty training split")

    train_paths = sorted_paths[:train_count]
    val_paths = sorted_paths[train_count : train_count + val_count]
    test_paths = sorted_paths[train_count + val_count :]
    return train_paths, val_paths, test_paths


class TemporalVideoDataset(Dataset[dict[str, torch.Tensor | str | int]]):
    """Read previous/current frame pairs directly from source videos."""

    def __init__(
        self,
        video_paths: list[Path],
        input_size: tuple[int, int] = (480, 736),
        temporal_gap: int = 1,
        frame_stride: int = 1,
        augment: bool = False,
        max_samples: int | None = None,
        seed: int = 7,
    ) -> None:
        if temporal_gap < 1:
            raise ValueError("temporal_gap must be >= 1")
        if frame_stride < 1:
            raise ValueError("frame_stride must be >= 1")

        self.video_paths = sorted(video_paths)
        self.input_size = input_size
        self.temporal_gap = temporal_gap
        self.frame_stride = frame_stride
        self.augment = augment
        self.max_samples = max_samples
        self.seed = seed
        self.sample_refs = self._build_sample_refs()
        self._captures: dict[str, cv2.VideoCapture] = {}

    @staticmethod
    def list_default_video_paths(video_dir: str | Path) -> list[Path]:
        """Return all mp4 files in the directory."""
        directory = Path(video_dir)
        if not directory.is_dir():
            raise NotADirectoryError(f"Not a video directory: {directory}")
        return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() == ".mp4")

    def _build_sample_refs(self) -> list[TemporalSampleRef]:
        """Build lightweight frame-pair references without materializing images."""
        sample_refs: list[TemporalSampleRef] = []
        for video_path in self.video_paths:
            capture = cv2.VideoCapture(str(video_path))
            try:
                if not capture.isOpened():
                    continue
                frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            finally:
                capture.release()

            for current_frame_index in range(self.temporal_gap, frame_count, self.frame_stride):
                sample_refs.append(
                    TemporalSampleRef(
                        video_path=video_path,
                        previous_frame_index=current_frame_index - self.temporal_gap,
                        current_frame_index=current_frame_index,
                    )
                )

        if self.max_samples is not None and len(sample_refs) > self.max_samples:
            rng = random.Random(self.seed)
            sample_refs = rng.sample(sample_refs, self.max_samples)

        return sample_refs

    def __len__(self) -> int:
        """Return the number of available temporal pairs."""
        return len(self.sample_refs)

    def _get_capture(self, video_path: Path) -> cv2.VideoCapture:
        """Lazily cache one capture handle per video path per dataset instance."""
        key = str(video_path)
        capture = self._captures.get(key)
        if capture is None or not capture.isOpened():
            capture = cv2.VideoCapture(key)
            if not capture.isOpened():
                capture.release()
                raise FileNotFoundError(f"Failed to open video: {video_path}")
            self._captures[key] = capture
        return capture

    def _read_frame(self, capture: cv2.VideoCapture, frame_index: int) -> np.ndarray:
        """Seek to a frame and return it as BGR."""
        if not capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index):
            raise RuntimeError(f"Failed to seek to frame {frame_index}")
        ok, frame_bgr = capture.read()
        if not ok or frame_bgr is None:
            raise RuntimeError(f"Failed to read frame {frame_index}")
        return frame_bgr

    def _sample_augmentation_params(self) -> dict[str, float | bool | int]:
        """Sample synchronized image augmentation parameters."""
        rng = random.random
        return {
            "flip": rng() < 0.5,
            "brightness_delta": random.uniform(-18.0, 18.0),
            "contrast_gain": random.uniform(0.9, 1.1),
            "rotation_deg": random.uniform(-7.5, 7.5),
            "blur": rng() < 0.18,
        }

    @staticmethod
    def _apply_transform(image_bgr: np.ndarray, params: dict[str, float | bool | int]) -> np.ndarray:
        """Apply one synchronized transform set to a frame."""
        transformed = image_bgr.copy()
        if bool(params["flip"]):
            transformed = cv2.flip(transformed, 1)

        rotation_deg = float(params["rotation_deg"])
        if abs(rotation_deg) > 1e-6:
            height, width = transformed.shape[:2]
            matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), rotation_deg, 1.0)
            transformed = cv2.warpAffine(
                transformed,
                matrix,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )

        transformed = transformed.astype(np.float32)
        transformed = transformed * float(params["contrast_gain"]) + float(params["brightness_delta"])
        transformed = np.clip(transformed, 0.0, 255.0).astype(np.uint8)

        if bool(params["blur"]):
            transformed = cv2.GaussianBlur(transformed, (3, 3), sigmaX=0.0)

        return transformed

    def _prepare_frame(self, frame_bgr: np.ndarray) -> torch.Tensor:
        """Resize, pad, and convert one frame into a float RGB tensor."""
        processed_bgr, _, _ = resize_with_padding(frame_bgr, self.input_size)
        image_rgb = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(image_rgb.transpose(2, 0, 1)).float().div(255.0)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int]:
        """Return one temporal pair."""
        sample_ref = self.sample_refs[index]
        capture = self._get_capture(sample_ref.video_path)
        previous_bgr = self._read_frame(capture, sample_ref.previous_frame_index)
        current_bgr = self._read_frame(capture, sample_ref.current_frame_index)

        if self.augment:
            params = self._sample_augmentation_params()
            previous_bgr = self._apply_transform(previous_bgr, params)
            current_bgr = self._apply_transform(current_bgr, params)

        previous_rgb = self._prepare_frame(previous_bgr)
        current_rgb = self._prepare_frame(current_bgr)

        return {
            "previous_rgb": previous_rgb,
            "current_rgb": current_rgb,
            "video_path": str(sample_ref.video_path),
            "previous_frame_index": sample_ref.previous_frame_index,
            "current_frame_index": sample_ref.current_frame_index,
        }

    def close(self) -> None:
        """Release cached video captures."""
        for capture in self._captures.values():
            capture.release()
        self._captures.clear()

    def __del__(self) -> None:
        """Best-effort cleanup for cached captures."""
        self.close()
