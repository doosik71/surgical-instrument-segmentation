"""Shared data models for the processing pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import numpy as np


Point = tuple[int, int]
BoundingBox = tuple[int, int, int, int]


class MediaKind(StrEnum):
    """Supported media source kinds."""

    IMAGE = "image"
    IMAGE_FOLDER = "image_folder"
    VIDEO = "video"


@dataclass(slots=True)
class ToolGeometry:
    """Geometry derived from a single contour."""

    contour_index: int
    contour: np.ndarray
    area: float
    bounding_box: BoundingBox
    center: Point
    axis_start: Point
    axis_end: Point
    tip: Point
    track_id: int | None = None


@dataclass(slots=True)
class FrameResult:
    """Processing result for one frame or one still image."""

    image_size: tuple[int, int] | None = None
    original_image_size: tuple[int, int] | None = None
    mask: np.ndarray | None = None
    contours: list[np.ndarray] = field(default_factory=list)
    tools: list[ToolGeometry] = field(default_factory=list)
    error_message: str | None = None


@dataclass(slots=True)
class FrameInput:
    """Input metadata and image payload for one frame-like item."""

    kind: MediaKind
    source_path: Path
    image_rgb: np.ndarray
    original_image_size: tuple[int, int] | None = None
    processing_metadata: tuple[float, tuple[int, int]] | None = None  # (scale, pad)
    frame_index: int = 0
    timestamp_seconds: float | None = None
    sequence_index: int | None = None
    sequence_length: int | None = None


@dataclass(slots=True)
class ProcessedFrame:
    """Pair one frame input with its analysis result."""

    frame: FrameInput
    result: FrameResult


@dataclass(slots=True)
class ImageFolderBatch:
    """Batch result for sequential still-image processing."""

    folder_path: Path
    items: list[ProcessedFrame]


@dataclass(slots=True)
class VideoStreamInfo:
    """Basic metadata for a video source."""

    source_path: Path
    fps: float
    frame_count: int
    frame_width: int
    frame_height: int
