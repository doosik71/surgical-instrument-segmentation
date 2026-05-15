"""Shared segmentation runtime contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol

import numpy as np

from app.domain.models import FrameResult


@dataclass(slots=True)
class LoadedModelInfo:
    """Metadata about the currently loaded model runtime."""

    runtime: str
    device: str
    weights_path: Path
    repo_id: str | None = None
    filename: str | None = None


class ModelRuntime(StrEnum):
    """Supported segmentation runtime types."""

    PYTORCH = "pt"
    TENSORRT = "trt"


@dataclass(slots=True)
class ModelOption:
    """One GUI-selectable model candidate."""

    runtime: ModelRuntime
    path: Path

    @property
    def label(self) -> str:
        suffix = ".pt" if self.runtime == ModelRuntime.PYTORCH else ".trt"
        return f"{self.path.name} ({suffix})"


class Segmenter(Protocol):
    """Runtime-agnostic segmentation contract used by pipelines."""

    input_size: tuple[int, int]
    model_info: LoadedModelInfo | None

    def load(self) -> LoadedModelInfo:
        """Load underlying runtime resources."""

    def analyze_image(
        self,
        image: np.ndarray,
        original_image_size: tuple[int, int] | None = None,
        mapping: tuple[float, tuple[int, int]] | None = None,
    ) -> FrameResult:
        """Run segmentation and contour-based geometry extraction on one image."""
