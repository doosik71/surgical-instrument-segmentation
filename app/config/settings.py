"""Application settings."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(slots=True)
class AppSettings:
    """Runtime settings for the application skeleton."""

    app_name: str = "Surgical Instrument Segmentation"
    app_version: str = "0.1.0"
    project_root: Path = Path(__file__).resolve().parents[2]
    data_dir: Path = Path(__file__).resolve().parents[2] / "data"
    model_dir: Path = Path(__file__).resolve().parents[2] / "data" / "model"
    model_repo_id: str = "MONAI/endoscopic_tool_segmentation"
    model_filename: str = "models/model.pt"
    huggingface_token: str | None = None
    require_gpu: bool = True

    @property
    def local_model_path(self) -> Path:
        """Return the expected local path for the downloaded model file."""
        return self.model_dir / Path(self.model_filename)

    @classmethod
    def from_env(cls) -> "AppSettings":
        """Create settings from environment variables."""
        load_dotenv()
        return cls(
            huggingface_token=os.getenv("huggingface_token") or os.getenv("HUGGINGFACE_TOKEN"),
            require_gpu=os.getenv("REQUIRE_GPU", "true").lower() != "false",
        )
