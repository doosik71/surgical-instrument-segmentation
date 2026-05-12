"""Download required model files into the local data directory."""

from __future__ import annotations

import logging
from pathlib import Path

from huggingface_hub import hf_hub_download

from app.config.settings import AppSettings


LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure simple console logging for setup scripts."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def download_monai_model(settings: AppSettings) -> Path:
    """Download the MONAI model into the local model directory if needed."""
    local_model_path = settings.local_model_path
    local_model_path.parent.mkdir(parents=True, exist_ok=True)
    if local_model_path.exists():
        LOGGER.info("Model already exists: %s", local_model_path)
        return local_model_path

    LOGGER.info("Downloading model from %s to %s", settings.model_repo_id, settings.model_dir)
    downloaded_path = hf_hub_download(
        repo_id=settings.model_repo_id,
        filename=settings.model_filename,
        token=settings.huggingface_token,
        local_dir=str(settings.model_dir),
        local_dir_use_symlinks=False,
    )

    resolved_path = Path(downloaded_path)

    LOGGER.info("Downloaded model: %s", resolved_path)
    return resolved_path


def main() -> int:
    """Download all required setup assets."""
    configure_logging()
    settings = AppSettings.from_env()

    if not settings.huggingface_token:
        LOGGER.error("Missing huggingface_token in .env")
        return 1

    model_path = download_monai_model(settings)
    LOGGER.info("Model ready: %s", model_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
