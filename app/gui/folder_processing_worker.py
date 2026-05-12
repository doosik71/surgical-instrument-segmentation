"""Background worker for sequential still-image processing."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QObject, pyqtSignal

from app.config.settings import AppSettings
from app.pipelines import ImagePipeline


class FolderProcessingWorker(QObject):
    """Process folder images sequentially on a background thread."""

    frame_processed = pyqtSignal(object, int, int)
    failed = pyqtSignal(str)
    finished = pyqtSignal(int, bool)

    def __init__(self, settings: AppSettings, image_paths: list[Path], start_index: int = 0) -> None:
        super().__init__()
        self.settings = settings
        self.image_paths = image_paths
        self.start_index = max(0, min(start_index, max(0, len(image_paths) - 1)))
        self._stop_requested = False

    def request_stop(self) -> None:
        """Request that processing stop after the current image finishes."""
        self._stop_requested = True

    def run(self) -> None:
        """Process the image list sequentially."""
        processed_count = 0
        stopped = False

        try:
            pipeline = ImagePipeline(settings=self.settings)
            total = len(self.image_paths)

            for index in range(self.start_index, total):
                if self._stop_requested:
                    stopped = True
                    break

                image_path = self.image_paths[index]
                processed_frame = pipeline.process_image(image_path)
                self.frame_processed.emit(processed_frame, index, total)
                processed_count += 1

            if self._stop_requested and processed_count < len(self.image_paths):
                stopped = True
        except Exception as error:
            self.failed.emit(str(error))
            return

        self.finished.emit(processed_count, stopped)
