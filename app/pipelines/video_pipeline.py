"""Video processing pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2

from app.config.settings import AppSettings
from app.domain.models import FrameInput, MediaKind, ProcessedFrame, VideoStreamInfo
from app.services.segmentation.monai_segmenter import MonaiToolSegmenter
from app.services.tracking import SimpleToolTracker


@dataclass(slots=True)
class VideoPipelineSession:
    """Stateful video-processing session for UI playback and stepping."""

    stream_info: VideoStreamInfo
    capture: cv2.VideoCapture
    segmenter: MonaiToolSegmenter
    tracker: SimpleToolTracker
    current_frame_index: int = 0
    closed: bool = False

    def read_next_processed_frame(self) -> ProcessedFrame | None:
        """Read the next frame, run segmentation, and return the result."""
        if self.closed:
            return None

        ok, frame_bgr = self.capture.read()
        if not ok or frame_bgr is None:
            return None

        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        timestamp = None
        if self.stream_info.fps > 0:
            timestamp = self.current_frame_index / self.stream_info.fps

        frame = FrameInput(
            kind=MediaKind.VIDEO,
            source_path=self.stream_info.source_path,
            image_rgb=image_rgb,
            frame_index=self.current_frame_index,
            timestamp_seconds=timestamp,
        )
        result = self.segmenter.analyze_image(image_rgb)
        result.tools = self.tracker.update(result.tools)
        processed = ProcessedFrame(frame=frame, result=result)

        self.current_frame_index += 1
        return processed

    def seek(self, frame_index: int) -> bool:
        """Seek to a specific zero-based frame index."""
        if self.closed:
            return False

        success = self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        if success:
            self.current_frame_index = frame_index
        return bool(success)

    def close(self) -> None:
        """Release the underlying video capture."""
        if not self.closed:
            self.capture.release()
            self.closed = True

    def trajectories(self) -> dict[int, list[tuple[int, int]]]:
        """Return active tracker trajectories."""
        return self.tracker.get_trajectories()


class VideoPipeline:
    """Pipeline for stateful video processing."""

    def __init__(
        self,
        settings: AppSettings,
        segmenter: MonaiToolSegmenter | None = None,
        tracker: SimpleToolTracker | None = None,
    ) -> None:
        self.settings = settings
        self.segmenter = segmenter or MonaiToolSegmenter(settings)
        self.tracker = tracker or SimpleToolTracker()

    def open(self, video_path: str | Path) -> VideoPipelineSession:
        """Open a video and create a processing session."""
        path = Path(video_path)
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            capture.release()
            raise FileNotFoundError(f"Failed to open video: {path}")

        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        stream_info = VideoStreamInfo(
            source_path=path,
            fps=fps,
            frame_count=frame_count,
            frame_width=frame_width,
            frame_height=frame_height,
        )
        self.tracker.reset()
        return VideoPipelineSession(
            stream_info=stream_info,
            capture=capture,
            segmenter=self.segmenter,
            tracker=self.tracker,
        )

    def iter_processed_frames(
        self,
        video_path: str | Path,
        max_frames: int | None = None,
        frame_stride: int = 1,
    ) -> list[ProcessedFrame]:
        """Process a video sequentially and return processed frames."""
        if frame_stride < 1:
            raise ValueError("frame_stride must be >= 1")

        session = self.open(video_path)
        processed_frames: list[ProcessedFrame] = []
        try:
            while max_frames is None or len(processed_frames) < max_frames:
                processed = session.read_next_processed_frame()
                if processed is None:
                    break
                processed_frames.append(processed)

                if frame_stride > 1:
                    next_frame_index = session.current_frame_index + (frame_stride - 1)
                    if not session.seek(next_frame_index):
                        break
        finally:
            session.close()

        return processed_frames
