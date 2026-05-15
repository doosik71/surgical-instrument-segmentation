"""CLI tool for high-speed surgical tip detection on images and videos."""

from __future__ import annotations
from app.services.tracking import SimpleToolTracker
from app.services.segmentation import ModelRuntime, MonaiToolSegmenter, Segmenter, TensorRTToolSegmenter
from app.pipelines import ImagePipeline, VideoPipeline
from app.domain.models import ProcessedFrame, ToolGeometry
from app.config.settings import AppSettings

import argparse
import json
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg",
                              ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True,
                        help="Path to a .pt or .trt model file.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--image", type=Path, help="Path to an image file or a folder of image files.")
    input_group.add_argument("--video", type=Path,
                             help="Path to a video file.")
    return parser.parse_args()


def resolve_media_kind(image_path: Path | None, video_path: Path | None) -> str:
    """Resolve whether the input request is image or video processing."""
    if image_path is not None:
        return "image"
    if video_path is not None:
        return "video"
    raise ValueError("Either --image or --video must be provided")


def validate_image_input(path: Path) -> None:
    """Validate an image file or folder input."""
    if path.is_file():
        if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {path.suffix}")
        return
    if path.is_dir():
        return
    raise FileNotFoundError(f"Image input not found: {path}")


def validate_video_input(path: Path) -> None:
    """Validate a video file input."""
    if not path.exists():
        raise FileNotFoundError(f"Video input not found: {path}")
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_VIDEO_EXTENSIONS:
        raise ValueError(f"Unsupported video extension: {path.suffix}")


def build_segmenter(settings: AppSettings, model_path: Path) -> tuple[ModelRuntime, Segmenter]:
    """Create the correct segmenter based on the model filename extension."""
    suffix = model_path.suffix.lower()
    if suffix == ".pt":
        return ModelRuntime.PYTORCH, MonaiToolSegmenter(settings=settings, model_path=model_path)
    if suffix == ".trt":
        return ModelRuntime.TENSORRT, TensorRTToolSegmenter(settings=settings, engine_path=model_path)
    raise ValueError(f"Unsupported model extension: {model_path.suffix}")


def serialize_tools(tools: list[ToolGeometry]) -> list[dict[str, object]]:
    """Serialize detected tools for JSONL stdout."""
    sorted_tools = sorted(
        tools,
        key=lambda tool: (
            tool.track_id is None,
            tool.track_id if tool.track_id is not None else tool.contour_index,
            tool.contour_index,
        ),
    )
    items: list[dict[str, object]] = []
    for tool in sorted_tools:
        items.append(
            {
                "contour_index": tool.contour_index,
                "track_id": tool.track_id,
                "tip": [tool.tip[0], tool.tip[1]],
                "center": [tool.center[0], tool.center[1]],
                "bounding_box": list(tool.bounding_box),
            }
        )
    return items


def emit_jsonl(payload: dict[str, object]) -> None:
    """Emit one JSON object per line to stdout."""
    print(json.dumps(payload, ensure_ascii=True), flush=True)


def print_model_load_summary(runtime: ModelRuntime, model_path: Path, elapsed_seconds: float) -> None:
    """Print model load timing."""
    emit_jsonl(
        {
            "event": "model_loaded",
            "runtime": runtime.value,
            "model_path": str(model_path),
            "load_time_ms": round(elapsed_seconds * 1000.0, 3),
        }
    )


def print_image_result(processed_frame: ProcessedFrame, elapsed_seconds: float) -> None:
    """Print one image result line."""
    emit_jsonl(
        {
            "event": "image_result",
            "input_path": str(processed_frame.frame.source_path),
            "processing_time_ms": round(elapsed_seconds * 1000.0, 3),
            "tools": serialize_tools(processed_frame.result.tools),
        }
    )


def print_image_folder_summary(input_path: Path, processed_count: int, elapsed_seconds: float) -> None:
    """Print end-of-folder aggregate timing."""
    images_per_second = (
        processed_count / elapsed_seconds) if elapsed_seconds > 0 else 0.0
    emit_jsonl(
        {
            "event": "image_folder_summary",
            "input_path": str(input_path),
            "images": processed_count,
            "total_time_s": round(elapsed_seconds, 3),
            "images_per_second": round(images_per_second, 3),
        }
    )


def print_video_frame_result(processed_frame: ProcessedFrame, elapsed_seconds: float) -> None:
    """Print one video frame result line."""
    frame = processed_frame.frame
    fps = (1.0 / elapsed_seconds) if elapsed_seconds > 0 else 0.0
    emit_jsonl(
        {
            "event": "video_frame",
            "input_path": str(frame.source_path),
            "frame_index": frame.frame_index,
            "timestamp_s": round(frame.timestamp_seconds, 3) if frame.timestamp_seconds is not None else None,
            "fps": round(fps, 3),
            "tools": serialize_tools(processed_frame.result.tools),
        }
    )


def print_video_summary(input_path: Path, frame_count: int, elapsed_seconds: float) -> None:
    """Print end-of-video aggregate timing."""
    total_fps = (frame_count / elapsed_seconds) if elapsed_seconds > 0 else 0.0
    emit_jsonl(
        {
            "event": "video_summary",
            "input_path": str(input_path),
            "frames": frame_count,
            "total_time_s": round(elapsed_seconds, 3),
            "total_fps": round(total_fps, 3),
        }
    )


def process_image(settings: AppSettings, segmenter: Segmenter, input_path: Path) -> int:
    """Run tip detection on one image or one folder of images."""
    pipeline = ImagePipeline(settings=settings, segmenter=segmenter)
    if input_path.is_dir():
        image_paths = pipeline.list_image_paths(input_path)
        if not image_paths:
            raise FileNotFoundError(
                f"No supported images found in folder: {input_path}")

        total_started_at = time.perf_counter()
        processed_count = 0
        for image_path in image_paths:
            started_at = time.perf_counter()
            processed_frame = pipeline.process_image(image_path)
            elapsed = time.perf_counter() - started_at
            print_image_result(processed_frame, elapsed)
            processed_count += 1
        total_elapsed = time.perf_counter() - total_started_at
        print_image_folder_summary(input_path, processed_count, total_elapsed)
        return 0

    started_at = time.perf_counter()
    processed_frame = pipeline.process_image(input_path)
    elapsed = time.perf_counter() - started_at
    print_image_result(processed_frame, elapsed)
    return 0


def process_video(settings: AppSettings, segmenter: Segmenter, input_path: Path) -> int:
    """Run tip detection on one video and print one line per frame."""
    tracker = SimpleToolTracker()
    pipeline = VideoPipeline(
        settings=settings, segmenter=segmenter, tracker=tracker)
    session = pipeline.open(input_path)

    frame_count = 0
    total_started_at = time.perf_counter()
    try:
        while True:
            frame_started_at = time.perf_counter()
            processed_frame = session.read_next_processed_frame()
            frame_elapsed = time.perf_counter() - frame_started_at
            if processed_frame is None:
                break
            print_video_frame_result(processed_frame, frame_elapsed)
            frame_count += 1
    finally:
        session.close()

    total_elapsed = time.perf_counter() - total_started_at
    print_video_summary(input_path, frame_count, total_elapsed)
    return 0


def main() -> int:
    """Run high-speed tip detection from the command line."""
    args = parse_args()
    settings = AppSettings.from_env()
    model_path = args.model
    input_path = args.image if args.image is not None else args.video

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    assert input_path is not None

    media_kind = resolve_media_kind(args.image, args.video)
    if media_kind == "image":
        validate_image_input(input_path)
    else:
        validate_video_input(input_path)

    runtime, segmenter = build_segmenter(settings, model_path)
    load_started_at = time.perf_counter()
    segmenter.load()
    load_elapsed = time.perf_counter() - load_started_at
    print_model_load_summary(runtime, model_path, load_elapsed)

    if media_kind == "image":
        return process_image(settings, segmenter, input_path)
    return process_video(settings, segmenter, input_path)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting...", file=sys.stderr)
        sys.exit(130)
