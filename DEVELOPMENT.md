# Development Guide

This document describes the architecture, data flow, and implementation responsibilities of the surgical instrument segmentation project.

## Goals

The codebase is designed around a few explicit priorities:

- use GPU for MONAI-based segmentation
- keep the processing flow easy to reason about
- separate segmentation, geometry, tracking, rendering, and GUI concerns
- preserve frame-to-result alignment in video mode
- keep the UI responsive for long-running still-image folder processing

## High-Level Architecture

```text
GUI
  -> pipelines
    -> segmentation
    -> geometry
    -> tracking
    -> rendering
```

The GUI is responsible for user interaction and presentation only. Heavy processing is delegated to services and pipelines.

## Directory Responsibilities

### `app/config`

Configuration and environment loading.

- [settings.py](./app/config/settings.py)
  - project root paths
  - model paths
  - environment-derived runtime options

### `app/domain`

Shared data models used across layers.

- [models.py](./app/domain/models.py)
  - `ToolGeometry`
  - `FrameResult`
  - `FrameInput`
  - `ProcessedFrame`
  - `ImageFolderBatch`
  - `VideoStreamInfo`
  - `MediaKind`

### `app/services`

Low-level, focused processing units.

#### `segmentation`

- [monai_segmenter.py](./app/services/segmentation/monai_segmenter.py)
  - loads local MONAI weights
  - prepares the model on GPU
  - runs inference and produces a binary mask

#### `geometry`

- [tool_geometry.py](./app/services/geometry/tool_geometry.py)
  - binary mask cleanup
  - connected component filtering
  - contour extraction
  - contour center computation
  - endpoint `A` and `B` extraction
  - tip selection

#### `tracking`

- [simple_tracker.py](./app/services/tracking/simple_tracker.py)
  - per-frame matching
  - `track_id` assignment
  - track aging
  - short trajectory history

#### `rendering`

- [overlay_renderer.py](./app/services/rendering/overlay_renderer.py)
  - segmentation mask blending
  - contour drawing
  - axis rendering
  - tip marker rendering
  - trajectory rendering

#### `runtime`

- [device.py](./app/services/runtime/device.py)
  - CUDA availability inspection
  - runtime device summary for UI and startup checks

### `app/pipelines`

High-level orchestration around services.

#### `image_pipeline.py`

- reads an image from disk
- converts BGR to RGB
- calls `MonaiToolSegmenter.analyze_image`
- returns `ProcessedFrame`

Also supports folder enumeration and sequential folder processing.

#### `video_pipeline.py`

- opens a video via OpenCV
- exposes a stateful session object
- reads one frame at a time
- runs segmentation and geometry
- applies tracking
- returns `ProcessedFrame`

The video path is intentionally session-based because the GUI needs:

- next-frame stepping
- playback
- pause
- seek
- trajectory state persistence

### `app/gui`

Presentation, controls, and UI thread coordination.

#### `main_window.py`

Main responsibilities:

- build the PyQt6 interface
- connect buttons to image/video workflows
- manage playback timer
- own folder-processing thread lifecycle
- render processed results into the viewer
- update processing info
- manage the frame slider

#### `folder_processing_worker.py`

Background worker for still-image folder sequence processing.

It exists because processing a whole folder synchronously on the UI thread freezes user interaction.

## Data Flow

### Single Still Image

```text
Open Image
-> ImagePipeline.process_image
-> MonaiToolSegmenter.segment_mask
-> geometry extraction
-> ProcessedFrame
-> OverlayRenderer.render
-> GUI display
```

### Folder Sequence

```text
Open Folder
-> folder image list
-> select current image in UI
-> Process Folder Sequence
-> FolderProcessingWorker on QThread
-> image-by-image ProcessedFrame signals
-> GUI display update per image
```

Processing starts from the currently selected folder image, not necessarily from the first image.

### Video

```text
Open Video
-> VideoPipeline.open
-> VideoPipelineSession
-> read_next_processed_frame
-> segmentation
-> geometry
-> tracker update
-> OverlayRenderer.render
-> GUI display
```

## Segmentation Design

The segmentation model is loaded locally from:

```text
data/model/models/model.pt
```

Important design choices:

- the model is loaded lazily
- GPU is preferred and treated as required for the intended workflow
- `FlexibleUNet` uses `pretrained=False`
  - this avoids accidental network download for backbone weights
- the output mask is resized back to the original input size

### Current Segmentation Steps

1. normalize input to RGB
2. resize to MONAI input size
3. convert to tensor
4. run model
5. apply softmax
6. extract foreground channel
7. resize to original image size
8. threshold into a binary mask

## Geometry Design

Geometry is derived entirely from the segmentation mask.

### Mask Cleanup

The current implementation applies:

- morphological open
- morphological close
- connected-component filtering by minimum area

### Contours

Contours are extracted using:

- `cv2.findContours(..., cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)`

Only external contours are used in the current version.

### Axis Estimation

For each contour:

1. compute center using contour moments
2. select the contour point farthest from the center as `A`
3. select the contour point farthest from `A` as `B`
4. define `A-B` as the virtual center axis

### Tip Selection

The endpoint closer to the image center is selected as the instrument tip.

This rule matches the intended laparoscopic view assumption where the working tip tends to point toward the image center.

## Tracking Design

Tracking is intentionally simple.

### Why simple tracking?

The project prioritizes:

- clear logic
- stable frame alignment
- low debugging overhead

Complex motion models can be added later, but the current version uses nearest-neighbor matching based on contour-derived tool geometry.

### Matching Rule

Each current tool is compared to existing tracks using:

- tip distance
- center distance

The combined score is:

```text
score = tip_distance + 0.35 * center_distance
```

Tracks outside the configured distance threshold are ignored.

### Track State

Each track stores:

- `track_id`
- last tip
- last center
- missed frame count
- trajectory history

### Track Lifecycle

- unmatched detections create new tracks
- unmatched tracks age
- stale tracks are removed after `max_missed_frames`

## Rendering Design

The renderer works on RGB frames and produces one rendered RGB frame for display.

Current overlays:

- segmentation mask
- contour outline
- bounding box
- axis line
- tip point
- contour / track labels
- trajectory polyline

The trajectory color was intentionally changed to `dodgerblue`-like RGB to separate it from the red surgical background.

## GUI Design

### Layout

Current layout has two main columns:

- left: controls
- center: viewer + processing info

The control area uses a tab widget:

- `Still Images`
- `Video`

This reduces clutter and keeps only the relevant controls visible.

### Still Images Tab

Contains:

- open single image
- open folder
- previous / next image
- process folder sequence
- stop process
- folder image list

### Video Tab

Contains:

- open video
- play
- pause
- next frame

### Viewer Footer

Contains:

- frame slider
- current frame position
- current timestamp

### Processing Info

Shows current state and per-frame summary, including:

- current source
- frame number
- timestamp
- folder sequence position
- video FPS
- processing FPS
- detected tools

## Threading Model

### Still-image folder sequence

Runs on a dedicated `QThread` through `FolderProcessingWorker`.

Why:

- processing an entire folder on the UI thread freezes the GUI
- the user should be able to see progressive results
- the user should be able to request stop

### Stop behavior

The stop button does not interrupt GPU inference in the middle of a single image. Instead:

- it requests stop
- the current image finishes
- processing stops before the next image begins

This is intentional for safety and simplicity.

### Video processing

Video currently advances frame-by-frame from the GUI timer and processes each frame synchronously. This keeps frame order clear, but it can still limit responsiveness at high processing cost.

If future performance work is needed, video inference can be moved to a background worker too.

## Installation Flow

[install.bat](./install.bat) is the canonical setup entry point.

It:

1. ensures `.venv`
2. runs `uv sync --extra dev`
3. downloads the MONAI model
4. verifies PyTorch CUDA availability

### Model download

[app/scripts/download_models.py](./app/scripts/download_models.py) downloads:

- repository: `MONAI/endoscopic_tool_segmentation`
- file: `models/model.pt`

into:

```text
data/model/models/model.pt
```

## Key Tradeoffs

### Chosen

- simple and inspectable tracking instead of more advanced Kalman-based modeling
- local model file instead of dynamic runtime download
- explicit pipelines instead of tightly coupling processing to widgets
- background folder processing but synchronous video processing

### Deferred

- result export
- automated tests for geometry and tracking
- asynchronous video inference worker
- richer playback controls
- model selection

## Extension Points

Likely future additions can go here:

- `app/services/tracking/` for alternative trackers
- `app/services/rendering/` for richer overlays
- `app/pipelines/` for batch export or offline video processing
- `app/gui/` for settings dialogs and result export tools

## Recommended Next Improvements

- add automated tests for geometry calculations
- add export for processed still images and video snapshots
- add CSV or JSON result export
- move video processing into a worker when needed
- add tracking confidence visualization
- add adjustable thresholds from the GUI
