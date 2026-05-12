# Surgical Instrument Segmentation and Tip Tracking

Python desktop application for surgical instrument segmentation, contour-based axis extraction, tip detection, and video tip tracking.

The project uses the `MONAI/endoscopic_tool_segmentation` model on GPU and provides a PyQt6 GUI for:

- single still-image processing
- folder-based sequential still-image review
- video playback with frame stepping
- contour, axis, tip, and trajectory overlays

## Features

- GPU-first MONAI segmentation
- local model loading from `data/model/models/model.pt`
- contour extraction from binary tool masks
- virtual center-axis estimation using contour geometry
- tip selection based on proximity to the image center
- multi-tool tip tracking for video
- still-image and video workflows in one desktop UI
- frame slider for video seek and frame position awareness
- per-frame processing FPS display for video

![Screen Shot](./screen_shot.png)

## Requirements

- Windows
- Python 3.12
- NVIDIA GPU with CUDA support
- `uv` installed and available on `PATH`
- Hugging Face token stored in `.env`

## Project Layout

```text
app/
  config/
  domain/
  gui/
  pipelines/
  scripts/
  services/
  main.py
data/
  model/
install.bat
run-app.bat
pyproject.toml
```

## Installation

1. Create `.env` from `.env.sample`.
2. Put your Hugging Face token in `.env`.

Example:

```ini
huggingface_token=hf_xxxxxxxxxxxxxxxxxxxx
```

1. Run:

```bat
install.bat
```

`install.bat` does the following:

- creates `.venv` with Python 3.12 if needed
- runs `uv sync --extra dev`
- downloads the MONAI model
- verifies CUDA availability in PyTorch

## Model Location

The application loads the model from:

```text
data/model/models/model.pt
```

The download step is implemented in [app/scripts/download_models.py](./app/scripts/download_models.py).

## Run

```bat
run-app.bat
```

This launches the PyQt6 application through [app/main.py](./app/main.py).

## GUI Overview

### Still Images tab

- `Open Image`: process one image immediately
- `Open Folder`: load a folder of still images
- `Previous Image` / `Next Image`: move through the loaded folder
- `Process Folder Sequence`: start sequential processing from the currently selected folder image
- `Stop Process`: stop folder processing after the current image finishes
- `Folder Images`: select an image directly from the folder list

### Video tab

- `Open Video`: load a video file
- `Play`: continuous frame processing and playback
- `Pause`: stop playback
- `Next Frame`: process exactly one more frame
- frame slider below the viewer:
  - shows current frame position
  - supports fast seeking
  - shows current timestamp

### Viewer overlays

- green mask overlay for segmentation
- gold contour outlines
- cyan axis line
- red tip marker
- track trajectory in dodger-blue
- contour and track labels

### Processing Info

The panel below the viewer shows:

- source path
- media kind
- frame index
- timestamp for video
- folder position for still-image sequences
- video FPS
- processing FPS
- contour count
- detected tool count
- per-tool geometry summary

## Processing Logic

### Segmentation

- input image is normalized to RGB
- image is resized for MONAI inference
- foreground probability map is generated on GPU
- mask is resized back to original image size
- thresholding produces a binary tool mask

### Geometry

For each contour:

1. compute the contour center
2. find point `A`, the contour point farthest from the center
3. find point `B`, the contour point farthest from `A`
4. define line `A-B` as the virtual tool center axis
5. choose the endpoint closer to image center as the tip

### Tracking

Video tracking uses a simple nearest-neighbor matcher based on:

- tip distance
- center distance

Each track keeps:

- stable `track_id`
- last tip
- last center
- short trajectory history

## Main Modules

- [app/services/segmentation/monai_segmenter.py](./app/services/segmentation/monai_segmenter.py)
  GPU segmentation and local model loading

- [app/services/geometry/tool_geometry.py](./app/services/geometry/tool_geometry.py)
  mask cleanup, contour extraction, axis and tip computation

- [app/services/tracking/simple_tracker.py](./app/services/tracking/simple_tracker.py)
  lightweight video tracking

- [app/services/rendering/overlay_renderer.py](./app/services/rendering/overlay_renderer.py)
  visualization overlay rendering

- [app/pipelines/image_pipeline.py](./app/pipelines/image_pipeline.py)
  still-image pipeline

- [app/pipelines/video_pipeline.py](./app/pipelines/video_pipeline.py)
  video session pipeline

- [app/gui/main_window.py](./app/gui/main_window.py)
  desktop application UI

## Notes

- GPU is treated as required for the intended workflow.
- Folder sequence processing runs on a background thread so the UI remains responsive.
- `Stop Process` stops after the current image finishes, not in the middle of GPU inference.
- The current tracker is intentionally simple and optimized for clarity and stability rather than advanced motion modeling.

## Troubleshooting

### Model file not found

Run:

```bat
install.bat
```

Then confirm the file exists at:

```text
data/model/models/model.pt
```

### CUDA not available

Check that:

- the NVIDIA driver is installed correctly
- your PyTorch CUDA build was installed successfully
- the GPU is visible to PyTorch

You can re-run:

```bat
install.bat
```

### GUI freezes during very heavy processing

Still-image folder sequence processing already runs in a background thread, but video frame inference is still processed synchronously per frame. If needed, that can be moved to a worker thread in a future iteration.

## Developer Docs

See [DEVELOPMENT.md](./DEVELOPMENT.md) for the detailed design and implementation notes.
