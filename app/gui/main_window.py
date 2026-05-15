"""Main window for the surgical instrument segmentation application."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

from PyQt6.QtCore import Qt, QThread, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.config.settings import AppSettings
from app.domain.models import ProcessedFrame
from app.gui.folder_processing_worker import FolderProcessingWorker
from app.pipelines import ImagePipeline, VideoPipeline, VideoPipelineSession
from app.services.rendering import OverlayRenderer
from app.services.runtime.device import DeviceStatus, get_device_status
from app.services.segmentation import ModelOption, ModelRuntime, MonaiToolSegmenter, Segmenter, TensorRTToolSegmenter
from app.services.tracking import SimpleToolTracker


class MainWindow(QMainWindow):
    """Desktop GUI for still-image and video workflows."""

    def __init__(self, settings: AppSettings) -> None:
        super().__init__()
        self.settings = settings
        self.device_status = get_device_status(require_gpu=settings.require_gpu)

        self.segmenter: Segmenter | None = None
        self.image_pipeline: ImagePipeline | None = None
        self.video_tracker = SimpleToolTracker()
        self.video_pipeline: VideoPipeline | None = None
        self.overlay_renderer = OverlayRenderer()
        self.model_options: list[ModelOption] = []
        self.selected_model_option: ModelOption | None = None
        self._switching_model = False

        self.folder_image_paths: list[Path] = []
        self.current_folder_index = -1
        self.video_session: VideoPipelineSession | None = None
        self.current_processed_frame: ProcessedFrame | None = None
        self.last_video_interval_ms = 33
        self.current_video_display_index = 0
        self.current_video_processing_fps: float | None = None
        self._slider_is_pressed = False

        self.folder_processing_thread: QThread | None = None
        self.folder_processing_worker: FolderProcessingWorker | None = None
        self.folder_processing_active = False

        self.playback_timer = QTimer(self)
        self.playback_timer.timeout.connect(self._play_next_video_frame)

        self.setWindowTitle(f"{settings.app_name} {settings.app_version}")
        self.resize(1480, 920)
        self._build_ui()
        self._initialize_model_selection()

    def _build_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)

        root_layout = QHBoxLayout(central)

        left_panel = QVBoxLayout()
        left_panel.addWidget(self._build_runtime_box())
        left_panel.addWidget(self._build_mode_tabs(), 1)

        center_panel = QVBoxLayout()
        center_panel.addWidget(self._build_viewer_box(), 4)
        center_panel.addWidget(self._build_info_box(), 2)

        root_layout.addLayout(left_panel, 1)
        root_layout.addLayout(center_panel, 3)

        self.statusBar().showMessage(self._status_text(self.device_status))

    def _build_runtime_box(self) -> QGroupBox:
        box = QGroupBox("Runtime")
        layout = QVBoxLayout(box)

        layout.addWidget(QLabel(f"Device: {self.device_status.device_label}"))
        layout.addWidget(QLabel(f"CUDA available: {self.device_status.cuda_available}"))
        if self.device_status.gpu_name:
            layout.addWidget(QLabel(f"GPU: {self.device_status.gpu_name}"))
        layout.addWidget(QLabel(f"GPU required: {self.settings.require_gpu}"))

        layout.addWidget(QLabel("Model"))
        self.model_selector = QComboBox()
        self.model_selector.currentIndexChanged.connect(self._on_model_selection_changed)
        layout.addWidget(self.model_selector)

        self.model_path_label = QLabel("Path: -")
        self.model_path_label.setWordWrap(True)
        layout.addWidget(self.model_path_label)

        self.model_status_label = QLabel("Status: no model detected")
        self.model_status_label.setWordWrap(True)
        layout.addWidget(self.model_status_label)
        return box

    def _build_mode_tabs(self) -> QTabWidget:
        tabs = QTabWidget()
        tabs.addTab(self._build_still_tab(), "Still Images")
        tabs.addTab(self._build_video_tab(), "Video")
        return tabs

    def _build_still_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        self.open_image_button = QPushButton("Open Image")
        self.open_image_button.clicked.connect(self._open_image)
        self.open_folder_button = QPushButton("Open Folder")
        self.open_folder_button.clicked.connect(self._open_folder)
        self.prev_image_button = QPushButton("Previous Image")
        self.prev_image_button.clicked.connect(self._show_previous_folder_image)
        self.next_image_button = QPushButton("Next Image")
        self.next_image_button.clicked.connect(self._show_next_folder_image)
        self.process_all_button = QPushButton("Process Folder Sequence")
        self.process_all_button.clicked.connect(self._start_folder_processing)
        self.stop_process_button = QPushButton("Stop Process")
        self.stop_process_button.clicked.connect(self._request_stop_folder_processing)

        layout.addWidget(self.open_image_button)
        layout.addWidget(self.open_folder_button)
        layout.addWidget(self.prev_image_button)
        layout.addWidget(self.next_image_button)
        layout.addWidget(self.process_all_button)
        layout.addWidget(self.stop_process_button)
        layout.addWidget(self._build_folder_list_box(), 1)

        self._set_folder_controls_enabled(False)
        self.stop_process_button.setEnabled(False)
        return widget

    def _build_video_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        self.open_video_button = QPushButton("Open Video")
        self.open_video_button.clicked.connect(self._open_video)
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self._start_video_playback)
        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(self._pause_video_playback)
        self.next_frame_button = QPushButton("Next Frame")
        self.next_frame_button.clicked.connect(self._step_video_frame)

        layout.addWidget(self.open_video_button)
        layout.addWidget(self.play_button)
        layout.addWidget(self.pause_button)
        layout.addWidget(self.next_frame_button)
        layout.addStretch(1)

        self._set_video_controls_enabled(False)
        return widget

    def _build_viewer_box(self) -> QGroupBox:
        box = QGroupBox("Viewer")
        layout = QVBoxLayout(box)

        self.viewer_label = QLabel("Load an image, folder, or video to begin.")
        self.viewer_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.viewer_label.setMinimumSize(860, 540)
        self.viewer_label.setStyleSheet("border: 1px solid #909090; background: #111; color: #f4f4f4;")

        layout.addWidget(self.viewer_label)
        layout.addWidget(self._build_video_slider_widget())
        return box

    def _build_video_slider_widget(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 4, 0, 0)

        label_row = QHBoxLayout()
        self.video_position_label = QLabel("Frame: - / -")
        self.video_time_label = QLabel("Time: -")
        label_row.addWidget(self.video_position_label)
        label_row.addStretch(1)
        label_row.addWidget(self.video_time_label)

        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setEnabled(False)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(0)
        self.frame_slider.sliderPressed.connect(self._on_frame_slider_pressed)
        self.frame_slider.sliderReleased.connect(self._on_frame_slider_released)
        self.frame_slider.valueChanged.connect(self._on_frame_slider_value_changed)

        layout.addLayout(label_row)
        layout.addWidget(self.frame_slider)
        return widget

    def _build_folder_list_box(self) -> QGroupBox:
        box = QGroupBox("Folder Images")
        layout = QVBoxLayout(box)

        self.folder_list = QListWidget()
        self.folder_list.currentRowChanged.connect(self._on_folder_selection_changed)
        layout.addWidget(self.folder_list)
        return box

    def _build_info_box(self) -> QGroupBox:
        box = QGroupBox("Processing Info")
        layout = QVBoxLayout(box)

        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        layout.addWidget(self.info_text)
        return box

    def _open_image(self) -> None:
        if self.image_pipeline is None:
            QMessageBox.warning(self, "Model Required", "Select a valid model before opening an image.")
            return
        if self.folder_processing_active:
            QMessageBox.information(self, "Folder Processing", "Stop the current folder processing first.")
            return

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select image",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp);;All Files (*)",
        )
        if not path:
            return

        self._request_stop_folder_processing()
        self._stop_video_session()
        self.folder_image_paths = []
        self.folder_list.clear()
        self._set_folder_controls_enabled(False)
        try:
            self._process_and_display_image(Path(path))
        except Exception as error:
            self._show_error("Failed to process image", error)

    def _open_folder(self) -> None:
        if self.image_pipeline is None:
            QMessageBox.warning(self, "Model Required", "Select a valid model before opening a folder.")
            return
        if self.folder_processing_active:
            QMessageBox.information(self, "Folder Processing", "Stop the current folder processing first.")
            return

        folder_path = QFileDialog.getExistingDirectory(self, "Select image folder", "")
        if not folder_path:
            return

        self._request_stop_folder_processing()
        self._stop_video_session()
        try:
            image_paths = self.image_pipeline.list_image_paths(folder_path)
        except Exception as error:
            self._show_error("Failed to open folder", error)
            return

        self.folder_image_paths = image_paths
        self.folder_list.clear()
        for path in image_paths:
            self.folder_list.addItem(path.name)

        enabled = len(image_paths) > 0
        self._set_folder_controls_enabled(enabled)
        if not enabled:
            self.statusBar().showMessage(f"No supported images found in folder: {folder_path}")
            return

        self.current_folder_index = 0
        self.folder_list.setCurrentRow(0)

    def _open_video(self) -> None:
        if self.video_pipeline is None:
            QMessageBox.warning(self, "Model Required", "Select a valid model before opening a video.")
            return
        if self.folder_processing_active:
            QMessageBox.information(self, "Folder Processing", "Stop the current folder processing first.")
            return

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select video",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.wmv *.m4v);;All Files (*)",
        )
        if not path:
            return

        self._request_stop_folder_processing()
        self._stop_video_session()
        self.folder_image_paths = []
        self.folder_list.clear()
        self._set_folder_controls_enabled(False)

        try:
            assert self.video_pipeline is not None
            self.video_session = self.video_pipeline.open(path)
            self._set_video_controls_enabled(True)
            self._configure_video_slider()
            if self.video_session.stream_info.fps > 0:
                self.last_video_interval_ms = max(1, int(1000 / self.video_session.stream_info.fps))
            else:
                self.last_video_interval_ms = 33
            self._step_video_frame()
        except Exception as error:
            self._show_error("Failed to open video", error)
            self._set_video_controls_enabled(False)

    def _show_previous_folder_image(self) -> None:
        if not self.folder_image_paths or self.folder_processing_active:
            return
        self.current_folder_index = max(0, self.current_folder_index - 1)
        self.folder_list.setCurrentRow(self.current_folder_index)

    def _show_next_folder_image(self) -> None:
        if not self.folder_image_paths or self.folder_processing_active:
            return
        self.current_folder_index = min(len(self.folder_image_paths) - 1, self.current_folder_index + 1)
        self.folder_list.setCurrentRow(self.current_folder_index)

    def _start_folder_processing(self) -> None:
        if not self.folder_image_paths or self.folder_processing_active:
            return

        self._stop_video_session()
        start_index = self.current_folder_index if self.current_folder_index >= 0 else 0
        if self.selected_model_option is None:
            QMessageBox.warning(self, "Model Required", "Select a valid model before processing a folder.")
            return
        self.folder_processing_thread = QThread(self)
        self.folder_processing_worker = FolderProcessingWorker(
            settings=self.settings,
            image_paths=self.folder_image_paths,
            runtime=self.selected_model_option.runtime,
            model_path=self.selected_model_option.path,
            start_index=start_index,
        )
        self.folder_processing_worker.moveToThread(self.folder_processing_thread)

        self.folder_processing_thread.started.connect(self.folder_processing_worker.run)
        self.folder_processing_worker.frame_processed.connect(self._on_folder_frame_processed)
        self.folder_processing_worker.failed.connect(self._on_folder_processing_failed)
        self.folder_processing_worker.finished.connect(self._on_folder_processing_finished)
        self.folder_processing_worker.finished.connect(self.folder_processing_thread.quit)
        self.folder_processing_thread.finished.connect(self._cleanup_folder_processing_thread)

        self.folder_processing_active = True
        self._set_folder_processing_running_state(True)
        self.folder_processing_thread.start()
        self.statusBar().showMessage(
            f"Folder sequence processing started from image {start_index + 1}/{len(self.folder_image_paths)}."
        )

    def _request_stop_folder_processing(self) -> None:
        if self.folder_processing_worker is not None:
            self.folder_processing_worker.request_stop()
            self.statusBar().showMessage("Stop requested. Waiting for current image to finish...")

    def _on_folder_frame_processed(self, processed_frame: object, index: int, total: int) -> None:
        frame = processed_frame
        if not isinstance(frame, ProcessedFrame):
            return

        self.current_folder_index = index
        self.folder_list.blockSignals(True)
        self.folder_list.setCurrentRow(index)
        self.folder_list.blockSignals(False)
        self._display_processed_frame(frame, trajectories=None)
        self.statusBar().showMessage(f"Processed folder image {index + 1}/{total}")

    def _on_folder_processing_failed(self, message: str) -> None:
        self.folder_processing_active = False
        self._set_folder_processing_running_state(False)
        QMessageBox.critical(self, "Folder Processing Failed", message)
        self.statusBar().showMessage(f"Folder processing failed: {message}")
        if self.folder_processing_thread is not None:
            self.folder_processing_thread.quit()

    def _on_folder_processing_finished(self, processed_count: int, stopped: bool) -> None:
        self.folder_processing_active = False
        self._set_folder_processing_running_state(False)
        if stopped:
            self.statusBar().showMessage(f"Folder processing stopped after {processed_count} images.")
        else:
            self.statusBar().showMessage(f"Folder processing completed: {processed_count} images.")

    def _cleanup_folder_processing_thread(self) -> None:
        if self.folder_processing_worker is not None:
            self.folder_processing_worker.deleteLater()
            self.folder_processing_worker = None
        if self.folder_processing_thread is not None:
            self.folder_processing_thread.deleteLater()
            self.folder_processing_thread = None

    def _on_folder_selection_changed(self, row: int) -> None:
        if self.folder_processing_active:
            return
        if row < 0 or row >= len(self.folder_image_paths):
            return
        self.current_folder_index = row
        try:
            self._process_and_display_image(self.folder_image_paths[row])
        except Exception as error:
            self._show_error("Failed to process folder image", error)

    def _start_video_playback(self) -> None:
        if self.video_session is None:
            return
        self.playback_timer.start(self.last_video_interval_ms)
        self.statusBar().showMessage("Video playback started.")

    def _pause_video_playback(self) -> None:
        self.playback_timer.stop()
        self.statusBar().showMessage("Video playback paused.")

    def _step_video_frame(self) -> None:
        if self.video_session is None:
            return
        self._play_next_video_frame()

    def _play_next_video_frame(self) -> None:
        if self.video_session is None:
            return

        started_at = perf_counter()
        try:
            processed_frame = self.video_session.read_next_processed_frame()
        except Exception as error:
            self._pause_video_playback()
            self._show_error("Failed to process video frame", error)
            return

        if processed_frame is None:
            self._pause_video_playback()
            self.statusBar().showMessage("Reached end of video.")
            return

        elapsed = perf_counter() - started_at
        self.current_video_processing_fps = (1.0 / elapsed) if elapsed > 0 else None
        self.current_video_display_index = processed_frame.frame.frame_index
        self._display_processed_frame(
            processed_frame,
            trajectories=self.video_session.trajectories(),
        )
        self._sync_video_slider_to_current_frame()

    def _process_and_display_image(self, image_path: Path) -> None:
        if self.image_pipeline is None:
            raise RuntimeError("No active image pipeline. Select a valid model first.")
        self.current_video_processing_fps = None
        processed_frame = self.image_pipeline.process_image(image_path)
        self._display_processed_frame(processed_frame, trajectories=None)
        self.statusBar().showMessage(f"Processed image: {image_path}")

    def _display_processed_frame(
        self,
        processed_frame: ProcessedFrame,
        trajectories: dict[int, list[tuple[int, int]]] | None,
    ) -> None:
        self.current_processed_frame = processed_frame
        rendered_rgb = self.overlay_renderer.render(processed_frame, trajectories=trajectories)
        self._set_viewer_image(rendered_rgb)
        self._update_info_panel(processed_frame, trajectories=trajectories)

    def _set_viewer_image(self, image_rgb) -> None:
        height, width, channels = image_rgb.shape
        bytes_per_line = channels * width
        q_image = QImage(image_rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        scaled = pixmap.scaled(
            self.viewer_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.viewer_label.setPixmap(scaled)

    def _update_info_panel(
        self,
        processed_frame: ProcessedFrame,
        trajectories: dict[int, list[tuple[int, int]]] | None,
    ) -> None:
        frame = processed_frame.frame
        result = processed_frame.result
        lines = [
            f"Model runtime: {self.selected_model_option.runtime.value if self.selected_model_option else '-'}",
            f"Model path: {self.selected_model_option.path if self.selected_model_option else '-'}",
            f"Source: {frame.source_path}",
            f"Kind: {frame.kind}",
            f"Frame index: {frame.frame_index}",
        ]

        if frame.timestamp_seconds is not None:
            lines.append(f"Timestamp: {frame.timestamp_seconds:.3f} s")
        if frame.sequence_index is not None and frame.sequence_length is not None:
            lines.append(f"Folder index: {frame.sequence_index + 1}/{frame.sequence_length}")
        if self.video_session is not None:
            lines.append(f"Video FPS: {self.video_session.stream_info.fps:.3f}")
        if self.current_video_processing_fps is not None:
            lines.append(f"Processing FPS: {self.current_video_processing_fps:.3f}")

        lines.extend(
            [
                f"Image size (proc): {result.image_size}",
                f"Image size (orig): {result.original_image_size}",
                f"Contours: {len(result.contours)}",
                f"Detected tools: {len(result.tools)}",
            ]
        )

        if trajectories:
            lines.append(f"Active tracks: {len(trajectories)}")

        for tool in result.tools:
            lines.append(
                f"- contour={tool.contour_index}, track={tool.track_id}, "
                f"center={tool.center}, tip={tool.tip}, bbox={tool.bounding_box}"
            )

        if result.error_message:
            lines.append(f"Error: {result.error_message}")

        self.info_text.setPlainText("\n".join(lines))

    def _set_folder_controls_enabled(self, enabled: bool) -> None:
        base_enabled = enabled and self.image_pipeline is not None
        self.open_image_button.setEnabled(self.image_pipeline is not None)
        self.open_folder_button.setEnabled(self.image_pipeline is not None)
        self.prev_image_button.setEnabled(base_enabled)
        self.next_image_button.setEnabled(base_enabled)
        self.process_all_button.setEnabled(base_enabled)
        self.folder_list.setEnabled(base_enabled)

    def _set_folder_processing_running_state(self, running: bool) -> None:
        model_ready = self.image_pipeline is not None
        self.open_image_button.setEnabled(not running and model_ready)
        self.open_folder_button.setEnabled(not running and model_ready)
        self.prev_image_button.setEnabled(not running and model_ready and bool(self.folder_image_paths))
        self.next_image_button.setEnabled(not running and model_ready and bool(self.folder_image_paths))
        self.process_all_button.setEnabled(not running and model_ready and bool(self.folder_image_paths))
        self.folder_list.setEnabled(not running)
        self.stop_process_button.setEnabled(running)

    def _set_video_controls_enabled(self, enabled: bool) -> None:
        self.open_video_button.setEnabled(self.video_pipeline is not None)
        self.play_button.setEnabled(enabled)
        self.pause_button.setEnabled(enabled)
        self.next_frame_button.setEnabled(enabled)
        if hasattr(self, "frame_slider"):
            self.frame_slider.setEnabled(enabled)

    def _stop_video_session(self) -> None:
        self.playback_timer.stop()
        if self.video_session is not None:
            self.video_session.close()
            self.video_session = None
        self.current_video_processing_fps = None
        self.current_video_display_index = 0
        self._set_video_controls_enabled(False)
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(0)
        self.frame_slider.blockSignals(False)
        self.video_position_label.setText("Frame: - / -")
        self.video_time_label.setText("Time: -")

    def _show_error(self, title: str, error: Exception) -> None:
        QMessageBox.critical(self, title, str(error))
        self.statusBar().showMessage(f"{title}: {error}")

    @staticmethod
    def _status_text(status: DeviceStatus) -> str:
        if status.ready:
            return f"Ready on {status.device_label}"
        return f"GPU check failed: {status.reason}"

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self.current_processed_frame is not None:
            trajectories = self.video_session.trajectories() if self.video_session is not None else None
            rendered_rgb = self.overlay_renderer.render(self.current_processed_frame, trajectories=trajectories)
            self._set_viewer_image(rendered_rgb)

    def _configure_video_slider(self) -> None:
        if self.video_session is None:
            return
        frame_count = max(1, self.video_session.stream_info.frame_count)
        self.frame_slider.blockSignals(True)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(frame_count - 1)
        self.frame_slider.setValue(0)
        self.frame_slider.blockSignals(False)
        self._update_video_slider_labels(0)

    def _sync_video_slider_to_current_frame(self) -> None:
        if self.video_session is None or self._slider_is_pressed:
            return
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(self.current_video_display_index)
        self.frame_slider.blockSignals(False)
        self._update_video_slider_labels(self.current_video_display_index)

    def _update_video_slider_labels(self, frame_index: int) -> None:
        if self.video_session is None:
            self.video_position_label.setText("Frame: - / -")
            self.video_time_label.setText("Time: -")
            return

        total = max(1, self.video_session.stream_info.frame_count)
        fps = self.video_session.stream_info.fps
        self.video_position_label.setText(f"Frame: {frame_index + 1} / {total}")
        if fps > 0:
            self.video_time_label.setText(f"Time: {frame_index / fps:.3f} s")
        else:
            self.video_time_label.setText("Time: -")

    def _on_frame_slider_pressed(self) -> None:
        self._slider_is_pressed = True

    def _on_frame_slider_value_changed(self, value: int) -> None:
        self._update_video_slider_labels(value)

    def _on_frame_slider_released(self) -> None:
        self._slider_is_pressed = False
        if self.video_session is None:
            return

        target_frame = self.frame_slider.value()
        was_playing = self.playback_timer.isActive()
        self.playback_timer.stop()
        if not self.video_session.seek(target_frame):
            self.statusBar().showMessage(f"Failed to seek to frame {target_frame}")
            return

        self.current_video_display_index = target_frame
        self._play_next_video_frame()
        if was_playing and self.video_session is not None:
            self.playback_timer.start(self.last_video_interval_ms)

    def closeEvent(self, event) -> None:  # noqa: N802
        self._request_stop_folder_processing()
        if self.folder_processing_thread is not None:
            self.folder_processing_thread.quit()
            self.folder_processing_thread.wait()
        self._stop_video_session()
        event.accept()

    def _initialize_model_selection(self) -> None:
        self.model_options = self._discover_model_options()
        self.model_selector.blockSignals(True)
        self.model_selector.clear()
        for option in self.model_options:
            self.model_selector.addItem(option.label)

        if not self.model_options:
            self.model_selector.blockSignals(False)
            self.selected_model_option = None
            self.model_path_label.setText("Path: no model detected")
            self.model_status_label.setText(
                "Status: none of model.pt, model-fp32.trt, model-fp16.trt, or model-int8.trt was found"
            )
            self._clear_current_media_view()
            self._set_folder_controls_enabled(False)
            self._set_video_controls_enabled(False)
            return

        self.model_selector.setCurrentIndex(0)
        self.model_selector.blockSignals(False)
        self._activate_model_option(self.model_options[0], restart_media=False)

    def _discover_model_options(self) -> list[ModelOption]:
        options: list[ModelOption] = []
        if self.settings.local_model_path.exists():
            options.append(ModelOption(runtime=ModelRuntime.PYTORCH, path=self.settings.local_model_path))
        trt_paths = [
            self.settings.local_trt_fp32_model_path,
            self.settings.local_trt_fp16_model_path,
            self.settings.local_trt_int8_model_path,
        ]
        for trt_path in trt_paths:
            if trt_path.exists():
                options.append(ModelOption(runtime=ModelRuntime.TENSORRT, path=trt_path))
        return options

    def _build_segmenter_for_option(self, option: ModelOption) -> Segmenter:
        if option.runtime == ModelRuntime.TENSORRT:
            return TensorRTToolSegmenter(settings=self.settings, engine_path=option.path)
        return MonaiToolSegmenter(settings=self.settings, model_path=option.path)

    def _on_model_selection_changed(self, index: int) -> None:
        if self._switching_model or index < 0 or index >= len(self.model_options):
            return
        option = self.model_options[index]
        try:
            self._switching_model = True
            self._activate_model_option(option, restart_media=True)
        except Exception as error:
            self.model_status_label.setText(f"Status: failed to load model ({error})")
            self._show_error("Failed to switch model", error)
            if self.selected_model_option is not None:
                previous_index = self.model_options.index(self.selected_model_option)
                self.model_selector.blockSignals(True)
                self.model_selector.setCurrentIndex(previous_index)
                self.model_selector.blockSignals(False)
        finally:
            self._switching_model = False

    def _activate_model_option(self, option: ModelOption, restart_media: bool) -> None:
        previous_video_path: Path | None = None
        previous_video_frame = 0
        previous_video_playing = False
        previous_image_path: Path | None = None
        previous_folder_row = self.current_folder_index

        if restart_media:
            previous_video_path, previous_video_frame, previous_video_playing = self._capture_video_restart_state()
            previous_image_path = self._capture_image_restart_state()
            self._stop_folder_processing_blocking()
            self._stop_video_session()

        segmenter = self._build_segmenter_for_option(option)
        model_info = segmenter.load()
        self.segmenter = segmenter
        self.image_pipeline = ImagePipeline(settings=self.settings, segmenter=segmenter)
        self.video_pipeline = VideoPipeline(settings=self.settings, segmenter=segmenter, tracker=self.video_tracker)
        self.selected_model_option = option

        self.model_path_label.setText(f"Path: {option.path}")
        self.model_status_label.setText(f"Status: loaded {model_info.runtime} on {model_info.device}")
        self._set_folder_controls_enabled(bool(self.folder_image_paths))
        self._set_video_controls_enabled(False)
        self.statusBar().showMessage(f"Active model: {option.path.name} ({option.runtime.value})")

        if restart_media:
            self._restart_media_after_model_change(
                previous_video_path=previous_video_path,
                previous_video_frame=previous_video_frame,
                previous_video_playing=previous_video_playing,
                previous_image_path=previous_image_path,
                previous_folder_row=previous_folder_row,
            )

    def _capture_video_restart_state(self) -> tuple[Path | None, int, bool]:
        if self.video_session is None:
            return None, 0, False
        return (
            self.video_session.stream_info.source_path,
            self.current_video_display_index,
            self.playback_timer.isActive(),
        )

    def _capture_image_restart_state(self) -> Path | None:
        if self.folder_image_paths:
            return None
        if self.current_processed_frame is None:
            return None
        if self.current_processed_frame.frame.kind.value.startswith("image"):
            return self.current_processed_frame.frame.source_path
        return None

    def _stop_folder_processing_blocking(self) -> None:
        if self.folder_processing_thread is None:
            return
        self._request_stop_folder_processing()
        self.folder_processing_thread.wait()

    def _restart_media_after_model_change(
        self,
        previous_video_path: Path | None,
        previous_video_frame: int,
        previous_video_playing: bool,
        previous_image_path: Path | None,
        previous_folder_row: int,
    ) -> None:
        if previous_video_path is not None:
            self._reopen_video_after_model_change(previous_video_path, previous_video_frame, previous_video_playing)
            return
        if self.folder_image_paths and 0 <= previous_folder_row < len(self.folder_image_paths):
            self.current_folder_index = previous_folder_row
            if self.folder_list.currentRow() != previous_folder_row:
                self.folder_list.setCurrentRow(previous_folder_row)
            else:
                self._process_and_display_image(self.folder_image_paths[previous_folder_row])
            return
        if previous_image_path is not None:
            self._process_and_display_image(previous_image_path)
            return
        self._clear_current_media_view()

    def _reopen_video_after_model_change(self, video_path: Path, frame_index: int, was_playing: bool) -> None:
        if self.video_pipeline is None:
            return
        self.video_session = self.video_pipeline.open(video_path)
        self._set_video_controls_enabled(True)
        self._configure_video_slider()
        if self.video_session.stream_info.fps > 0:
            self.last_video_interval_ms = max(1, int(1000 / self.video_session.stream_info.fps))
        else:
            self.last_video_interval_ms = 33

        target_frame = min(frame_index, max(0, self.video_session.stream_info.frame_count - 1))
        if target_frame > 0:
            self.video_session.seek(target_frame)
        self.current_video_display_index = target_frame
        self._play_next_video_frame()
        if was_playing:
            self.playback_timer.start(self.last_video_interval_ms)

    def _clear_current_media_view(self) -> None:
        self.current_processed_frame = None
        self.current_video_processing_fps = None
        self.viewer_label.clear()
        self.viewer_label.setText("Load an image, folder, or video to begin.")
        self.info_text.clear()
