"""Overlay rendering for still images and video frames."""

from __future__ import annotations

import cv2
import numpy as np

from app.domain.models import ProcessedFrame


class OverlayRenderer:
    """Render segmentation and tracking overlays onto RGB frames."""

    def __init__(self, mask_alpha: float = 0.35) -> None:
        self.mask_alpha = mask_alpha
        self.trajectory_color = (30, 144, 255)

    def render(
        self,
        processed_frame: ProcessedFrame,
        trajectories: dict[int, list[tuple[int, int]]] | None = None,
    ) -> np.ndarray:
        """Render mask, contour, axis, tip, and track overlays."""
        base_rgb = processed_frame.frame.image_rgb.copy()
        overlay = base_rgb.copy()
        result = processed_frame.result

        if result.mask is not None:
            mask = (result.mask > 0)
            overlay[mask] = np.array([40, 230, 90], dtype=np.uint8)
            base_rgb = cv2.addWeighted(overlay, self.mask_alpha, base_rgb, 1.0 - self.mask_alpha, 0.0)

        for contour in result.contours:
            cv2.drawContours(base_rgb, [contour], -1, (255, 215, 0), 2)

        for tool in result.tools:
            cv2.line(base_rgb, tool.axis_start, tool.axis_end, (0, 255, 255), 2)
            cv2.circle(base_rgb, tool.center, 4, (80, 160, 255), -1)
            cv2.circle(base_rgb, tool.tip, 7, (255, 80, 80), -1)
            x, y, w, h = tool.bounding_box
            cv2.rectangle(base_rgb, (x, y), (x + w, y + h), (140, 140, 255), 1)

            label_parts = [f"C{tool.contour_index}"]
            if tool.track_id is not None:
                label_parts.append(f"T{tool.track_id}")
            label = " ".join(label_parts)
            cv2.putText(
                base_rgb,
                label,
                (x, max(20, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        if trajectories:
            for track_id, points in trajectories.items():
                if len(points) < 2:
                    continue
                polyline = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(base_rgb, [polyline], False, self.trajectory_color, 2, cv2.LINE_AA)
                last_x, last_y = points[-1]
                cv2.putText(
                    base_rgb,
                    f"T{track_id}",
                    (last_x + 8, last_y + 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    self.trajectory_color,
                    2,
                    cv2.LINE_AA,
                )

        return base_rgb
