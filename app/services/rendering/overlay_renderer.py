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
        meta = processed_frame.frame.processing_metadata

        def to_proc(pt):
            if meta is None:
                return pt
            scale, pad = meta
            return (int(pt[0] * scale + pad[0]), int(pt[1] * scale + pad[1]))

        def to_proc_cnt(cnt):
            if meta is None:
                return cnt
            scale, pad = meta
            m_cnt = cnt.copy().astype(np.float32)
            m_cnt[:, 0, 0] = m_cnt[:, 0, 0] * scale + pad[0]
            m_cnt[:, 0, 1] = m_cnt[:, 0, 1] * scale + pad[1]
            return m_cnt.astype(np.int32)

        if result.mask is not None:
            mask = (result.mask > 0)
            overlay[mask] = np.array([40, 230, 90], dtype=np.uint8)
            base_rgb = cv2.addWeighted(overlay, self.mask_alpha, base_rgb, 1.0 - self.mask_alpha, 0.0)

        for contour in result.contours:
            cv2.drawContours(base_rgb, [to_proc_cnt(contour)], -1, (255, 215, 0), 2)

        for tool in result.tools:
            proc_start = to_proc(tool.axis_start)
            proc_end = to_proc(tool.axis_end)
            proc_center = to_proc(tool.center)
            proc_tip = to_proc(tool.tip)
            
            cv2.line(base_rgb, proc_start, proc_end, (0, 255, 255), 2)
            cv2.circle(base_rgb, proc_center, 4, (80, 160, 255), -1)
            cv2.circle(base_rgb, proc_tip, 7, (255, 80, 80), -1)
            
            # Map bounding box back to processed space
            x, y, w, h = tool.bounding_box
            p1 = to_proc((x, y))
            p2 = to_proc((x + w, y + h))
            cv2.rectangle(base_rgb, p1, p2, (140, 140, 255), 1)

            label_parts = [f"C{tool.contour_index}"]
            if tool.track_id is not None:
                label_parts.append(f"T{tool.track_id}")
            label = " ".join(label_parts)
            cv2.putText(
                base_rgb,
                label,
                (p1[0], max(20, p1[1] - 6)),
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
                proc_points = [to_proc(pt) for pt in points]
                polyline = np.array(proc_points, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(base_rgb, [polyline], False, self.trajectory_color, 2, cv2.LINE_AA)
                last_x, last_y = proc_points[-1]
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
