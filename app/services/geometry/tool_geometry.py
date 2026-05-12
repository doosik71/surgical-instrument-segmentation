"""Contour-based geometry extraction for surgical tools."""

from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np
from scipy.spatial.distance import cdist

from app.domain.models import FrameResult, Point, ToolGeometry


def _as_binary_mask(mask: np.ndarray) -> np.ndarray:
    """Normalize a mask to a uint8 binary image."""
    if mask.ndim != 2:
        raise ValueError("Expected a 2D mask")

    return (mask > 0).astype(np.uint8) * 255


def postprocess_binary_mask(
    mask: np.ndarray,
    kernel_size: int = 5,
    min_component_area: int = 400,
) -> np.ndarray:
    """Clean a binary mask before contour extraction."""
    binary = _as_binary_mask(mask)

    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    filtered = np.zeros_like(cleaned)

    for label_index in range(1, num_labels):
        area = stats[label_index, cv2.CC_STAT_AREA]
        if area >= min_component_area:
            filtered[labels == label_index] = 255

    return filtered


def extract_contours(mask: np.ndarray, min_contour_area: int = 400) -> list[np.ndarray]:
    """Extract external contours from a binary mask."""
    binary = _as_binary_mask(mask)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [contour for contour in contours if cv2.contourArea(contour) >= min_contour_area]


def contour_points(contour: np.ndarray) -> np.ndarray:
    """Return a contour as an (N, 2) array."""
    points = contour.reshape(-1, 2)
    if len(points) < 2:
        raise ValueError("Contour must contain at least two points")
    return points


def contour_center(contour: np.ndarray) -> Point:
    """Compute the contour center using image moments."""
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        points = contour_points(contour)
        fallback = np.mean(points, axis=0)
        return int(fallback[0]), int(fallback[1])

    x = int(moments["m10"] / moments["m00"])
    y = int(moments["m01"] / moments["m00"])
    return x, y


def farthest_point(points: np.ndarray, reference: Iterable[float]) -> np.ndarray:
    """Return the contour point farthest from a reference point."""
    reference_array = np.asarray([reference], dtype=np.float32)
    distances = cdist(points.astype(np.float32), reference_array).ravel()
    return points[int(np.argmax(distances))]


def axis_endpoints(contour: np.ndarray) -> tuple[Point, Point, Point]:
    """Compute center, endpoint A, and endpoint B for a contour."""
    points = contour_points(contour)
    center = contour_center(contour)
    endpoint_a = farthest_point(points, center)
    endpoint_b = farthest_point(points, endpoint_a)

    return (
        center,
        (int(endpoint_a[0]), int(endpoint_a[1])),
        (int(endpoint_b[0]), int(endpoint_b[1])),
    )


def select_tip(axis_start: Point, axis_end: Point, image_size: tuple[int, int]) -> Point:
    """Select the endpoint closer to the image center as the tip."""
    image_height, image_width = image_size
    image_center = np.array([[image_width / 2.0, image_height / 2.0]], dtype=np.float32)
    endpoints = np.array([axis_start, axis_end], dtype=np.float32)
    distances = cdist(endpoints, image_center).ravel()
    selected = axis_start if distances[0] <= distances[1] else axis_end
    return int(selected[0]), int(selected[1])


def build_tool_geometry(
    contour: np.ndarray,
    contour_index: int,
    image_size: tuple[int, int],
) -> ToolGeometry:
    """Create a tool-geometry object from one contour."""
    center, axis_start, axis_end = axis_endpoints(contour)
    tip = select_tip(axis_start, axis_end, image_size=image_size)
    bounding_box = cv2.boundingRect(contour)

    return ToolGeometry(
        contour_index=contour_index,
        contour=contour,
        area=float(cv2.contourArea(contour)),
        bounding_box=bounding_box,
        center=center,
        axis_start=axis_start,
        axis_end=axis_end,
        tip=tip,
    )


def extract_tool_geometries(contours: list[np.ndarray], image_size: tuple[int, int]) -> list[ToolGeometry]:
    """Convert contours into structured tool geometries."""
    return [
        build_tool_geometry(contour=contour, contour_index=index, image_size=image_size)
        for index, contour in enumerate(contours)
    ]


def extract_tool_geometries_from_mask(
    mask: np.ndarray,
    image_size: tuple[int, int] | None = None,
    kernel_size: int = 5,
    min_component_area: int = 400,
    min_contour_area: int = 400,
) -> FrameResult:
    """Postprocess a mask and derive contour geometry."""
    cleaned_mask = postprocess_binary_mask(
        mask=mask,
        kernel_size=kernel_size,
        min_component_area=min_component_area,
    )
    resolved_image_size = image_size or cleaned_mask.shape[:2]
    contours = extract_contours(cleaned_mask, min_contour_area=min_contour_area)
    tools = extract_tool_geometries(contours=contours, image_size=resolved_image_size)

    return FrameResult(
        image_size=resolved_image_size,
        mask=(cleaned_mask > 0).astype(np.uint8),
        contours=contours,
        tools=tools,
    )
