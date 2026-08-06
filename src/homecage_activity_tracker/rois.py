"""ROI configuration parsing and polygon utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_BODYPARTS_BY_NAME = {
    "food": ("nose",),
    "water": ("nose",),
    "water_1": ("nose",),
    "water_2": ("nose",),
    "wheel": ("mouse_center",),
    "tunnel": ("mouse_center",),
    "house": ("mouse_center",),
    "house_entrance": ("mouse_center",),
}


@dataclass(frozen=True)
class ROIConfig:
    name: str
    polygon: tuple[tuple[float, float], ...]
    bodyparts: tuple[str, ...]
    count_rule: str = "any"
    description: str = ""


@dataclass(frozen=True)
class ROISet:
    rois: tuple[ROIConfig, ...]
    image_size: tuple[int, int] | None
    source_path: Path


def _polygon_from_value(value: Any, name: str) -> tuple[tuple[float, float], ...]:
    if (
        isinstance(value, list)
        and len(value) == 4
        and all(isinstance(item, (int, float)) for item in value)
    ):
        x1, y1, x2, y2 = (float(item) for item in value)
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"ROI '{name}' has an invalid rectangle: {value}")
        return ((x1, y1), (x2, y1), (x2, y2), (x1, y2))

    if not isinstance(value, list) or len(value) < 3:
        raise ValueError(f"ROI '{name}' requires a rectangle or at least 3 polygon points.")

    polygon = []
    for point in value:
        if (
            not isinstance(point, (list, tuple))
            or len(point) != 2
            or not all(isinstance(item, (int, float)) for item in point)
        ):
            raise ValueError(f"ROI '{name}' has an invalid point: {point}")
        polygon.append((float(point[0]), float(point[1])))
    return tuple(polygon)


def _normalize_image_size(value: Any) -> tuple[int, int] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("image_size must be [width, height].")
    width, height = int(value[0]), int(value[1])
    if width <= 0 or height <= 0:
        raise ValueError("image_size values must be positive.")
    return width, height


def load_rois(path: str | Path) -> ROISet:
    """Load an extended or legacy ROI JSON file.

    Extended files use ``{"image_size": [...], "rois": {...}}``. Legacy files
    may map ROI names directly to rectangles or polygons.
    """

    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError("ROI JSON must contain an object at the top level.")

    if "rois" in payload:
        roi_payload = payload["rois"]
        image_size = _normalize_image_size(payload.get("image_size"))
    else:
        roi_payload = payload
        image_size = None

    if not isinstance(roi_payload, dict) or not roi_payload:
        raise ValueError("ROI JSON does not define any ROIs.")

    rois = []
    for name, value in roi_payload.items():
        if isinstance(value, dict):
            if "polygon" not in value:
                raise ValueError(f"ROI '{name}' is missing 'polygon'.")
            polygon_value = value["polygon"]
            bodyparts_value = value.get(
                "bodyparts", DEFAULT_BODYPARTS_BY_NAME.get(name, ("mouse_center",))
            )
            count_rule = str(value.get("count_rule", "any")).lower()
            description = str(value.get("description", ""))
        else:
            polygon_value = value
            bodyparts_value = DEFAULT_BODYPARTS_BY_NAME.get(name, ("mouse_center",))
            count_rule = "any"
            description = ""

        if isinstance(bodyparts_value, str):
            bodyparts = (bodyparts_value,)
        else:
            bodyparts = tuple(str(item) for item in bodyparts_value)
        if not bodyparts:
            raise ValueError(f"ROI '{name}' must select at least one bodypart.")
        if count_rule not in {"any", "majority", "all"}:
            raise ValueError(
                f"ROI '{name}' has invalid count_rule '{count_rule}'. "
                "Use any, majority, or all."
            )

        polygon = _polygon_from_value(polygon_value, name)
        rois.append(
            ROIConfig(
                name=str(name),
                polygon=polygon,
                bodyparts=bodyparts,
                count_rule=count_rule,
                description=description,
            )
        )

    if image_size is not None:
        width, height = image_size
        for roi in rois:
            for x, y in roi.polygon:
                if x < 0 or y < 0 or x >= width or y >= height:
                    raise ValueError(
                        f"ROI '{roi.name}' point {(x, y)} is outside image_size {image_size}."
                    )

    return ROISet(rois=tuple(rois), image_size=image_size, source_path=source)


def points_in_polygon(
    x: np.ndarray,
    y: np.ndarray,
    polygon: tuple[tuple[float, float], ...],
) -> np.ndarray:
    """Vectorized point-in-polygon test using the even-odd rule.

    Points on a polygon edge are considered inside.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    vertices = np.asarray(polygon, dtype=float)
    inside = np.zeros(x.shape, dtype=bool)
    on_boundary = np.zeros(x.shape, dtype=bool)
    epsilon = 1e-9

    xj, yj = vertices[-1]
    for xi, yi in vertices:
        dx = xi - xj
        dy = yi - yj
        cross = (x - xj) * dy - (y - yj) * dx
        within_x = (x >= min(xi, xj) - epsilon) & (x <= max(xi, xj) + epsilon)
        within_y = (y >= min(yi, yj) - epsilon) & (y <= max(yi, yj) + epsilon)
        on_boundary |= (np.abs(cross) <= epsilon) & within_x & within_y

        intersects = ((yi > y) != (yj > y)) & (
            x < (xj - xi) * (y - yi) / ((yj - yi) + epsilon) + xi
        )
        inside ^= intersects
        xj, yj = xi, yi

    return inside | on_boundary
