"""Write provenance, plots, and ROI overlays."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .rois import ROISet


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def plot_summary(summary: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    pooled = summary.loc[summary["raw_slot"] == "ALL_RAW_SLOTS"].copy()
    pooled = pooled.sort_values("roi")

    occupancy_path = output_dir / "roi_occupancy_pct.png"
    fig, ax = plt.subplots(figsize=(max(7, 1.15 * len(pooled)), 4.8))
    ax.bar(pooled["roi"], pooled["occupancy_pct_of_analyzable"], color="#2A6F97")
    ax.set_ylabel("Visible occupancy (% of analyzable slot-time)")
    ax.set_xlabel("ROI")
    ax.set_ylim(0, 100)
    ax.set_title("ROI occupancy from raw DLC detection slots")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(occupancy_path, dpi=180)
    plt.close(fig)

    coverage_path = output_dir / "roi_tracking_coverage_pct.png"
    fig, ax = plt.subplots(figsize=(max(7, 1.15 * len(pooled)), 4.8))
    colors = np.where(pooled["coverage_pct"] >= 80, "#2A9D8F", "#E9C46A")
    ax.bar(pooled["roi"], pooled["coverage_pct"], color=colors)
    ax.axhline(80, color="#6C757D", linestyle="--", linewidth=1, label="80% reference")
    ax.set_ylabel("Analyzable slot-time (%)")
    ax.set_xlabel("ROI")
    ax.set_ylim(0, 100)
    ax.set_title("Tracking coverage used in each ROI estimate")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(coverage_path, dpi=180)
    plt.close(fig)
    return occupancy_path, coverage_path


def save_roi_overlay(reference_image: Path, roi_set: ROISet, output_path: Path) -> None:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - package dependency
        raise RuntimeError("ROI overlays require opencv-python-headless.") from exc

    image = cv2.imread(str(reference_image), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read reference image: {reference_image}")
    height, width = image.shape[:2]
    if roi_set.image_size is not None and roi_set.image_size != (width, height):
        raise ValueError(
            f"Reference image is {(width, height)}, but ROI image_size is {roi_set.image_size}."
        )

    palette = [
        (52, 152, 219),
        (46, 204, 113),
        (231, 76, 60),
        (241, 196, 15),
        (155, 89, 182),
        (230, 126, 34),
        (26, 188, 156),
    ]
    for index, roi in enumerate(roi_set.rois):
        color = palette[index % len(palette)]
        points = np.asarray(roi.polygon, dtype=np.int32)
        cv2.polylines(image, [points], True, color, 2, cv2.LINE_AA)
        x, y = points[0]
        cv2.putText(
            image,
            roi.name,
            (int(x), max(18, int(y) - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(f"Could not write ROI overlay: {output_path}")
