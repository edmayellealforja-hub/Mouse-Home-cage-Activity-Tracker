"""Quality-aware ROI occupancy calculations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .dlc import DLCData, coordinate_arrays
from .rois import ROISet, points_in_polygon


@dataclass(frozen=True)
class AnalysisResult:
    frame_summary: pd.DataFrame
    slot_summary: pd.DataFrame
    slot_frame_table: pd.DataFrame | None
    warnings: tuple[str, ...]


def _resolve_bodyparts(
    data: DLCData,
    raw_slot: str,
    requested: tuple[str, ...],
) -> tuple[str, ...]:
    available = {
        bodypart
        for slot, bodypart, coord in data.table.columns
        if slot == raw_slot and coord == "x"
    }
    selected = tuple(bodypart for bodypart in requested if bodypart in available)
    if not selected:
        raise ValueError(
            f"Raw slot '{raw_slot}' has none of the requested bodyparts {requested}. "
            f"Available bodyparts include: {sorted(available)}"
        )
    return selected


def _combine_points(
    valid_points: np.ndarray,
    inside_points: np.ndarray,
    count_rule: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return analyzable and inside masks for one slot and ROI."""

    if count_rule == "any":
        analyzable = valid_points.any(axis=0)
        inside = inside_points.any(axis=0)
    elif count_rule == "majority":
        analyzable = valid_points.all(axis=0)
        inside = inside_points.sum(axis=0) > (inside_points.shape[0] / 2.0)
    elif count_rule == "all":
        analyzable = valid_points.all(axis=0)
        inside = inside_points.all(axis=0)
    else:  # pragma: no cover - validated while loading ROIs
        raise ValueError(f"Unsupported count rule: {count_rule}")
    return analyzable, inside & analyzable


def analyze_occupancy(
    data: DLCData,
    roi_set: ROISet,
    *,
    fps: float,
    likelihood_threshold: float = 0.6,
    expected_mice: int | None = None,
    excluded_frames: set[int] | None = None,
    save_slot_frame_table: bool = False,
) -> AnalysisResult:
    """Calculate visible point occupancy for every raw DLC slot and ROI.

    Missing or excluded frames are marked not analyzable. They never enter the
    denominator as zero occupancy.
    """

    if fps <= 0:
        raise ValueError("fps must be greater than zero.")
    if not 0 <= likelihood_threshold <= 1:
        raise ValueError("likelihood_threshold must be between 0 and 1.")
    if expected_mice is not None and expected_mice != len(data.raw_slots):
        raise ValueError(
            f"Expected {expected_mice} mice, but the DLC file contains "
            f"{len(data.raw_slots)} raw slots: {list(data.raw_slots)}"
        )

    frame_count = len(data.table)
    excluded = np.zeros(frame_count, dtype=bool)
    invalid_excluded_frames = []
    for frame in excluded_frames or set():
        if 0 <= int(frame) < frame_count:
            excluded[int(frame)] = True
        else:
            invalid_excluded_frames.append(int(frame))

    frame_summary = pd.DataFrame(
        {
            "frame_index": np.arange(frame_count, dtype=int),
            "time_s": np.arange(frame_count, dtype=float) / float(fps),
            "excluded_by_qc": excluded,
        }
    )
    slot_rows: list[dict[str, object]] = []
    slot_frame_rows: list[pd.DataFrame] = []
    warnings = [
        "Raw DLC slot labels are detector slots, not verified persistent mouse identities."
    ]
    if invalid_excluded_frames:
        warnings.append(
            f"Ignored {len(invalid_excluded_frames)} excluded frame indices outside the data range."
        )

    for roi in roi_set.rois:
        if roi.name.lower() == "house":
            warnings.append(
                "The house ROI reports visible point occupancy only. It does not infer "
                "total time inside an opaque or covered house."
            )
        observed_matrix = np.zeros((len(data.raw_slots), frame_count), dtype=bool)
        inside_matrix = np.zeros((len(data.raw_slots), frame_count), dtype=bool)

        for slot_index, raw_slot in enumerate(data.raw_slots):
            bodyparts = _resolve_bodyparts(data, raw_slot, roi.bodyparts)
            if bodyparts != roi.bodyparts:
                warnings.append(
                    f"ROI '{roi.name}', slot '{raw_slot}' used available subset "
                    f"{list(bodyparts)} of requested bodyparts {list(roi.bodyparts)}."
                )

            valid_points = []
            inside_points = []
            for bodypart in bodyparts:
                x, y, likelihood = coordinate_arrays(data, raw_slot, bodypart)
                valid = (
                    np.isfinite(x)
                    & np.isfinite(y)
                    & np.isfinite(likelihood)
                    & (likelihood >= likelihood_threshold)
                )
                inside = valid & points_in_polygon(x, y, roi.polygon)
                valid_points.append(valid)
                inside_points.append(inside)

            observed, inside = _combine_points(
                np.vstack(valid_points), np.vstack(inside_points), roi.count_rule
            )
            observed &= ~excluded
            inside &= observed
            observed_matrix[slot_index] = observed
            inside_matrix[slot_index] = inside

            analyzable_frames = int(observed.sum())
            inside_frames = int(inside.sum())
            slot_rows.append(
                {
                    "roi": roi.name,
                    "raw_slot": raw_slot,
                    "bodyparts": ";".join(bodyparts),
                    "count_rule": roi.count_rule,
                    "total_frames": frame_count,
                    "analyzable_frames": analyzable_frames,
                    "inside_frames": inside_frames,
                    "coverage_pct": 100.0 * analyzable_frames / frame_count
                    if frame_count
                    else np.nan,
                    "occupancy_pct_of_analyzable": 100.0 * inside_frames / analyzable_frames
                    if analyzable_frames
                    else np.nan,
                    "analyzable_seconds": analyzable_frames / fps,
                    "inside_seconds": inside_frames / fps,
                }
            )

            if save_slot_frame_table:
                slot_frame_rows.append(
                    pd.DataFrame(
                        {
                            "frame_index": np.arange(frame_count, dtype=int),
                            "time_s": np.arange(frame_count, dtype=float) / float(fps),
                            "roi": roi.name,
                            "raw_slot": raw_slot,
                            "analyzable": observed,
                            "inside": pd.array(
                                np.where(observed, inside.astype(object), pd.NA),
                                dtype="boolean",
                            ),
                        }
                    )
                )

        observed_slots = observed_matrix.sum(axis=0).astype(int)
        counts = inside_matrix.sum(axis=0).astype(float)
        counts[observed_slots == 0] = np.nan
        frame_summary[f"observed_slots__{roi.name}"] = observed_slots
        frame_summary[f"count_in__{roi.name}"] = counts

        pooled_analyzable = int(observed_matrix.sum())
        pooled_inside = int(inside_matrix.sum())
        possible_slot_frames = frame_count * len(data.raw_slots)
        slot_rows.append(
            {
                "roi": roi.name,
                "raw_slot": "ALL_RAW_SLOTS",
                "bodyparts": ";".join(roi.bodyparts),
                "count_rule": roi.count_rule,
                "total_frames": possible_slot_frames,
                "analyzable_frames": pooled_analyzable,
                "inside_frames": pooled_inside,
                "coverage_pct": 100.0 * pooled_analyzable / possible_slot_frames
                if possible_slot_frames
                else np.nan,
                "occupancy_pct_of_analyzable": 100.0 * pooled_inside / pooled_analyzable
                if pooled_analyzable
                else np.nan,
                "analyzable_seconds": pooled_analyzable / fps,
                "inside_seconds": pooled_inside / fps,
            }
        )

    slot_summary = pd.DataFrame(slot_rows)
    slot_frame_table = (
        pd.concat(slot_frame_rows, ignore_index=True) if slot_frame_rows else None
    )
    return AnalysisResult(
        frame_summary=frame_summary,
        slot_summary=slot_summary,
        slot_frame_table=slot_frame_table,
        warnings=tuple(dict.fromkeys(warnings)),
    )
