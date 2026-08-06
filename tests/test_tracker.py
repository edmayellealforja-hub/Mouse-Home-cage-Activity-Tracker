from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from homecage_activity_tracker.analysis import analyze_occupancy
from homecage_activity_tracker.dlc import load_dlc
from homecage_activity_tracker.rois import load_rois, points_in_polygon

EXAMPLES = Path(__file__).parents[1] / "examples"


def test_loads_four_row_multi_animal_csv() -> None:
    data = load_dlc(EXAMPLES / "synthetic_dlc.csv")

    assert data.header_rows == 4
    assert data.raw_slots == ("mouse_1", "mouse_2")
    assert set(data.bodyparts) == {"mouse_center", "nose"}
    assert len(data.table) == 5
    assert data.table.columns.names == ["raw_slot", "bodypart", "coord"]


def test_loads_three_row_single_animal_csv(tmp_path: Path) -> None:
    source = tmp_path / "single.csv"
    source.write_text(
        "scorer,model,model,model\n"
        "bodyparts,nose,nose,nose\n"
        "coords,x,y,likelihood\n"
        "0,5,6,0.9\n",
        encoding="utf-8",
    )

    data = load_dlc(source)

    assert data.header_rows == 3
    assert data.raw_slots == ("raw_slot_1",)
    assert data.bodyparts == ("nose",)


def test_missing_and_excluded_frames_are_not_zero_occupancy() -> None:
    data = load_dlc(EXAMPLES / "synthetic_dlc.csv")
    rois = load_rois(EXAMPLES / "synthetic_rois.json")
    result = analyze_occupancy(
        data,
        rois,
        fps=1,
        likelihood_threshold=0.6,
        expected_mice=2,
        excluded_frames={2},
        save_slot_frame_table=True,
    )

    food_mouse_1 = result.slot_summary.query(
        "roi == 'food' and raw_slot == 'mouse_1'"
    ).iloc[0]
    assert food_mouse_1["analyzable_frames"] == 3
    assert food_mouse_1["inside_frames"] == 2

    food_frame_1 = result.frame_summary.loc[1]
    assert food_frame_1["observed_slots__food"] == 1
    assert food_frame_1["count_in__food"] == 0

    food_frame_2 = result.frame_summary.loc[2]
    assert food_frame_2["observed_slots__food"] == 0
    assert np.isnan(food_frame_2["count_in__food"])

    slot_frame = result.slot_frame_table
    assert slot_frame is not None
    excluded_row = slot_frame.query(
        "frame_index == 2 and roi == 'food' and raw_slot == 'mouse_1'"
    ).iloc[0]
    assert not bool(excluded_row["analyzable"])
    assert pd.isna(excluded_row["inside"])


def test_expected_mouse_count_guard() -> None:
    data = load_dlc(EXAMPLES / "synthetic_dlc.csv")
    rois = load_rois(EXAMPLES / "synthetic_rois.json")

    with pytest.raises(ValueError, match="Expected 5 mice"):
        analyze_occupancy(data, rois, fps=30, expected_mice=5)


def test_points_on_polygon_boundary_are_inside() -> None:
    polygon = ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0))
    result = points_in_polygon(
        np.array([5.0, 10.0, 11.0]),
        np.array([5.0, 5.0, 5.0]),
        polygon,
    )
    assert result.tolist() == [True, True, False]
