from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).parents[1]


def _load_legacy_module():
    spec = importlib.util.spec_from_file_location("trial11", ROOT / "trial11.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_legacy_script_reads_four_row_multi_animal_csv() -> None:
    legacy = _load_legacy_module()
    table = legacy.read_dlc_table(str(ROOT / "examples" / "synthetic_dlc.csv"))
    point_columns = legacy.parse_dlc_columns(table)

    raw_slots = sorted({columns["individual"] for columns in point_columns.values()})
    assert raw_slots == ["mouse_1", "mouse_2"]
    assert set(legacy.available_bodyparts(point_columns)) == {"mouse_center", "nose"}


def test_legacy_script_uses_roi_specific_bodyparts() -> None:
    legacy = _load_legacy_module()
    table = legacy.read_dlc_table(str(ROOT / "examples" / "synthetic_dlc.csv"))
    point_columns = legacy.parse_dlc_columns(table)
    selected, _ = legacy.select_dlc_points_for_frame(
        table,
        0,
        point_columns,
        ["nose", "mouse_center"],
        0.6,
    )
    counts, _, _ = legacy.count_dlc_occupancy(
        selected,
        {
            "food": [(0, 0), (10, 0), (10, 10), (0, 10)],
            "house": [(10, 10), (19, 10), (19, 19), (10, 19)],
        },
        "any",
    )

    assert counts == {"raw_count_in_food": 1, "raw_count_in_house": 1}
