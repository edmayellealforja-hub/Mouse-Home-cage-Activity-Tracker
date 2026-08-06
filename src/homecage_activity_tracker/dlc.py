"""Load DeepLabCut CSV and HDF5 outputs without assuming persistent identities."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

COORDINATES = {"x", "y", "likelihood"}


@dataclass(frozen=True)
class DLCData:
    """Normalized DLC coordinates.

    ``table`` always has a three-level column index named
    ``raw_slot``, ``bodypart``, and ``coord``. Raw slots are detector
    identities. They must not be interpreted as stable biological identities.
    """

    table: pd.DataFrame
    source_path: Path
    source_format: str
    header_rows: int | None
    raw_slots: tuple[str, ...]
    bodyparts: tuple[str, ...]


def _clean_label(value: Any) -> str:
    label = str(value).strip()
    if not label or label.lower().startswith("unnamed:"):
        return ""
    return label


def _detect_csv_header_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = []
        reader = csv.reader(stream)
        for _ in range(4):
            try:
                rows.append(next(reader))
            except StopIteration:
                break

    labels = [row[0].strip().lower() if row else "" for row in rows]
    if labels[:4] == ["scorer", "individuals", "bodyparts", "coords"]:
        return 4
    if labels[:3] == ["scorer", "bodyparts", "coords"]:
        return 3
    raise ValueError(
        "Could not recognize the DLC CSV header. Expected scorer/individuals/"
        "bodyparts/coords or scorer/bodyparts/coords rows."
    )


def _normalize_columns(table: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(table.columns, pd.MultiIndex):
        raise ValueError("DLC data must use MultiIndex columns.")

    normalized: dict[tuple[str, str, str], pd.Series] = {}
    for column in table.columns:
        values = [_clean_label(value) for value in column]
        coord_positions = [
            index for index, value in enumerate(values) if value.lower() in COORDINATES
        ]
        if not coord_positions:
            continue

        coord_index = coord_positions[-1]
        coord = values[coord_index].lower()
        if coord_index < 1:
            continue
        bodypart = values[coord_index - 1]
        if not bodypart:
            continue

        if coord_index >= 3:
            raw_slot = values[coord_index - 2] or "raw_slot_1"
        else:
            raw_slot = "raw_slot_1"

        key = (raw_slot, bodypart, coord)
        if key in normalized:
            raise ValueError(f"Duplicate DLC coordinate column after normalization: {key}")
        normalized[key] = pd.to_numeric(table[column], errors="coerce")

    if not normalized:
        raise ValueError("No DLC x/y coordinate columns were found.")

    result = pd.DataFrame(normalized, index=table.index)
    result.columns = pd.MultiIndex.from_tuples(
        result.columns, names=["raw_slot", "bodypart", "coord"]
    )
    result = result.sort_index(axis=1)

    coordinate_groups = result.columns.droplevel("coord").unique()
    incomplete = [
        tuple(group)
        for group in coordinate_groups
        if (group[0], group[1], "x") not in result.columns
        or (group[0], group[1], "y") not in result.columns
    ]
    if incomplete:
        raise ValueError(f"DLC bodyparts missing x or y coordinates: {incomplete[:8]}")

    return result.reset_index(drop=True)


def load_dlc(path: str | Path) -> DLCData:
    """Load and normalize a DLC CSV or HDF5 file.

    Multi-animal CSV files with four header rows and single-animal CSV files
    with three header rows are supported. HDF5 files require PyTables.
    """

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"DLC file not found: {source}")

    suffix = source.suffix.lower()
    if suffix == ".csv":
        header_rows = _detect_csv_header_rows(source)
        table = pd.read_csv(
            source,
            header=list(range(header_rows)),
            index_col=0,
            low_memory=False,
        )
        source_format = "csv"
    elif suffix in {".h5", ".hdf", ".hdf5"}:
        try:
            table = pd.read_hdf(source)
        except ImportError as exc:
            raise RuntimeError(
                "Reading DLC HDF5 files requires PyTables. Install the project with "
                "the 'hdf5' extra or use the CSV output."
            ) from exc
        header_rows = None
        source_format = "hdf5"
    else:
        raise ValueError("DLC input must be .csv, .h5, .hdf, or .hdf5.")

    normalized = _normalize_columns(table)
    raw_slots = tuple(dict.fromkeys(normalized.columns.get_level_values("raw_slot")))
    bodyparts = tuple(sorted(set(normalized.columns.get_level_values("bodypart"))))

    return DLCData(
        table=normalized,
        source_path=source,
        source_format=source_format,
        header_rows=header_rows,
        raw_slots=raw_slots,
        bodyparts=bodyparts,
    )


def coordinate_arrays(
    data: DLCData,
    raw_slot: str,
    bodypart: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return x, y, and likelihood arrays for one raw slot and bodypart."""

    table = data.table
    x = table[(raw_slot, bodypart, "x")].to_numpy(dtype=float, copy=False)
    y = table[(raw_slot, bodypart, "y")].to_numpy(dtype=float, copy=False)
    likelihood_key = (raw_slot, bodypart, "likelihood")
    if likelihood_key in table.columns:
        likelihood = table[likelihood_key].to_numpy(dtype=float, copy=False)
    else:
        likelihood = np.ones(len(table), dtype=float)
    return x, y, likelihood
