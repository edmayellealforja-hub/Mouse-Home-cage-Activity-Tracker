"""Command-line interface for the home-cage activity tracker."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import __version__
from .analysis import analyze_occupancy
from .dlc import load_dlc
from .reporting import plot_summary, save_roi_overlay, sha256_file, write_json
from .rois import load_rois

OUTPUT_FILENAMES = {
    "frame_summary.csv",
    "roi_slot_summary.csv",
    "run_metadata.json",
    "roi_occupancy_pct.png",
    "roi_tracking_coverage_pct.png",
    "roi_overlay.png",
    "slot_frame_roi.csv.gz",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate quality-aware visible ROI occupancy from multi-animal "
            "DeepLabCut CSV or HDF5 output."
        )
    )
    parser.add_argument("--dlc", required=True, help="DLC .csv or .h5 file")
    parser.add_argument("--rois", required=True, help="ROI JSON file")
    parser.add_argument("--output-dir", required=True, help="Directory for analysis outputs")
    parser.add_argument("--fps", required=True, type=float, help="DLC/video frame rate")
    parser.add_argument(
        "--likelihood-threshold",
        type=float,
        default=0.6,
        help="Minimum DLC likelihood for a point to be analyzable (default: 0.6)",
    )
    parser.add_argument(
        "--expected-mice",
        type=int,
        default=None,
        help="Fail if the number of raw DLC slots differs from this value",
    )
    parser.add_argument(
        "--excluded-frames",
        default=None,
        help="Optional CSV containing zero-based frame_index values excluded by QC",
    )
    parser.add_argument(
        "--reference-image",
        default=None,
        help="Optional reference image used to generate roi_overlay.png",
    )
    parser.add_argument(
        "--save-slot-frame-table",
        action="store_true",
        help="Also save the large per-frame, per-slot, per-ROI table",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of tracker output files already in output-dir",
    )
    parser.add_argument("--version", action="version", version=__version__)
    return parser


def _read_excluded_frames(path: str | None) -> set[int]:
    if path is None:
        return set()
    table = pd.read_csv(path)
    if "frame_index" not in table.columns:
        raise ValueError("Excluded-frames CSV must contain a frame_index column.")
    values = pd.to_numeric(table["frame_index"], errors="raise")
    if not (values % 1 == 0).all():
        raise ValueError("Excluded frame indices must be whole numbers.")
    return set(values.astype(int).tolist())


def _prepare_output_dir(path: Path, overwrite: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    conflicts = sorted(item.name for item in path.iterdir() if item.name in OUTPUT_FILENAMES)
    if conflicts and not overwrite:
        raise FileExistsError(
            "Output directory already contains tracker results: "
            f"{conflicts}. Use --overwrite or choose another directory."
        )


def run(args: argparse.Namespace) -> int:
    dlc_path = Path(args.dlc).expanduser().resolve()
    roi_path = Path(args.rois).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    _prepare_output_dir(output_dir, args.overwrite)

    data = load_dlc(dlc_path)
    roi_set = load_rois(roi_path)
    excluded_frames = _read_excluded_frames(args.excluded_frames)
    result = analyze_occupancy(
        data,
        roi_set,
        fps=args.fps,
        likelihood_threshold=args.likelihood_threshold,
        expected_mice=args.expected_mice,
        excluded_frames=excluded_frames,
        save_slot_frame_table=args.save_slot_frame_table,
    )

    result.frame_summary.to_csv(output_dir / "frame_summary.csv", index=False)
    result.slot_summary.to_csv(output_dir / "roi_slot_summary.csv", index=False)
    if result.slot_frame_table is not None:
        result.slot_frame_table.to_csv(
            output_dir / "slot_frame_roi.csv.gz", index=False, compression="gzip"
        )
    plot_summary(result.slot_summary, output_dir)

    reference_image = None
    if args.reference_image:
        reference_image = Path(args.reference_image).expanduser().resolve()
        save_roi_overlay(reference_image, roi_set, output_dir / "roi_overlay.png")

    metadata = {
        "tracker_version": __version__,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "measurement": "visible point occupancy from raw DLC detection slots",
        "identity_interpretation": (
            "Raw DLC slots are not treated as persistent biological mouse identities."
        ),
        "dlc": {
            "path": str(dlc_path),
            "sha256": sha256_file(dlc_path),
            "format": data.source_format,
            "csv_header_rows": data.header_rows,
            "frame_count": len(data.table),
            "raw_slots": list(data.raw_slots),
            "bodyparts": list(data.bodyparts),
        },
        "rois": {
            "path": str(roi_path),
            "sha256": sha256_file(roi_path),
            "image_size": list(roi_set.image_size) if roi_set.image_size else None,
            "definitions": [
                {
                    "name": roi.name,
                    "polygon": [list(point) for point in roi.polygon],
                    "bodyparts": list(roi.bodyparts),
                    "count_rule": roi.count_rule,
                    "description": roi.description,
                }
                for roi in roi_set.rois
            ],
        },
        "settings": {
            "fps": args.fps,
            "likelihood_threshold": args.likelihood_threshold,
            "expected_mice": args.expected_mice,
            "excluded_frame_count_requested": len(excluded_frames),
            "excluded_frame_count_applied": sum(
                0 <= frame < len(data.table) for frame in excluded_frames
            ),
            "save_slot_frame_table": args.save_slot_frame_table,
        },
        "reference_image": str(reference_image) if reference_image else None,
        "warnings": list(result.warnings),
    }
    write_json(output_dir / "run_metadata.json", metadata)

    print(f"Analyzed {len(data.table):,} frames across {len(data.raw_slots)} raw DLC slots.")
    print(f"Saved results to: {output_dir}")
    for warning in result.warnings:
        print(f"WARNING: {warning}")
    return 0


def main() -> int:
    parser = build_parser()
    try:
        return run(parser.parse_args())
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
