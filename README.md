# Mouse Home-cage Activity Tracker

[![Tests](https://github.com/edmayellealforja-hub/Mouse-Home-cage-Activity-Tracker/actions/workflows/tests.yml/badge.svg)](https://github.com/edmayellealforja-hub/Mouse-Home-cage-Activity-Tracker/actions/workflows/tests.yml)

Quality-aware region-of-interest (ROI) occupancy analysis for group-housed mouse videos tracked with multi-animal DeepLabCut.

This first release is designed for fixed overhead camera views, polygon ROIs for enrichments, and multi-animal DLC CSV or HDF5 output. It supports DLC files with the four header rows used by multi-animal exports:

```text
scorer
individuals
bodyparts
coords
```

## Scientific interpretation

The tracker deliberately treats labels such as `mouse_1`, `mouse_2`, and similar values as **raw DLC detection slots**. These labels are not reported as persistent biological identities unless a separate, validated identity-matching step has already been applied.

The primary measurement is **visible point occupancy**:

- Food and water ROIs normally use the nose point.
- Wheel, tunnel, and house ROIs normally use the mouse-center point.
- A point is analyzable only when its DLC likelihood reaches the selected threshold.
- Missing detections and QC-excluded frames are removed from the denominator. They are not silently counted as zero occupancy.
- Occupancy percentage is calculated from analyzable slot-time, and tracking coverage is reported alongside every estimate.

An opaque or covered house creates a special limitation. Standard DLC cannot see a mouse after it enters. Therefore, the `house` result is only visible mouse-center occupancy unless an entry/exit method has been independently validated. It should not be described as total time inside the house.

Full experimental DLC CSV and HDF5 files are intentionally not stored in this public repository. They can be large and may contain study-specific information. A small synthetic four-header-row DLC file is included for testing.

## Installation

Python 3.10 or newer is required.

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[hdf5]"
```

On Linux or macOS:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[hdf5]"
```

The `hdf5` extra installs PyTables for DLC `.h5` files. CSV-only use does not require it.

## Legacy script

The root-level `trial11.py` script is retained for its interactive ROI selection, reference-frame extraction, and overlay-video workflow. Its raw-slot movement and covered-house state estimates are exploratory. Use the packaged `homecage-track` command for the quality-aware ROI occupancy summaries documented below.

## Quick test

From the repository root:

```bash
homecage-track \
  --dlc examples/synthetic_dlc.csv \
  --rois examples/synthetic_rois.json \
  --fps 30 \
  --expected-mice 2 \
  --excluded-frames examples/excluded_frames.csv \
  --output-dir outputs/synthetic
```

For an experimental recording, set `--expected-mice` to the actual number of mice and use paths to your local data:

```powershell
homecage-track `
  --dlc "C:\path\to\DLC_output.csv" `
  --rois "C:\path\to\objects.json" `
  --reference-image "C:\path\to\reference.png" `
  --fps 30 `
  --expected-mice 4 `
  --likelihood-threshold 0.60 `
  --output-dir "C:\path\to\results"
```

## ROI file format

The recommended format records image size, polygon geometry, bodypart selection, and the rule used to combine points:

```json
{
  "image_size": [1280, 720],
  "rois": {
    "food": {
      "polygon": [[100, 100], [250, 100], [250, 220], [100, 220]],
      "bodyparts": ["nose"],
      "count_rule": "any",
      "description": "Food interaction area"
    },
    "tunnel": {
      "polygon": [[400, 300], [650, 300], [650, 500], [400, 500]],
      "bodyparts": ["mouse_center"],
      "count_rule": "any"
    }
  }
}
```

Legacy JSON files that map ROI names directly to rectangles or polygons are also accepted. Name-based defaults are applied: nose for food and water, mouse center for other known enrichments. Explicit extended definitions are preferred because they preserve the analysis assumptions.

Supported `count_rule` values are:

- `any`: analyzable when at least one configured point is valid; inside when any valid point is inside.
- `majority`: requires every configured point to be valid; more than half must be inside.
- `all`: requires every configured point to be valid and inside.

## QC-excluded frames

If a separate video-quality review identifies unusable frames, provide a CSV with zero-based frame indices:

```csv
frame_index
103
104
105
```

These frames are marked not analyzable for every ROI. Their occupancy counts are stored as missing values, not zero.

## Outputs

| File | Purpose |
|---|---|
| `roi_slot_summary.csv` | ROI occupancy and coverage for every raw slot plus pooled raw slots |
| `frame_summary.csv` | Per-frame observed slot counts and visible counts inside each ROI |
| `run_metadata.json` | Input hashes, detected header structure, settings, ROI definitions, and interpretation warnings |
| `roi_occupancy_pct.png` | Visible ROI occupancy as a percentage of analyzable slot-time |
| `roi_tracking_coverage_pct.png` | Data coverage supporting each ROI estimate |
| `roi_overlay.png` | ROI geometry over the supplied reference image |
| `slot_frame_roi.csv.gz` | Optional long-form frame-by-slot table created with `--save-slot-frame-table` |

`run_metadata.json` records SHA-256 hashes so each result can be traced to the exact DLC and ROI inputs.

## What this version does not claim

- It does not identify individual mice from raw DLC slot names.
- It does not correct identity swaps.
- It does not infer total occupancy inside an opaque shelter.
- It does not convert pixels to physical distance or report per-mouse movement speed.
- It does not replace manual review of ROI placement, lighting, camera position, and DLC tracking quality.

Persistent per-mouse trajectories and displacement metrics should only be added after a coordinate-matching method has been validated for the cage occupancy and occlusion conditions.

## Tests

```bash
python -m pip install -e ".[test,hdf5]"
pytest
```

The synthetic test specifically checks the four-row multi-animal DLC header, raw slot detection, ROI-specific bodyparts, expected mouse-count validation, and missing/QC-excluded frame handling.

## Methodological references

1. Marcus AD, Achanta S, Jordt SE. Protocol for non-invasive assessment of spontaneous movements of group-housed animals using remote video monitoring. *STAR Protocols*. 2022;3:101326. [https://doi.org/10.1016/j.xpro.2022.101326](https://doi.org/10.1016/j.xpro.2022.101326)
2. Benedict J, Cudmore RH. PiE: an open-source pipeline for home cage behavioral analysis. *Frontiers in Neuroscience*. 2023;17:1222644. [https://doi.org/10.3389/fnins.2023.1222644](https://doi.org/10.3389/fnins.2023.1222644)
3. Raspberry Pi Ltd. *The Picamera2 Library*, release 2.0, build version c3fc3a5148eb, 2025. [Official Picamera2 repository](https://github.com/raspberrypi/picamera2)

These references motivate remote, low-disturbance home-cage recording, explicit quality control, reproducible open-source workflows, and stable camera configuration. They do not validate the specific ROI definitions or identity assumptions for a new experiment. Each cage and recording period still requires its own visual QC.
