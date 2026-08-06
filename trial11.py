import argparse
import csv
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

Rect = Tuple[int, int, int, int]
Point = Tuple[int, int]
Polygon = List[Point]

OBJECT_NAMES = ["wheel", "food", "water_1", "water_2", "house", "house_entrance", "tunnel"]

# These objects are treated as fixed ROIs. This is the recommended mode for your HCS setup.
DEFAULT_STATIC_OBJECTS = ["house", "house_entrance", "wheel", "food", "water", "tunnel"]


@dataclass
class Config:
    input_video: str
    output_dir: str
    objects_path: str
    dlc_path: Optional[str] = None
    max_frames: Optional[int] = None
    save_overlay_video: bool = True
    save_reference_image: bool = True
    reference_image_name: str = "reference_best_frame.png"
    reference_overlay_name: str = "reference_best_frame_with_rois.png"
    reference_scan_frames: int = 300
    reference_skip_frames: int = 5
    reference_sample_every: int = 5
    n_mice: int = 1
    likelihood_threshold: float = 0.60
    bodyparts: Optional[List[str]] = None
    count_rule: str = "any"  # any, majority, all

    # QC for object movement / flip / fall using an automatically selected clear reference frame.
    use_object_qc: bool = True
    qc_objects: Optional[List[str]] = None
    qc_absdiff_threshold: int = 35
    qc_bad_fraction: float = 0.28
    exclude_qc_review_frames: bool = True

    # House entrance/exit estimation.
    # The house itself can hide mice, so direct DLC-in-house counts underestimate occupancy.
    # This state machine uses body/center points crossing the house_entrance ROI and holds
    # the mouse as inside for a short period when it disappears after entry.
    use_house_entrance_logic: bool = True
    house_name: str = "house"
    house_entrance_name: str = "house_entrance"
    house_bodyparts: Optional[List[str]] = None
    house_missing_hold_frames: int = 90


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def polygon_to_rect(poly: Polygon) -> Rect:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


def rect_to_polygon(rect: Rect) -> Polygon:
    x1, y1, x2, y2 = rect
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def clamp_polygon(poly: Polygon, width: int, height: int) -> Polygon:
    out = []
    for x, y in poly:
        out.append((max(0, min(width - 1, int(x))), max(0, min(height - 1, int(y)))))
    return out


def clamp_rect(rect: Rect, width: int, height: int) -> Rect:
    x1, y1, x2, y2 = rect
    x1 = max(0, min(width - 1, int(x1)))
    y1 = max(0, min(height - 1, int(y1)))
    x2 = max(0, min(width - 1, int(x2)))
    y2 = max(0, min(height - 1, int(y2)))
    if x2 <= x1:
        x2 = min(width - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(height - 1, y1 + 1)
    return x1, y1, x2, y2


def load_objects(path: str) -> Tuple[Dict[str, Polygon], Dict[str, Rect]]:
    """
    Supports:
    {
        "wheel": [x1, y1, x2, y2],
        "tunnel": [[x, y], [x, y], [x, y]]
    }
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    polygons: Dict[str, Polygon] = {}
    rects: Dict[str, Rect] = {}

    for name, value in data.items():
        if isinstance(value, list) and len(value) == 4 and all(isinstance(v, (int, float)) for v in value):
            x1, y1, x2, y2 = map(int, value)
            if x2 <= x1 or y2 <= y1:
                raise ValueError(f"Invalid rectangle for '{name}': {value}")
            rect = (x1, y1, x2, y2)
            poly = rect_to_polygon(rect)
        elif isinstance(value, list) and len(value) >= 3:
            poly = []
            for point in value:
                if not isinstance(point, list) or len(point) != 2:
                    raise ValueError(f"Invalid polygon point for '{name}': {point}")
                poly.append((int(point[0]), int(point[1])))
            rect = polygon_to_rect(poly)
        else:
            raise ValueError(f"Object '{name}' must be [x1,y1,x2,y2] or [[x,y],...]")

        polygons[name] = poly
        rects[name] = rect

    return polygons, rects




def frame_sharpness_score(frame: np.ndarray) -> float:
    """Return a blur/sharpness score. Higher means sharper.

    The variance of the Laplacian is a common focus metric. It is useful here
    because blurred frames have weaker edges and therefore lower variance.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def select_best_reference_frame(
    input_video: str,
    scan_frames: int = 300,
    skip_frames: int = 5,
    sample_every: int = 5,
) -> Tuple[np.ndarray, int, float]:
    """Select the clearest early frame instead of assuming frame 1 is usable.

    The script skips the first few frames because HCS videos can start with
    encoder/camera blur. Then it samples frames from the early part of the video
    and chooses the frame with the highest sharpness score.
    """
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_video}")

    scan_frames = max(1, int(scan_frames))
    skip_frames = max(0, int(skip_frames))
    sample_every = max(1, int(sample_every))

    best_frame: Optional[np.ndarray] = None
    best_index = -1
    best_score = -1.0

    frame_index0 = -1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_index0 += 1

        if frame_index0 < skip_frames:
            continue
        if frame_index0 >= skip_frames + scan_frames:
            break
        if (frame_index0 - skip_frames) % sample_every != 0:
            continue

        score = frame_sharpness_score(frame)
        if score > best_score:
            best_score = score
            best_index = frame_index0
            best_frame = frame.copy()

    cap.release()

    if best_frame is None:
        raise RuntimeError("Could not select a reference frame from the video.")

    # Return 1-based frame number for user-facing logs.
    return best_frame, best_index + 1, best_score


def save_reference_images(
    reference_frame: np.ndarray,
    output_dir: str,
    object_polygons: Optional[Dict[str, Polygon]] = None,
    reference_image_name: str = "reference_best_frame.png",
    reference_overlay_name: str = "reference_best_frame_with_rois.png",
) -> Tuple[str, Optional[str]]:
    """Save the selected reference image and an optional ROI overlay copy."""
    ensure_dir(output_dir)
    reference_path = os.path.join(output_dir, reference_image_name)
    ok = cv2.imwrite(reference_path, reference_frame)
    if not ok:
        raise RuntimeError(f"Could not write reference image to: {reference_path}")

    overlay_path: Optional[str] = None
    if object_polygons:
        overlay = reference_frame.copy()
        for name, poly in object_polygons.items():
            if len(poly) < 3:
                continue
            pts = np.array(poly, dtype=np.int32)
            cv2.polylines(overlay, [pts], isClosed=True, color=(0, 255, 255), thickness=2)
            x, y = poly[0]
            cv2.putText(
                overlay,
                name,
                (int(x), max(20, int(y) - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        overlay_path = os.path.join(output_dir, reference_overlay_name)
        ok = cv2.imwrite(overlay_path, overlay)
        if not ok:
            raise RuntimeError(f"Could not write ROI reference image to: {overlay_path}")

    return reference_path, overlay_path


def extract_reference_frame(
    input_video: str,
    output_image: str,
    scan_frames: int = 300,
    skip_frames: int = 5,
    sample_every: int = 5,
) -> None:
    """Extract and save the clearest early reference frame, then exit."""
    frame, frame_number, sharpness = select_best_reference_frame(
        input_video=input_video,
        scan_frames=scan_frames,
        skip_frames=skip_frames,
        sample_every=sample_every,
    )
    out_dir = os.path.dirname(os.path.abspath(output_image))
    if out_dir:
        ensure_dir(out_dir)
    ok = cv2.imwrite(output_image, frame)
    if not ok:
        raise RuntimeError(f"Could not write reference frame to: {output_image}")
    print(f"Saved best reference frame to: {output_image}")
    print(f"Selected source frame: {frame_number} | sharpness score: {sharpness:.2f}")

def save_example_objects(path: str) -> None:
    example = {
        "wheel": [[720, 320], [1280, 320], [1280, 720], [720, 720]],
        "food": [[20, 20], [240, 20], [240, 180], [20, 180]],
        "water_1": [[0, 480], [160, 480], [160, 720], [0, 720]],
        "water_2": [[1080, 480], [1280, 480], [1280, 720], [1080, 720]],
        "house": [[430, 20], [780, 20], [780, 260], [430, 260]],
        "house_entrance": [[520, 250], [690, 250], [690, 330], [520, 330]],
        "tunnel": [[0, 220], [470, 220], [470, 560], [0, 560]],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(example, f, indent=2)
    print(f"Example polygon object file written to: {path}")


class PolygonSelector:
    def __init__(self, frame: np.ndarray, object_name: str):
        self.frame = frame.copy()
        self.display = frame.copy()
        self.object_name = object_name
        self.points: Polygon = []
        self.cancelled = False

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((int(x), int(y)))
            self.redraw()
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.points:
                self.points.pop()
                self.redraw()

    def redraw(self):
        self.display = self.frame.copy()
        cv2.putText(
            self.display,
            f"{self.object_name.upper()}: left click points, right click undo, ENTER/SPACE finish, C clear, ESC skip",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        for i, point in enumerate(self.points):
            cv2.circle(self.display, point, 4, (0, 0, 255), -1)
            cv2.putText(self.display, str(i + 1), (point[0] + 5, point[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        if len(self.points) >= 2:
            for i in range(len(self.points) - 1):
                cv2.line(self.display, self.points[i], self.points[i + 1], (0, 255, 255), 2)
        if len(self.points) >= 3:
            cv2.line(self.display, self.points[-1], self.points[0], (0, 255, 255), 2)
            pts = np.array(self.points, dtype=np.int32)
            fill = self.display.copy()
            cv2.fillPoly(fill, [pts], (0, 255, 255))
            self.display = cv2.addWeighted(fill, 0.25, self.display, 0.75, 0)

    def select(self) -> Optional[Polygon]:
        window_name = "Select polygon objects"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        self.redraw()
        while True:
            cv2.imshow(window_name, self.display)
            key = cv2.waitKey(20) & 0xFF
            if key in [13, 10, 32]:
                if len(self.points) >= 3:
                    break
                print(f"Need at least 3 points for {self.object_name}.")
            elif key == ord("c"):
                self.points = []
                self.redraw()
            elif key == 27:
                self.cancelled = True
                break
        cv2.destroyWindow(window_name)
        if self.cancelled or len(self.points) < 3:
            return None
        return self.points


def select_objects_from_reference_frame(
    input_video: str,
    output_json: str,
    scan_frames: int = 300,
    skip_frames: int = 5,
    sample_every: int = 5,
) -> None:
    frame, frame_number, sharpness = select_best_reference_frame(
        input_video=input_video,
        scan_frames=scan_frames,
        skip_frames=skip_frames,
        sample_every=sample_every,
    )
    print(f"Using source frame {frame_number} for ROI selection | sharpness score: {sharpness:.2f}")

    reference_dir = os.path.dirname(os.path.abspath(output_json)) or "."
    reference_path, _ = save_reference_images(
        frame,
        reference_dir,
        object_polygons=None,
        reference_image_name="reference_best_frame.png",
        reference_overlay_name="reference_best_frame_with_rois.png",
    )
    print(f"Saved selected reference image to: {reference_path}")

    objects: Dict[str, Polygon] = {}
    preview = frame.copy()
    for name in OBJECT_NAMES:
        selector = PolygonSelector(preview, name)
        poly = selector.select()
        if poly is None:
            print(f"Skipped {name}")
            continue
        objects[name] = poly
        pts = np.array(poly, dtype=np.int32)
        cv2.polylines(preview, [pts], isClosed=True, color=(0, 255, 255), thickness=2)
        cv2.putText(preview, name, poly[0], cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump({name: [[int(x), int(y)] for x, y in poly] for name, poly in objects.items()}, f, indent=2)

    _, overlay_path = save_reference_images(
        frame,
        reference_dir,
        object_polygons=objects,
        reference_image_name="reference_best_frame.png",
        reference_overlay_name="reference_best_frame_with_rois.png",
    )
    print(f"Saved selected polygon object boundaries to: {output_json}")
    if overlay_path:
        print(f"Saved ROI reference image to: {overlay_path}")


def point_in_polygon(point: Tuple[float, float], poly: Polygon) -> bool:
    if len(poly) < 3:
        return False
    contour = np.array(poly, dtype=np.int32)
    result = cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), False)
    return result >= 0


def pairwise_distances(points: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return np.array([], dtype=float)
    diffs = points[:, None, :] - points[None, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=-1))
    iu = np.triu_indices(len(points), k=1)
    return dists[iu]


def min_distance_to_previous(point: np.ndarray, previous_points: np.ndarray) -> float:
    if previous_points.size == 0:
        return float("nan")
    d = np.sqrt(np.sum((previous_points - point) ** 2, axis=1))
    return float(np.min(d))


def make_polygon_mask(frame_shape: Tuple[int, int], poly: Polygon) -> np.ndarray:
    h, w = frame_shape
    mask = np.zeros((h, w), dtype=np.uint8)
    if len(poly) >= 3:
        pts = np.array(poly, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
    return mask


def object_in_list(name: str, items: Optional[List[str]]) -> bool:
    if not items:
        return False
    clean = name.lower().strip()
    for item in items:
        item = item.lower().strip()
        if not item:
            continue
        if clean == item or clean.startswith(item + "_"):
            return True
    return False


def build_qc_reference(first_frame: np.ndarray, object_polygons: Dict[str, Polygon], qc_objects: Optional[List[str]]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    reference_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    h, w = reference_gray.shape
    qc_masks: Dict[str, np.ndarray] = {}
    for name, poly in object_polygons.items():
        if object_in_list(name, qc_objects):
            qc_masks[name] = make_polygon_mask((h, w), poly)
    return reference_gray, qc_masks


def evaluate_object_qc(frame: np.ndarray, reference_gray: np.ndarray, qc_masks: Dict[str, np.ndarray], raw_counts: Dict[str, int], absdiff_threshold: int, bad_fraction: float) -> Dict[str, Dict[str, object]]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray, reference_gray)
    out: Dict[str, Dict[str, object]] = {}
    for name, mask in qc_masks.items():
        inside = mask > 0
        n_pixels = int(np.count_nonzero(inside))
        if n_pixels == 0:
            out[name] = {"state": "no_qc_mask", "review_flag": 0, "changed_fraction": 0.0, "count_valid": 1}
            continue
        changed_fraction = float(np.mean(diff[inside] >= absdiff_threshold))
        raw_count = int(raw_counts.get(f"raw_count_in_{name}", 0))
        if changed_fraction <= bad_fraction:
            state, review_flag, count_valid = "ok", 0, 1
        elif raw_count > 0:
            state, review_flag, count_valid = "mouse_occlusion_or_changed", 1, 1
        else:
            state, review_flag, count_valid = "review_possible_flip_or_fall", 1, 0
        out[name] = {"state": state, "review_flag": int(review_flag), "changed_fraction": float(changed_fraction), "count_valid": int(count_valid)}
    return out


def flatten_columns(columns) -> List[str]:
    out = []
    for col in columns:
        if isinstance(col, tuple):
            parts = [str(x) for x in col if str(x) not in ["", "nan", "None"]]
            out.append("|".join(parts))
        else:
            out.append(str(col))
    return out


def read_dlc_table(path: str) -> pd.DataFrame:
    """Read DLC CSV or H5 and return a DataFrame with flattened column names."""
    lower = path.lower()
    if lower.endswith(".h5") or lower.endswith(".hdf5"):
        df = pd.read_hdf(path)
        df.columns = flatten_columns(df.columns)
        return df.reset_index(drop=True)

    # Multi-animal DLC CSV files have 4 header rows; single-animal files have 3.
    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream)
            labels = []
            for _ in range(4):
                try:
                    row = next(reader)
                except StopIteration:
                    break
                labels.append(row[0].strip().lower() if row else "")
        if labels[:4] == ["scorer", "individuals", "bodyparts", "coords"]:
            header_rows = 4
        elif labels[:3] == ["scorer", "bodyparts", "coords"]:
            header_rows = 3
        else:
            raise ValueError("Unrecognized DLC CSV header")

        df = pd.read_csv(path, header=list(range(header_rows)), index_col=0)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = flatten_columns(df.columns)
            return df.reset_index(drop=True)
    except Exception:
        pass

    # Fallback for already flattened CSV.
    df = pd.read_csv(path)
    df.columns = [str(c) for c in df.columns]
    return df.reset_index(drop=True)


def parse_dlc_columns(df: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    """
    Returns mapping:
    point_key -> {x: col, y: col, likelihood: col, individual: str, bodypart: str}

    Works with common DLC exports:
    scorer|bodypart|x
    scorer|individual|bodypart|x
    flat columns containing bodypart_x, bodypart_y, bodypart_likelihood
    """
    candidates: Dict[Tuple[str, str], Dict[str, str]] = {}

    for col in df.columns:
        name = str(col)
        parts = name.split("|")
        coord = parts[-1].strip().lower() if parts else ""
        if coord not in ["x", "y", "likelihood"]:
            # Flat fallback: nose_x, nose_y, nose_likelihood
            m = re.match(r"(.+)[_\-\.](x|y|likelihood)$", name, flags=re.IGNORECASE)
            if not m:
                continue
            bodypart = m.group(1).strip()
            coord = m.group(2).lower()
            individual = "mouse_1"
        else:
            if len(parts) >= 4:
                individual = parts[-3].strip()
                bodypart = parts[-2].strip()
            elif len(parts) >= 3:
                individual = "mouse_1"
                bodypart = parts[-2].strip()
            else:
                continue

        key = (individual, bodypart)
        candidates.setdefault(key, {"individual": individual, "bodypart": bodypart})
        candidates[key][coord] = col

    points: Dict[str, Dict[str, str]] = {}
    for (individual, bodypart), cols in candidates.items():
        if "x" in cols and "y" in cols:
            key = f"{individual}:{bodypart}"
            points[key] = cols
    if not points:
        raise RuntimeError("Could not find DLC x/y columns. Use a DLC CSV/H5 with x, y, likelihood columns.")
    return points


def available_bodyparts(point_columns: Dict[str, Dict[str, str]]) -> List[str]:
    return sorted(set(str(v["bodypart"]) for v in point_columns.values()))


def select_dlc_points_for_frame(df: pd.DataFrame, frame_index0: int, point_columns: Dict[str, Dict[str, str]], requested_bodyparts: Optional[List[str]], likelihood_threshold: float) -> Tuple[List[Dict[str, object]], Dict[str, List[Tuple[float, float]]]]:
    if frame_index0 < 0 or frame_index0 >= len(df):
        return [], {}

    row = df.iloc[frame_index0]
    selected: List[Dict[str, object]] = []
    by_individual: Dict[str, List[Tuple[float, float]]] = {}
    wanted = set([b.lower().strip() for b in requested_bodyparts]) if requested_bodyparts else None

    for key, cols in point_columns.items():
        bodypart = str(cols["bodypart"])
        individual = str(cols["individual"])
        if wanted is not None and not point_bodypart_allowed(bodypart, list(wanted)):
            continue
        try:
            x = float(row[cols["x"]])
            y = float(row[cols["y"]])
        except Exception:
            continue
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        l_col = cols.get("likelihood")
        likelihood = 1.0
        if l_col is not None:
            try:
                likelihood = float(row[l_col])
            except Exception:
                likelihood = 0.0
        if likelihood < likelihood_threshold:
            continue
        rec = {"individual": individual, "bodypart": bodypart, "x": x, "y": y, "likelihood": likelihood}
        selected.append(rec)
        by_individual.setdefault(individual, []).append((x, y))

    return selected, by_individual


def summarize_mouse_centers(by_individual: Dict[str, List[Tuple[float, float]]]) -> List[Tuple[float, float]]:
    centers = []
    for pts in by_individual.values():
        if not pts:
            continue
        arr = np.array(pts, dtype=float)
        centers.append((float(np.mean(arr[:, 0])), float(np.mean(arr[:, 1]))))
    return centers


def count_dlc_occupancy(selected_points: List[Dict[str, object]], object_polygons: Dict[str, Polygon], count_rule: str) -> Tuple[Dict[str, int], Dict[str, Dict[str, int]], Dict[str, Dict[str, int]]]:
    """
    Counts unique individuals per object.

    count_rule:
    - any: one valid bodypart inside ROI counts that mouse inside.
    - majority: more than half of selected bodyparts for that mouse are inside ROI.
    - all: all selected bodyparts for that mouse must be inside ROI.
    """
    individuals = sorted(set(str(p["individual"]) for p in selected_points))
    per_mouse_total: Dict[str, int] = {ind: 0 for ind in individuals}
    per_object_inside: Dict[str, Dict[str, int]] = {name: {ind: 0 for ind in individuals} for name in object_polygons}
    per_object_total: Dict[str, Dict[str, int]] = {name: {ind: 0 for ind in individuals} for name in object_polygons}

    for p in selected_points:
        ind = str(p["individual"])
        per_mouse_total[ind] += 1
        pt = (float(p["x"]), float(p["y"]))
        for name, poly in object_polygons.items():
            if name.lower().startswith(("food", "water")):
                wanted = ["nose"]
            else:
                wanted = ["mouse_center", "center", "bodycenter", "body_center", "mid_back"]
            if not point_bodypart_allowed(str(p.get("bodypart", "")), wanted):
                continue
            per_object_total[name][ind] += 1
            if point_in_polygon(pt, poly):
                per_object_inside[name][ind] += 1

    counts: Dict[str, int] = {}
    for name in object_polygons:
        n_inside = 0
        for ind in individuals:
            inside = per_object_inside[name][ind]
            total = per_object_total[name][ind]
            if total <= 0:
                continue
            if count_rule == "all":
                is_in = inside == total
            elif count_rule == "majority":
                is_in = inside > (total / 2.0)
            else:
                is_in = inside >= 1
            if is_in:
                n_inside += 1
        counts[f"raw_count_in_{name}"] = int(n_inside)
    return counts, per_object_inside, {"total_valid_bodyparts": per_mouse_total}




def normalize_name(name: str) -> str:
    """Normalize DLC bodypart/object names for tolerant matching."""
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def point_bodypart_allowed(bodypart: str, wanted_bodyparts: Optional[List[str]]) -> bool:
    """
    Bodypart match with common DLC naming variants.

    Examples this should match:
    - wanted center  -> DLC bodypart mouse_center
    - wanted tailbase -> DLC bodypart tail_base
    - wanted body_center -> DLC bodypart mouse_center, if center is present in the name
    """
    if not wanted_bodyparts:
        return True

    bp_norm = normalize_name(bodypart)
    wanted_norm = {normalize_name(bp) for bp in wanted_bodyparts}

    if bp_norm in wanted_norm:
        return True

    for wn in wanted_norm:
        if not wn:
            continue
        # Tolerant matching for center/body-center names such as mouse_center.
        if wn in {"center", "bodycenter", "body", "mousecenter"}:
            if "center" in bp_norm or bp_norm in {"midback", "neck"}:
                return True
        # Tolerant matching for tail base variants.
        if wn in {"tailbase", "tail"}:
            if bp_norm in {"tailbase", "tail1"}:
                return True
        # General contains match for harmless variants like left_ear_tip vs lefteartip.
        if wn == bp_norm or wn in bp_norm or bp_norm in wn:
            return True

    return False


def group_points_by_individual(selected_points: List[Dict[str, object]]) -> Dict[str, List[Dict[str, object]]]:
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for p in selected_points:
        grouped.setdefault(str(p["individual"]), []).append(p)
    return grouped


def estimate_house_entrance_states(
    selected_points: List[Dict[str, object]],
    house_poly: Polygon,
    entrance_poly: Polygon,
    previous_state: Dict[str, bool],
    missing_frames: Dict[str, int],
    house_bodyparts: Optional[List[str]],
    hold_frames: int,
) -> Tuple[int, Dict[str, bool], Dict[str, int], Dict[str, Dict[str, object]]]:
    """
    Estimate house occupancy using entrance/exit body-point logic.

    Rule:
    - If a body/center point is inside the house or house_entrance, mark that mouse as inside.
    - If the mouse then disappears, keep it inside for up to hold_frames.
    - If the mouse is visible outside both the house and entrance, mark it outside.

    This is intended for covered houses where DLC cannot see a mouse once it is fully inside.
    """
    grouped = group_points_by_individual(selected_points)
    individuals = sorted(set(previous_state.keys()) | set(grouped.keys()))

    new_state: Dict[str, bool] = {}
    new_missing: Dict[str, int] = {}
    debug: Dict[str, Dict[str, object]] = {}

    for ind in individuals:
        pts_all = grouped.get(ind, [])
        pts_body = [p for p in pts_all if point_bodypart_allowed(str(p.get("bodypart", "")), house_bodyparts)]

        # If the requested body/center point is not available, use all valid points as a fallback.
        # This prevents the state machine from failing completely when DLC uses a different bodypart name.
        pts_for_state = pts_body if pts_body else pts_all

        visible = len(pts_for_state) > 0
        at_entrance = False
        inside_house_visible = False

        for p in pts_for_state:
            pt = (float(p["x"]), float(p["y"]))
            if point_in_polygon(pt, entrance_poly):
                at_entrance = True
            if point_in_polygon(pt, house_poly):
                inside_house_visible = True

        was_inside = bool(previous_state.get(ind, False))
        prev_missing = int(missing_frames.get(ind, 0))

        if at_entrance or inside_house_visible:
            inside = True
            miss = 0
            reason = "body_at_entrance_or_inside_house"
        elif not visible:
            if was_inside and prev_missing < hold_frames:
                inside = True
                miss = prev_missing + 1
                reason = "hidden_after_entry_hold"
            else:
                inside = False
                miss = prev_missing + 1
                reason = "not_visible_hold_expired"
        else:
            inside = False
            miss = 0
            reason = "visible_outside_house"

        new_state[ind] = inside
        new_missing[ind] = miss
        debug[ind] = {
            "inside": int(inside),
            "visible": int(visible),
            "at_entrance": int(at_entrance),
            "inside_house_visible": int(inside_house_visible),
            "missing_frames": int(miss),
            "reason": reason,
            "n_state_points": int(len(pts_for_state)),
            "n_body_points": int(len(pts_body)),
        }

    count = int(sum(1 for v in new_state.values() if v))
    return count, new_state, new_missing, debug

def draw_overlay(frame: np.ndarray, object_rects: Dict[str, Rect], object_polygons: Dict[str, Polygon], object_qc_info: Optional[Dict[str, Dict[str, object]]], selected_points: List[Dict[str, object]], mouse_centers: List[Tuple[float, float]], frame_idx: int, metrics_text: List[str]) -> np.ndarray:
    overlay = frame.copy()

    for name, poly in object_polygons.items():
        color = (255, 255, 0)
        status_text = "fixed"
        if object_qc_info is not None and name in object_qc_info:
            qc_state = str(object_qc_info[name].get("state", ""))
            if qc_state == "review_possible_flip_or_fall":
                color = (0, 0, 255)
                status_text = "QC REVIEW"
            elif qc_state == "mouse_occlusion_or_changed":
                color = (0, 165, 255)
                status_text = "QC OCCL/CHG"
            else:
                status_text = "fixed QC ok"

        pts = np.array(poly, dtype=np.int32)
        cv2.polylines(overlay, [pts], isClosed=True, color=color, thickness=2)
        if name in object_rects:
            x1, y1, x2, y2 = object_rects[name]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 1)
        label_x, label_y = poly[0]
        cv2.putText(overlay, f"{name.upper()} ({status_text})", (label_x, max(20, label_y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

    for p in selected_points:
        x, y = int(round(float(p["x"]))), int(round(float(p["y"])))
        cv2.circle(overlay, (x, y), 3, (0, 0, 255), -1)
        label = f"{p['individual']}:{p['bodypart']}"
        cv2.putText(overlay, label, (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

    for i, (cx, cy) in enumerate(mouse_centers):
        cv2.circle(overlay, (int(cx), int(cy)), 6, (80, 160, 255), 2)
        cv2.putText(overlay, f"center_{i+1}", (int(cx) + 5, int(cy) + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 160, 255), 1, cv2.LINE_AA)

    y = 24
    cv2.putText(overlay, f"frame={frame_idx}", (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    for idx, text in enumerate(metrics_text, start=1):
        cv2.putText(overlay, text, (12, y + 22 * idx), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)
    return overlay


def build_summary(df: pd.DataFrame, object_names: List[str], fps: float) -> Dict[str, object]:
    out = {
        "n_processed_frames": int(len(df)),
        "fps_used": float(fps),
        "duration_s": float(df["time_s"].iloc[-1]) if len(df) else 0.0,
        "mean_n_dlc_points": float(df["n_dlc_points"].mean()),
        "mean_n_individuals_visible": float(df["n_individuals_visible"].mean()),
        "mean_pairwise_distance_px": float(df["mean_pairwise_distance_px"].mean(skipna=True)),
        "clustered_fraction": float(df["clustered_flag"].mean()),
        "active_fraction": float(df["active_flag"].mean()),
        "mean_activity_proxy_px": float(df["activity_proxy_px"].mean()),
    }
    for name in object_names:
        col = f"count_in_{name}"
        valid = df[col].notna()
        out[f"mean_count_in_{name}"] = float(df[col].mean())
        out[f"analyzable_fraction_{name}"] = float(valid.mean())
        out[f"fraction_analyzable_frames_any_in_{name}"] = float((df.loc[valid, col] > 0).mean()) if valid.any() else None
        out[f"fraction_analyzable_frames_two_or_more_in_{name}"] = float((df.loc[valid, col] >= 2).mean()) if valid.any() else None
        raw_col = f"raw_count_in_{name}"
        if raw_col in df.columns:
            out[f"raw_mean_count_in_{name}"] = float(df[raw_col].mean())
        qc_col = f"qc_review_flag_{name}"
        if qc_col in df.columns:
            out[f"qc_review_fraction_{name}"] = float(df[qc_col].mean())
    return out


def build_object_time_summary(df: pd.DataFrame, object_names: List[str], fps: float, n_mice: int) -> Dict[str, object]:
    total_frames = int(len(df))
    total_duration_s = total_frames / fps if fps > 0 else 0.0
    out = {
        "n_processed_frames": total_frames,
        "fps_used": float(fps),
        "n_mice": int(n_mice),
        "total_duration_s": float(total_duration_s),
        "total_duration_min": float(total_duration_s / 60.0),
        "total_duration_hr": float(total_duration_s / 3600.0),
        "objects": {},
    }
    for name in object_names:
        col = f"count_in_{name}"
        analyzable_frames = int(df[col].notna().sum())
        analyzable_time_s = analyzable_frames / fps if fps > 0 else 0.0
        occupied_frames = int((df[col] > 0).sum())
        occupied_time_s = occupied_frames / fps if fps > 0 else 0.0
        total_mouse_frames = float(df[col].sum())
        total_mouse_time_s = total_mouse_frames / fps if fps > 0 else 0.0
        avg_time_per_mouse_s = total_mouse_time_s / n_mice if n_mice > 0 else 0.0
        out["objects"][name] = {
            "analyzable_frames": analyzable_frames,
            "analyzable_time_s": float(analyzable_time_s),
            "tracking_coverage_percent": float(100.0 * analyzable_frames / total_frames) if total_frames > 0 else 0.0,
            "occupied_frames": occupied_frames,
            "occupied_time_s": float(occupied_time_s),
            "occupied_time_min": float(occupied_time_s / 60.0),
            "occupied_time_hr": float(occupied_time_s / 3600.0),
            "occupied_percent_of_analyzable_time": float(100.0 * occupied_time_s / analyzable_time_s) if analyzable_time_s > 0 else None,
            "occupied_percent_of_recording": float(100.0 * occupied_time_s / total_duration_s) if analyzable_frames == total_frames and total_duration_s > 0 else None,
            "total_mouse_frames": float(total_mouse_frames),
            "total_mouse_time_s": float(total_mouse_time_s),
            "total_mouse_time_min": float(total_mouse_time_s / 60.0),
            "total_mouse_time_hr": float(total_mouse_time_s / 3600.0),
            "avg_time_per_mouse_s": float(avg_time_per_mouse_s),
            "avg_time_per_mouse_min": float(avg_time_per_mouse_s / 60.0),
            "avg_time_per_mouse_hr": float(avg_time_per_mouse_s / 3600.0),
        }
    return out


def save_object_time_summary_csv(summary: Dict[str, object], output_csv: str) -> None:
    rows = []
    for object_name, z in summary["objects"].items():
        rows.append({"object": object_name, **z})
    pd.DataFrame(rows).to_csv(output_csv, index=False)


def make_graphs(df: pd.DataFrame, object_names: List[str], output_dir: str) -> None:
    ensure_dir(output_dir)
    df = df.copy()
    df["sec"] = df["time_s"].astype(int)
    sec_df = df.groupby("sec", as_index=False).mean(numeric_only=True)

    plt.figure(figsize=(12, 4))
    plt.plot(sec_df["sec"], sec_df["n_individuals_visible"])
    plt.xlabel("Time (s)")
    plt.ylabel("Visible individuals")
    plt.title("DLC visible individuals over time")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "graph_dlc_visible_individuals_over_time.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(12, 4))
    plt.plot(sec_df["sec"], sec_df["activity_proxy_px"])
    plt.xlabel("Time (s)")
    plt.ylabel("Activity proxy (px)")
    plt.title("Activity over time from DLC centers")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "graph_dlc_activity_over_time.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(12, 5))
    for name in object_names:
        plt.plot(sec_df["sec"], sec_df[f"count_in_{name}"], label=name)
    plt.xlabel("Time (s)")
    plt.ylabel("Mean mouse count inside fixed polygon ROI")
    plt.title("DLC fixed-polygon ROI occupancy over time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "graph_dlc_object_occupancy_over_time.png"), dpi=200)
    plt.close()

    means = [float(df[f"count_in_{name}"].mean()) for name in object_names]
    plt.figure(figsize=(10, 5))
    plt.bar(object_names, means)
    plt.ylabel("Mean mouse count")
    plt.title("Average DLC polygon ROI occupancy")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "graph_dlc_average_object_occupancy.png"), dpi=200)
    plt.close()

    qc_cols = [f"qc_review_flag_{name}" for name in object_names if f"qc_review_flag_{name}" in df.columns]
    if qc_cols:
        plt.figure(figsize=(10, 5))
        qc_rates = [float(df[col].mean()) for col in qc_cols]
        labels = [col.replace("qc_review_flag_", "") for col in qc_cols]
        plt.bar(labels, qc_rates)
        plt.ylabel("Fraction of frames flagged for review")
        plt.title("Object QC review rate")
        plt.ylim(0, 1.05)
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "graph_object_qc_review_fraction.png"), dpi=200)
        plt.close()


def make_object_time_graph(summary: Dict[str, object], output_dir: str) -> None:
    objects = list(summary["objects"].keys())
    avg_per_mouse_hours = [summary["objects"][z]["avg_time_per_mouse_hr"] for z in objects]
    total_mouse_hours = [summary["objects"][z]["total_mouse_time_hr"] for z in objects]

    plt.figure(figsize=(10, 5))
    plt.bar(objects, total_mouse_hours)
    plt.ylabel("Mouse-hours")
    plt.title("Total DLC mouse-time per fixed polygon ROI")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "graph_dlc_total_mouse_time_hours.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(objects, avg_per_mouse_hours)
    plt.ylabel("Hours")
    plt.title("Average DLC time per mouse per fixed polygon ROI")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "graph_dlc_avg_time_per_mouse_per_object_hours.png"), dpi=200)
    plt.close()


def run(cfg: Config) -> None:
    ensure_dir(cfg.output_dir)
    initial_polygons, _ = load_objects(cfg.objects_path)
    object_names = list(initial_polygons.keys())

    cap = cv2.VideoCapture(cfg.input_video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {cfg.input_video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    reference_frame, reference_frame_number, reference_sharpness = select_best_reference_frame(
        input_video=cfg.input_video,
        scan_frames=cfg.reference_scan_frames,
        skip_frames=cfg.reference_skip_frames,
        sample_every=cfg.reference_sample_every,
    )
    print(
        f"Selected reference frame {reference_frame_number} "
        f"with sharpness score {reference_sharpness:.2f}"
    )

    current_polygons = {name: clamp_polygon(poly, width, height) for name, poly in initial_polygons.items()}
    current_rects = {name: clamp_rect(polygon_to_rect(poly), width, height) for name, poly in current_polygons.items()}

    if cfg.save_reference_image:
        reference_path, overlay_reference_path = save_reference_images(
            reference_frame=reference_frame,
            output_dir=cfg.output_dir,
            object_polygons=current_polygons,
            reference_image_name=cfg.reference_image_name,
            reference_overlay_name=cfg.reference_overlay_name,
        )
        print(f"Saved selected reference image to: {reference_path}")
        if overlay_reference_path:
            print(f"Saved ROI reference image to: {overlay_reference_path}")

    if not cfg.dlc_path:
        raise RuntimeError("You must provide --dlc_csv or --dlc_h5 for DLC mode.")
    dlc_df = read_dlc_table(cfg.dlc_path)
    point_columns = parse_dlc_columns(dlc_df)
    raw_slots = sorted(set(str(columns["individual"]) for columns in point_columns.values()))
    if len(raw_slots) != cfg.n_mice:
        raise RuntimeError(
            f"Expected {cfg.n_mice} mice, but the DLC file contains {len(raw_slots)} "
            f"raw detection slots: {raw_slots}"
        )
    print(f"Loaded DLC file: {cfg.dlc_path}")
    print(f"DLC frames: {len(dlc_df)}")
    print(f"Raw DLC detection slots: {', '.join(raw_slots)}")
    print("WARNING: raw DLC slot labels are not verified persistent mouse identities.")
    print("Available bodyparts:", ", ".join(available_bodyparts(point_columns)))
    if cfg.bodyparts:
        print("Using bodyparts:", ", ".join(cfg.bodyparts))
    else:
        print("Using all available bodyparts.")

    qc_reference_gray = None
    qc_masks: Dict[str, np.ndarray] = {}
    if cfg.use_object_qc:
        qc_reference_gray, qc_masks = build_qc_reference(reference_frame, current_polygons, cfg.qc_objects)
        if qc_masks:
            print(f"QC enabled for: {', '.join(qc_masks.keys())}")
        else:
            print("QC enabled, but none of the requested QC objects were found in objects.json.")

    cap = cv2.VideoCapture(cfg.input_video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not reopen video: {cfg.input_video}")

    overlay_writer = None
    if cfg.save_overlay_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        overlay_writer = cv2.VideoWriter(os.path.join(cfg.output_dir, "overlay_dlc_fixed_polygons.mp4"), fourcc, fps, (width, height))

    rows = []
    prev_centers_np = np.empty((0, 2), dtype=float)
    house_inside_state: Dict[str, bool] = {}
    house_missing_frames: Dict[str, int] = {}
    frame_idx = 0
    processed = 0

    house_logic_active = (
        cfg.use_house_entrance_logic
        and cfg.house_name in current_polygons
        and cfg.house_entrance_name in current_polygons
    )
    if cfg.use_house_entrance_logic and not house_logic_active:
        print(
            f"House entrance logic requested but could not find both '{cfg.house_name}' "
            f"and '{cfg.house_entrance_name}' in objects.json. Direct polygon counting will be used for house."
        )
    elif house_logic_active:
        print(
            f"House entrance logic enabled: house='{cfg.house_name}', "
            f"entrance='{cfg.house_entrance_name}', hold_frames={cfg.house_missing_hold_frames}"
        )
        print(
            "WARNING: covered-house occupancy is an exploratory entry/exit estimate and "
            "must be validated before biological interpretation."
        )

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        frame_index0 = frame_idx - 1
        if frame_index0 >= len(dlc_df):
            break
        if cfg.max_frames is not None and processed >= cfg.max_frames:
            break

        selected_points, by_individual = select_dlc_points_for_frame(dlc_df, frame_index0, point_columns, cfg.bodyparts, cfg.likelihood_threshold)
        mouse_centers = summarize_mouse_centers(by_individual)
        centers_np = np.array(mouse_centers, dtype=float) if mouse_centers else np.empty((0, 2), dtype=float)
        dists = pairwise_distances(centers_np)

        clustered_pairs = []
        if len(centers_np) >= 2:
            for i in range(len(centers_np)):
                for j in range(i + 1, len(centers_np)):
                    dij = float(np.linalg.norm(centers_np[i] - centers_np[j]))
                    if dij < 90.0:
                        clustered_pairs.append((i, j))

        raw_object_counts, _, totals_info = count_dlc_occupancy(selected_points, current_polygons, cfg.count_rule)

        # House entrance/exit state machine. This replaces the direct house polygon count
        # with an estimated inside-house count when house_entrance is available.
        house_debug: Dict[str, Dict[str, object]] = {}
        if house_logic_active:
            house_count, house_inside_state, house_missing_frames, house_debug = estimate_house_entrance_states(
                selected_points=selected_points,
                house_poly=current_polygons[cfg.house_name],
                entrance_poly=current_polygons[cfg.house_entrance_name],
                previous_state=house_inside_state,
                missing_frames=house_missing_frames,
                house_bodyparts=cfg.house_bodyparts,
                hold_frames=cfg.house_missing_hold_frames,
            )
            raw_object_counts[f"raw_count_in_{cfg.house_name}"] = int(house_count)

        object_mouse_present = {f"mouse_present_near_{name}": int(raw_object_counts[f"raw_count_in_{name}"] > 0) for name in current_polygons}

        object_qc_info: Dict[str, Dict[str, object]] = {}
        if cfg.use_object_qc and qc_reference_gray is not None and qc_masks:
            object_qc_info = evaluate_object_qc(frame, qc_reference_gray, qc_masks, raw_object_counts, cfg.qc_absdiff_threshold, cfg.qc_bad_fraction)

        object_counts = {f"count_in_{name}": 0 for name in current_polygons}
        for name in current_polygons:
            raw_col = f"raw_count_in_{name}"
            final_col = f"count_in_{name}"
            qc = object_qc_info.get(name)
            count_valid = 1 if qc is None else int(qc.get("count_valid", 1))
            if cfg.exclude_qc_review_frames and count_valid == 0:
                object_counts[final_col] = np.nan
            else:
                object_counts[final_col] = raw_object_counts[raw_col]

        movement_values = []
        for pt in centers_np:
            dmin = min_distance_to_previous(pt, prev_centers_np)
            if not np.isnan(dmin):
                movement_values.append(dmin)
        activity_proxy = float(np.mean(movement_values)) if movement_values else 0.0

        row = {
            "frame_idx": frame_idx,
            "time_s": frame_idx / fps,
            "n_dlc_points": len(selected_points),
            "n_individuals_visible": len(by_individual),
            "mean_pairwise_distance_px": float(np.mean(dists)) if len(dists) else np.nan,
            "min_pairwise_distance_px": float(np.min(dists)) if len(dists) else np.nan,
            "n_clustered_pairs": len(clustered_pairs),
            "clustered_flag": int(len(clustered_pairs) > 0),
            "activity_proxy_px": activity_proxy,
            "active_flag": int(activity_proxy >= 10.0),
            "house_entrance_logic_active": int(house_logic_active),
            "estimated_house_inside_mice": int(raw_object_counts.get(f"raw_count_in_{cfg.house_name}", 0)) if house_logic_active else int(raw_object_counts.get(f"raw_count_in_{cfg.house_name}", 0)),
        }

        for name in object_names:
            rect = current_rects[name]
            poly = current_polygons[name]
            x1, y1, x2, y2 = rect
            row[f"{name}_bbox_x1"] = x1
            row[f"{name}_bbox_y1"] = y1
            row[f"{name}_bbox_x2"] = x2
            row[f"{name}_bbox_y2"] = y2
            row[f"{name}_polygon"] = json.dumps([[int(x), int(y)] for x, y in poly])
            row[f"tracker_status_{name}"] = 2
            row[f"tracker_ok_{name}"] = 1
            row[f"is_static_{name}"] = 1

        row.update(raw_object_counts)
        row.update(object_counts)
        row.update(object_mouse_present)

        for name in object_names:
            qc = object_qc_info.get(name) if cfg.use_object_qc else None
            if qc is None:
                row[f"qc_state_{name}"] = "not_checked"
                row[f"qc_review_flag_{name}"] = 0
                row[f"qc_changed_fraction_{name}"] = 0.0
                row[f"qc_count_valid_{name}"] = 1
            else:
                row[f"qc_state_{name}"] = str(qc.get("state", "not_checked"))
                row[f"qc_review_flag_{name}"] = int(qc.get("review_flag", 0))
                row[f"qc_changed_fraction_{name}"] = float(qc.get("changed_fraction", 0.0))
                row[f"qc_count_valid_{name}"] = int(qc.get("count_valid", 1))

        # Save per-mouse valid bodypart counts.
        for ind, total in totals_info["total_valid_bodyparts"].items():
            row[f"{ind}_valid_bodyparts"] = int(total)

        # Save per-mouse house entrance/exit state diagnostics.
        for ind, info in house_debug.items():
            safe_ind = str(ind).replace(" ", "_")
            row[f"{safe_ind}_estimated_inside_house"] = int(info.get("inside", 0))
            row[f"{safe_ind}_house_at_entrance"] = int(info.get("at_entrance", 0))
            row[f"{safe_ind}_house_visible_inside"] = int(info.get("inside_house_visible", 0))
            row[f"{safe_ind}_house_missing_frames"] = int(info.get("missing_frames", 0))
            row[f"{safe_ind}_house_state_reason"] = str(info.get("reason", ""))

        rows.append(row)

        if overlay_writer is not None:
            metrics_text = [
                f"DLC points={len(selected_points)}",
                f"visible mice={len(by_individual)}",
                f"activity={activity_proxy:.1f}",
            ]
            for name in current_polygons:
                qc_state = object_qc_info.get(name, {}).get("state", "not_checked") if cfg.use_object_qc else "off"
                metrics_text.append(f"{name}={object_counts[f'count_in_{name}']} raw={raw_object_counts[f'raw_count_in_{name}']} qc={qc_state}")
            overlay = draw_overlay(frame, current_rects, current_polygons, object_qc_info, selected_points, mouse_centers, frame_idx, metrics_text)
            overlay_writer.write(overlay)

        prev_centers_np = centers_np.copy()
        processed += 1

    cap.release()
    if overlay_writer is not None:
        overlay_writer.release()

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No frames were processed.")

    df.to_csv(os.path.join(cfg.output_dir, "per_frame_dlc_polygon_metrics.csv"), index=False)
    summary = build_summary(df, object_names, fps)
    with open(os.path.join(cfg.output_dir, "summary_dlc_polygon_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    object_time_summary = build_object_time_summary(df, object_names, fps, cfg.n_mice)
    with open(os.path.join(cfg.output_dir, "dlc_polygon_time_summary.json"), "w", encoding="utf-8") as f:
        json.dump(object_time_summary, f, indent=2)
    save_object_time_summary_csv(object_time_summary, os.path.join(cfg.output_dir, "dlc_polygon_time_summary.csv"))
    make_graphs(df, object_names, cfg.output_dir)
    make_object_time_graph(object_time_summary, cfg.output_dir)

    print("Saved DLC polygon occupancy metrics, QC flags, object-time summary, overlay video, and graphs.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Home-cage DLC zone occupancy with nose water/food and house entrance-exit logic")
    parser.add_argument("--input_video", type=str, help="Path to input video")
    parser.add_argument("--output_dir", type=str, default="output_dlc_polygon_detection", help="Output directory")
    parser.add_argument("--objects", type=str, help="Path to object polygon JSON file")
    parser.add_argument("--dlc_csv", type=str, default=None, help="Path to DLC CSV file")
    parser.add_argument("--dlc_h5", type=str, default=None, help="Path to DLC H5 file")
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--n_mice", type=int, default=None, help="Actual number of mice in the cage; required for DLC analysis")
    parser.add_argument("--likelihood_threshold", type=float, default=0.60, help="Minimum DLC likelihood to use a bodypart")
    parser.add_argument("--bodyparts", type=str, default="nose,center,bodycenter,body_center,tailbase,tail_base", help="Comma-separated DLC bodyparts to use. Use 'all' for all bodyparts.")
    parser.add_argument("--count_rule", type=str, default="any", choices=["any", "majority", "all"], help="How many selected bodyparts must be inside ROI to count a mouse inside")

    parser.add_argument("--no_object_qc", action="store_true", help="Turn off object visual QC")
    parser.add_argument("--qc_objects", type=str, default="tunnel,house,house_entrance,wheel,food,water_1,water_2", help="Comma-separated object names/prefixes to check for flip/fall/large visual change")
    parser.add_argument("--qc_absdiff_threshold", type=int, default=35)
    parser.add_argument("--qc_bad_fraction", type=float, default=0.28)
    parser.add_argument("--count_during_qc_review", action="store_true", help="Keep counting occupancy even when QC flags object")

    parser.add_argument("--no_house_entrance_logic", action="store_true", help="Turn off house entrance/exit state logic")
    parser.add_argument("--house_name", type=str, default="house", help="Object name for the covered house ROI")
    parser.add_argument("--house_entrance_name", type=str, default="house_entrance", help="Object name for the house entrance ROI")
    parser.add_argument("--house_bodyparts", type=str, default="center,mouse_center,bodycenter,body_center,body,mid_back,neck", help="Body/center DLC bodyparts to use for house entrance/exit logic")
    parser.add_argument("--house_missing_hold_frames", type=int, default=90, help="Frames to keep a mouse counted as inside house after it disappears following entry")

    parser.add_argument("--no_overlay", action="store_true")
    parser.add_argument("--no_reference_image", action="store_true", help="Do not save reference images during DLC analysis")
    parser.add_argument("--reference_image_name", type=str, default="reference_best_frame.png", help="Filename for the saved selected reference image")
    parser.add_argument("--reference_overlay_name", type=str, default="reference_best_frame_with_rois.png", help="Filename for the selected reference image with ROI polygons drawn")
    parser.add_argument("--reference_scan_frames", type=int, default=300, help="How many early frames to scan for the clearest reference image")
    parser.add_argument("--reference_skip_frames", type=int, default=5, help="How many startup frames to skip before reference selection")
    parser.add_argument("--reference_sample_every", type=int, default=5, help="Sample every N frames during reference selection")
    parser.add_argument("--extract_reference_frame", type=str, default=None, help="Save the selected clearest early reference frame to this image path and exit")

    parser.add_argument("--write_example_objects", type=str, default=None, help="Write example polygon object file and exit")
    parser.add_argument("--select_objects", action="store_true", help="Manually select polygon boundaries")
    parser.add_argument("--selected_objects_output", type=str, default="objects.json", help="Where to save selected polygon boundaries")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.extract_reference_frame:
        if not args.input_video:
            raise SystemExit("You must provide --input_video with --extract_reference_frame.")
        extract_reference_frame(
            args.input_video,
            args.extract_reference_frame,
            scan_frames=args.reference_scan_frames,
            skip_frames=args.reference_skip_frames,
            sample_every=args.reference_sample_every,
        )
        return
    if args.write_example_objects:
        save_example_objects(args.write_example_objects)
        return
    if args.select_objects:
        if not args.input_video:
            raise SystemExit("You must provide --input_video when using --select_objects.")
        select_objects_from_reference_frame(args.input_video, args.selected_objects_output, scan_frames=args.reference_scan_frames, skip_frames=args.reference_skip_frames, sample_every=args.reference_sample_every)
        return
    if not args.input_video or not args.objects:
        raise SystemExit("You must provide --input_video and --objects, or use --select_objects first.")

    dlc_path = args.dlc_h5 if args.dlc_h5 else args.dlc_csv
    if not dlc_path:
        raise SystemExit("You must provide --dlc_csv or --dlc_h5 for DLC mode.")
    if args.n_mice is None or args.n_mice <= 0:
        raise SystemExit("You must provide a positive --n_mice value for DLC mode.")

    bodyparts = None
    if args.bodyparts and args.bodyparts.strip().lower() != "all":
        bodyparts = [x.strip() for x in args.bodyparts.split(",") if x.strip()]

    cfg = Config(
        input_video=args.input_video,
        output_dir=args.output_dir,
        objects_path=args.objects,
        dlc_path=dlc_path,
        max_frames=args.max_frames,
        save_overlay_video=not args.no_overlay,
        save_reference_image=not args.no_reference_image,
        reference_image_name=args.reference_image_name,
        reference_overlay_name=args.reference_overlay_name,
        reference_scan_frames=max(1, args.reference_scan_frames),
        reference_skip_frames=max(0, args.reference_skip_frames),
        reference_sample_every=max(1, args.reference_sample_every),
        n_mice=args.n_mice,
        likelihood_threshold=args.likelihood_threshold,
        bodyparts=bodyparts,
        count_rule=args.count_rule,
        use_object_qc=not args.no_object_qc,
        qc_objects=[x.strip() for x in args.qc_objects.split(",") if x.strip()] if args.qc_objects is not None else [],
        qc_absdiff_threshold=args.qc_absdiff_threshold,
        qc_bad_fraction=args.qc_bad_fraction,
        exclude_qc_review_frames=not args.count_during_qc_review,
        use_house_entrance_logic=not args.no_house_entrance_logic,
        house_name=args.house_name,
        house_entrance_name=args.house_entrance_name,
        house_bodyparts=[x.strip() for x in args.house_bodyparts.split(",") if x.strip()] if args.house_bodyparts else [],
        house_missing_hold_frames=max(0, args.house_missing_hold_frames),
    )
    run(cfg)


if __name__ == "__main__":
    main()
