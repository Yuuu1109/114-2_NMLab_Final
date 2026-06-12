try:
    from picamera2 import Picamera2
except ImportError:
    Picamera2 = None
    print("Warning: Picamera2 not found. Camera mode will not be available.")

import cv2
import numpy as np
import time
import json
import math
import os
import csv
from pathlib import Path


# All vision-related files are under the same vision/ folder.
VISION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = VISION_DIR.parent
CONFIG_PATH = VISION_DIR / "vision_config.json"
MAP_CONFIG_CACHE = None



def load_config(path=CONFIG_PATH):
    with open(path, "r") as f:
        return json.load(f)


def save_config(config, path=CONFIG_PATH):
    config_to_save = config.copy()
    # Runtime-only state; do not persist it to vision_config.json.
    for runtime_key in (
        "last_dominant_angle",
        "last_raw_line_count",
        "last_angle_filtered_line_count",
        "last_dense_rejected_line_count",
        "last_merged_line_count",
        "last_detected_angle_update_deg",
        "last_detected_angle_support_count",
        "last_detected_angle_support_length",
        "active_motion_state",
        "last_motion_state",
        "motion_state_changed",
        "current_straight_reference_angle_deg",
        "current_yaw_error_deg",
        "current_angle_status",
        "effective_grid_angle_deg",
        "turn_tracking_active",
        "turn_start_grid_angle_deg",
        "turn_prev_grid_angle_deg",
        "turn_image_delta_deg",
        "turn_vehicle_delta_deg",
        "turn_image_rotation_deg",
        "turn_vehicle_rotation_deg",
        "turn_progress_deg",
        "turn_completed_by_vision",
        "turn_tracking_status",
        "turn_direction_runtime",
        "motion_source",
        "selected_response",
        "phase_dx_pixel",
        "phase_dy_pixel",
        "phase_response_raw",
        "light_track_status",
        "light_track_dx_pixel",
        "light_track_dy_pixel",
        "light_track_match_distance_px",
        "light_track_prev_cx",
        "light_track_prev_cy",
        "light_track_prev_area",
        "grid_track_status",
        "grid_track_dx_pixel",
        "grid_track_dy_pixel",
        "grid_track_y_match_count",
        "grid_track_x_match_count",
        "grid_track_prev_y_positions",
        "grid_track_prev_x_positions",
        "turn_target_x_cell",
        "turn_target_y_cell",
        "turn_cell_distance_cell",
        "turn_target_source",
    ):
        config_to_save.pop(runtime_key, None)
    with open(path, "w") as f:
        json.dump(config_to_save, f, indent=2)


def setup_camera(width, height, config):
    """
    Camera setup for Raspberry Pi 4 + OV5647 / Pi Camera Board.

    Notes:
    - Uses Picamera2 directly, so no preview window is opened.
    - Keeps frame rate modest on Pi 4 to avoid CPU overload.
    - Exposure can be auto or manual from vision_config.json.
    """
    if Picamera2 is None:
        raise RuntimeError("Picamera2 is not available. Install python3-picamera2 or set use_camera=false.")

    width = int(width)
    height = int(height)

    picam2 = Picamera2()

    camera_fps = float(config.get("camera_fps", config.get("sample_hz", 3)))
    camera_fps = max(1.0, min(camera_fps, 30.0))
    frame_duration_us = int(1_000_000 / camera_fps)

    camera_config = picam2.create_video_configuration(
        main={"size": (width, height), "format": "RGB888"},
        controls={"FrameDurationLimits": (frame_duration_us, frame_duration_us)},
    )

    picam2.configure(camera_config)

    controls = {}
    if not config.get("camera_auto_exposure", True):
        controls.update({
            "AeEnable": False,
            "ExposureTime": int(config.get("camera_exposure_time", 10000)),
            "AnalogueGain": float(config.get("camera_analogue_gain", 4.0)),
        })
    else:
        controls["AeEnable"] = True

    # Optional fixed white balance. Usually leave camera_auto_wb=true.
    if not config.get("camera_auto_wb", True):
        controls["AwbEnable"] = False
        controls["ColourGains"] = tuple(config.get("camera_colour_gains", [1.5, 1.5]))

    picam2.start()
    time.sleep(float(config.get("camera_warmup_sec", 1.5)))

    if controls:
        try:
            picam2.set_controls(controls)
        except Exception as exc:
            print(f"Warning: some camera controls were not applied: {exc}")

    time.sleep(float(config.get("camera_control_settle_sec", 0.5)))

    print(f"Camera configured: {width}x{height}, RGB888, target_fps={camera_fps:.1f}")
    print(f"Camera properties: {picam2.camera_properties.get('Model', 'unknown')}")

    return picam2

def load_test_image(config):
    """
    Load a test image for offline debugging.

    In this project layout, test images are expected under:
        vision/test_data/...

    test_image_path may be:
        1. relative to vision/
        2. relative to project root
        3. absolute path
    """
    image_path = Path(config.get("test_image_path", "test_data/IMG_3488.jpeg"))

    if image_path.is_absolute():
        candidate_paths = [image_path]
    else:
        candidate_paths = [
            VISION_DIR / image_path,
            PROJECT_ROOT / image_path,
        ]

    real_path = None
    for candidate in candidate_paths:
        if candidate.exists():
            real_path = candidate
            break

    if real_path is None:
        raise FileNotFoundError(
            "Cannot find test image: "
            f"{config.get('test_image_path', 'test_data/IMG_3488.jpeg')}\n"
            "Tried:\n" + "\n".join(str(p) for p in candidate_paths)
        )

    frame_bgr = cv2.imread(str(real_path))

    if frame_bgr is None:
        raise RuntimeError(f"Failed to read image: {real_path}")

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    target_w = int(config.get("frame_width", frame_rgb.shape[1]))
    target_h = int(config.get("frame_height", frame_rgb.shape[0]))

    if config.get("resize_test_image", False):
        frame_rgb = cv2.resize(frame_rgb, (target_w, target_h))

    print(f"Loaded test image: {real_path}")
    print(f"Image shape: {frame_rgb.shape}")

    return frame_rgb

def normalize_angle_90(angle):
    """
    將角度正規化到 [-90, 90)
    """
    while angle >= 90:
        angle -= 180
    while angle < -90:
        angle += 180
    return angle


def angle_distance_180(a, b):
    """
    線的方向有 180 度週期，所以 0 度和 180 度等價。
    回傳兩個角度的最小差值。
    """
    diff = abs(normalize_angle_90(a - b))
    return diff

def angle_distance_90(a, b):
    """
    計算兩個格線座標系角度的差距。
    因為天花板格線有 90 度週期：
    0 度和 90 度代表同一個格線座標系。
    """
    d1 = angle_distance_180(a, b)
    d2 = angle_distance_180(a, normalize_angle_90(b + 90))
    return min(d1, d2)

def estimate_dominant_grid_angle(raw_lines_info):
    """
    從所有 Hough 線段中估計主要格線方向。
    回傳 dominant_angle，範圍約 [-90, 90)。
    """
    if len(raw_lines_info) == 0:
        return None

    # 用角度 histogram 找最多線段支持的方向
    # bin 範圍：-90 ~ 90
    bin_size = 2
    bins = np.arange(-90, 90 + bin_size, bin_size)
    hist = np.zeros(len(bins) - 1)

    for line in raw_lines_info:
        angle = line["angle"]
        length = line["length"]

        # 用線段長度當權重，長格線比短雜線更重要
        idx = np.searchsorted(bins, angle, side="right") - 1
        if 0 <= idx < len(hist):
            hist[idx] += length

    best_idx = int(np.argmax(hist))
    dominant_angle = (bins[best_idx] + bins[best_idx + 1]) / 2

    return normalize_angle_90(dominant_angle)

def detect_lights(frame_rgb, config):
    """
    偵測天花板燈。
    輸入：RGB frame
    輸出：
        lights: list of dict
        raw_mask: threshold 後、尚未清理的 mask
        clean_mask: morphology 後的 mask
    """
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)

    threshold_value = config["light_threshold"]

    # 原始 threshold mask
    _, raw_mask = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)

    # 清理 mask
    kernel = np.ones((5, 5), np.uint8)
    clean_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel)
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    lights = []

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area < config["light_min_area"]:
            continue

        if area > config["light_max_area"]:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        if w == 0 or h == 0:
            continue

        aspect = w / h

        if aspect < 0.3 or aspect > 3.5:
            continue

        cx = x + w / 2
        cy = y + h / 2

        lights.append({
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "cx": cx,
            "cy": cy,
            "area": area,
            "aspect": aspect
        })

    lights = sorted(lights, key=lambda item: item["area"], reverse=True)

    return lights, raw_mask, clean_mask



def apply_gamma_correction(frame_rgb, config):
    """
    Apply gamma correction to brighten dark ceiling regions without increasing exposure.
    gamma < 1.0 brightens shadows; gamma = 1.0 keeps the image unchanged.
    """
    if not config.get("use_gamma_correction", False):
        return frame_rgb

    gamma = float(config.get("gamma", 1.0))
    if abs(gamma - 1.0) < 1e-6:
        return frame_rgb

    gamma = max(0.2, min(gamma, 3.0))
    inv_gamma = 1.0 / gamma
    table = np.array([
        ((i / 255.0) ** gamma) * 255 for i in range(256)
    ]).astype("uint8")

    return cv2.LUT(frame_rgb, table)


def estimate_light_zone(lights, frame_shape):
    """
    用燈的位置做粗分區。
    目前先用畫面 x 座標粗分成 left / center / right。
    """
    h, w = frame_shape[:2]

    if len(lights) == 0:
        return "NO_LIGHT"

    main_light = lights[0]
    cx = main_light["cx"]

    if cx < w / 3:
        return "LEFT_LIGHT_ZONE"
    elif cx > 2 * w / 3:
        return "RIGHT_LIGHT_ZONE"
    else:
        return "CENTER_LIGHT_ZONE"

def normalize_angle_90(angle):
    """
    將線段角度正規化到 [-90, 90)
    因為線的方向 0 度和 180 度等價。
    """
    while angle >= 90:
        angle -= 180
    while angle < -90:
        angle += 180
    return angle


def angle_distance_180(a, b):
    """
    計算兩條線方向的最小角度差。
    例如 89 度和 -89 度其實只差 2 度。
    """
    return abs(normalize_angle_90(a - b))


def estimate_dominant_grid_angle(raw_lines_info, config, prior_angle=None):
    """
    從所有 Hough 線段中估計主要格線方向。
    加入 prior_angle 限制，避免風扇、出風口等斜線誤導 dominant angle。
    """
    if len(raw_lines_info) == 0:
        return None

    use_angle_prior = config.get("use_angle_prior", True)
    max_rotation = config.get("max_grid_rotation_deg", 30)

    if prior_angle is None:
        prior_angle = config.get("expected_grid_angle", 0)

    bin_size = 2
    bins = np.arange(-90, 90 + bin_size, bin_size)
    hist = np.zeros(len(bins) - 1)

    for line in raw_lines_info:
        angle = line["angle"]
        length = line["length"]

        # 只讓接近預期格線方向的線參與 dominant angle 投票
        # 這裡用 90 度週期，所以水平/垂直兩組格線都會被接受
        if use_angle_prior:
            if angle_distance_90(angle, prior_angle) > max_rotation:
                continue

        idx = np.searchsorted(bins, angle, side="right") - 1
        if 0 <= idx < len(hist):
            hist[idx] += length

    if np.max(hist) == 0:
        return prior_angle

    best_idx = int(np.argmax(hist))
    dominant_angle = (bins[best_idx] + bins[best_idx + 1]) / 2
    dominant_angle = normalize_angle_90(dominant_angle)

    # 避免 dominant 選到另一組垂直方向，讓它盡量靠近 prior
    if angle_distance_180(dominant_angle, prior_angle) > angle_distance_180(
        normalize_angle_90(dominant_angle + 90), prior_angle
    ):
        dominant_angle = normalize_angle_90(dominant_angle + 90)

    return dominant_angle


def _grid_line_basis(direction_angle_deg):
    """
    Return unit direction vector u and normal vector n for a grid-line family.

    A line in this family can be represented by:
        p(t) = rho * n + t * u

    rho is the perpendicular distance of the line from the image origin.
    Using rho lets us compare and merge parallel Hough segments reliably.
    """
    theta = math.radians(float(direction_angle_deg))
    ux = math.cos(theta)
    uy = math.sin(theta)
    nx = -math.sin(theta)
    ny = math.cos(theta)
    return (ux, uy), (nx, ny)


def _annotate_line_parallel_geometry(line, direction_angle_deg):
    """Add rho/t-span information for a line with respect to one grid direction."""
    (ux, uy), (nx, ny) = _grid_line_basis(direction_angle_deg)

    x1 = float(line["x1"])
    y1 = float(line["y1"])
    x2 = float(line["x2"])
    y2 = float(line["y2"])

    rho1 = x1 * nx + y1 * ny
    rho2 = x2 * nx + y2 * ny
    t1 = x1 * ux + y1 * uy
    t2 = x2 * ux + y2 * uy

    out = line.copy()
    out["rho"] = 0.5 * (rho1 + rho2)
    out["t_min"] = min(t1, t2)
    out["t_max"] = max(t1, t2)
    return out


def _classify_grid_lines(raw_lines_info, direction_a, direction_b, angle_tolerance):
    """Keep only Hough segments aligned with the two orthogonal ceiling-grid directions."""
    grid_a = []
    grid_b = []

    for raw in raw_lines_info:
        line = raw.copy()
        angle = line["angle"]

        dist_a = angle_distance_180(angle, direction_a)
        dist_b = angle_distance_180(angle, direction_b)

        if dist_a < angle_tolerance:
            line["type"] = "grid_a"
            grid_a.append(_annotate_line_parallel_geometry(line, direction_a))
        elif dist_b < angle_tolerance:
            line["type"] = "grid_b"
            grid_b.append(_annotate_line_parallel_geometry(line, direction_b))

    return grid_a, grid_b


def _line_angle_as_grid_a(line):
    """
    Convert an individual detected line angle into the grid-A angle basis.

    grid_a lines already represent the grid angle.
    grid_b lines are perpendicular to grid_a, so subtract 90 degrees.

    This keeps the measured Hough angle instead of snapping the result to the
    theoretical/prior angle.
    """
    angle = float(line.get("angle", 0.0))
    if line.get("type") == "grid_b":
        angle = normalize_angle_90(angle - 90.0)
    return normalize_angle_90(angle)


def _weighted_detected_grid_angle(lines_info, reference_angle, config):
    """
    Estimate the grid angle from actual detected line angles.

    The reference angle is only used to unwrap angles and reject outliers. The
    returned value follows the detected grid direction, so slow vehicle yaw or
    camera rotation can be reflected in grid_angle.
    """
    if not config.get("use_detected_grid_angle", True):
        return reference_angle

    if not lines_info:
        config["last_detected_angle_update_deg"] = 0.0
        config["last_detected_angle_support_count"] = 0
        config["last_detected_angle_support_length"] = 0.0
        return reference_angle

    min_count = int(config.get("detected_angle_min_line_count", 2))
    max_delta = float(config.get("detected_angle_max_delta_deg", 25.0))

    weighted_diffs = []
    support_length = 0.0

    for line in lines_info:
        detected_as_a = _line_angle_as_grid_a(line)
        diff = normalize_angle_90(detected_as_a - float(reference_angle))

        # The angle-tolerance stage already filtered most outliers. This extra
        # guard prevents one wrong family assignment from jumping the basis.
        if abs(diff) > max_delta:
            continue

        weight = max(float(line.get("length", 1.0)), 1.0)
        weighted_diffs.append((diff, weight))
        support_length += weight

    if len(weighted_diffs) < min_count:
        config["last_detected_angle_update_deg"] = 0.0
        config["last_detected_angle_support_count"] = len(weighted_diffs)
        config["last_detected_angle_support_length"] = support_length
        return reference_angle

    avg_diff = sum(diff * weight for diff, weight in weighted_diffs) / sum(weight for _, weight in weighted_diffs)
    detected_angle = normalize_angle_90(float(reference_angle) + avg_diff)

    # Optional low-pass filtering. Keep alpha=1.0 to fully use the detected
    # angle; lower it only if the debug angle jitters too much.
    alpha = float(config.get("grid_angle_smoothing_alpha", 1.0))
    alpha = max(0.0, min(1.0, alpha))
    if alpha < 1.0 and config.get("last_dominant_angle", None) is not None:
        prev = float(config["last_dominant_angle"])
        detected_angle = normalize_angle_90(prev + alpha * normalize_angle_90(detected_angle - prev))

    config["last_detected_angle_update_deg"] = avg_diff
    config["last_detected_angle_support_count"] = len(weighted_diffs)
    config["last_detected_angle_support_length"] = support_length

    return detected_angle


def _reject_dense_parallel_stripes(lines_info, config):
    """
    Reject local groups of many closely spaced parallel lines.

    This targets air-conditioner / vent grilles. A grille creates many parallel
    Hough lines with small rho gaps, while real ceiling grid lines are much more
    sparsely spaced. Very tight duplicate detections of one real grid line are
    protected by dense_stripe_min_span_px: they usually do not span enough rho.
    """
    if not config.get("reject_dense_parallel_stripes", True):
        return lines_info, []

    if len(lines_info) < int(config.get("dense_stripe_min_count", 6)):
        return lines_info, []

    max_spacing = float(config.get("dense_stripe_max_spacing_px", 45.0))
    min_count = int(config.get("dense_stripe_min_count", 6))
    min_span = float(config.get("dense_stripe_min_span_px", 80.0))
    max_median_len = float(config.get("dense_stripe_max_median_length_px", 1e9))

    ordered = sorted(lines_info, key=lambda item: item.get("rho", 0.0))
    groups = []
    current = [ordered[0]]

    for line in ordered[1:]:
        prev = current[-1]
        if abs(line.get("rho", 0.0) - prev.get("rho", 0.0)) <= max_spacing:
            current.append(line)
        else:
            groups.append(current)
            current = [line]
    groups.append(current)

    rejected_ids = set()
    for group in groups:
        if len(group) < min_count:
            continue

        rhos = [float(item.get("rho", 0.0)) for item in group]
        rho_span = max(rhos) - min(rhos)
        lengths = [float(item.get("length", 0.0)) for item in group]
        median_length = float(np.median(lengths)) if lengths else 0.0

        if rho_span >= min_span and median_length <= max_median_len:
            for item in group:
                rejected_ids.add(id(item))
                item["rejected_reason"] = "dense_parallel_stripes"

    kept = [item for item in ordered if id(item) not in rejected_ids]
    rejected = [item for item in ordered if id(item) in rejected_ids]
    return kept, rejected


def _make_merged_parallel_line(cluster, direction_angle_deg, line_type):
    """Create one representative segment from a cluster of parallel Hough segments."""
    (ux, uy), (nx, ny) = _grid_line_basis(direction_angle_deg)

    weights = np.array([max(float(item.get("length", 1.0)), 1.0) for item in cluster], dtype=np.float64)
    rhos = np.array([float(item.get("rho", 0.0)) for item in cluster], dtype=np.float64)
    rho = float(np.average(rhos, weights=weights))

    t_min = min(float(item.get("t_min", 0.0)) for item in cluster)
    t_max = max(float(item.get("t_max", 0.0)) for item in cluster)

    x1 = rho * nx + t_min * ux
    y1 = rho * ny + t_min * uy
    x2 = rho * nx + t_max * ux
    y2 = rho * ny + t_max * uy

    length = math.hypot(x2 - x1, y2 - y1)

    return {
        "x1": int(round(x1)),
        "y1": int(round(y1)),
        "x2": int(round(x2)),
        "y2": int(round(y2)),
        "angle": normalize_angle_90(direction_angle_deg),
        "length": length,
        "rho": rho,
        "t_min": t_min,
        "t_max": t_max,
        "type": line_type,
        "support_count": len(cluster),
        "support_length": float(np.sum(weights)),
    }


def _merge_parallel_lines(lines_info, direction_angle_deg, line_type, config):
    """
    Merge multiple Hough detections that belong to the same physical ceiling line.

    Hough often detects both edges of a ceiling seam, shadow edges, or fragmented
    line segments. We group by rho and draw one representative line per group.
    """
    if not config.get("line_merge_enabled", True):
        return lines_info

    if not lines_info:
        return []

    rho_threshold = float(config.get("line_merge_rho_threshold_px", 35.0))
    min_merged_length = float(config.get("merged_line_min_length_px", 80.0))

    ordered = sorted(lines_info, key=lambda item: item.get("rho", 0.0))
    clusters = []
    current = [ordered[0]]

    for line in ordered[1:]:
        prev = current[-1]
        if abs(line.get("rho", 0.0) - prev.get("rho", 0.0)) <= rho_threshold:
            current.append(line)
        else:
            clusters.append(current)
            current = [line]
    clusters.append(current)

    merged = []
    for cluster in clusters:
        rep = _make_merged_parallel_line(cluster, direction_angle_deg, line_type)
        if rep["length"] >= min_merged_length:
            merged.append(rep)

    return merged


def _postprocess_grid_lines(raw_lines_info, dominant_angle, config):
    """
    Classify, reject dense grille stripes, estimate the actual detected angle,
    and merge lines.

    Important behavior:
      - expected_grid_angle / last_dominant_angle is only a prior for deciding
        which Hough lines are plausible.
      - grid_angle is then refined from the measured Hough angles of the kept
        lines, so it can change when the vehicle/camera becomes slightly tilted.
    """
    angle_tolerance = float(config.get("line_angle_tolerance", 12))

    direction_a = dominant_angle
    direction_b = normalize_angle_90(dominant_angle + 90)

    # First pass: use the prior-supported dominant angle only for filtering.
    grid_a, grid_b = _classify_grid_lines(raw_lines_info, direction_a, direction_b, angle_tolerance)
    kept_a, rejected_a = _reject_dense_parallel_stripes(grid_a, config)
    kept_b, rejected_b = _reject_dense_parallel_stripes(grid_b, config)

    # Second pass: use the real detected angles of the remaining lines as the
    # grid angle. This is what lets the angle follow slow yaw drift.
    if config.get("reestimate_grid_angle_after_filter", True):
        detected_source = kept_a + kept_b
        if detected_source:
            detected_angle = _weighted_detected_grid_angle(
                detected_source,
                dominant_angle,
                config,
            )
            if detected_angle is not None:
                dominant_angle = detected_angle
                direction_a = dominant_angle
                direction_b = normalize_angle_90(dominant_angle + 90)

                # Re-classify using the newly detected angle so line merging and
                # debug drawing are consistent with the measured orientation.
                grid_a, grid_b = _classify_grid_lines(raw_lines_info, direction_a, direction_b, angle_tolerance)
                kept_a, rejected_a = _reject_dense_parallel_stripes(grid_a, config)
                kept_b, rejected_b = _reject_dense_parallel_stripes(grid_b, config)

    merged_a = _merge_parallel_lines(kept_a, direction_a, "grid_a", config)
    merged_b = _merge_parallel_lines(kept_b, direction_b, "grid_b", config)

    config["last_angle_filtered_line_count"] = len(grid_a) + len(grid_b)
    config["last_dense_rejected_line_count"] = len(rejected_a) + len(rejected_b)
    config["last_merged_line_count"] = len(merged_a) + len(merged_b)

    return merged_a + merged_b, dominant_angle



def detect_grid_lines(frame_rgb, config):
    """
    Detect ceiling grid lines.

    Pipeline:
      1. Canny + HoughLinesP gets raw line segments.
      2. Dominant grid angle is estimated with the configured angle prior.
      3. Lines are classified into the two orthogonal grid families.
      4. Dense parallel stripes, such as air-conditioner grilles, are rejected.
      5. Near-duplicate parallel segments are merged into one representative line.

    Return:
        lines_info: filtered and merged grid lines
        edges: Canny edge image
        dominant_angle: detected grid angle
    """
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)

    # Optional: CLAHE local contrast enhancement.
    if config.get("use_clahe", False):
        clahe = cv2.createCLAHE(
            clipLimit=config.get("clahe_clip_limit", 2.0),
            tileGridSize=(8, 8)
        )
        gray_proc = clahe.apply(gray)
    else:
        gray_proc = gray

    blur = cv2.GaussianBlur(gray_proc, (3, 3), 0)

    edges = cv2.Canny(
        blur,
        config["canny_low"],
        config["canny_high"]
    )

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=config["hough_threshold"],
        minLineLength=config["hough_min_line_length"],
        maxLineGap=config["hough_max_line_gap"]
    )

    raw_lines_info = []

    if lines is None:
        config["last_raw_line_count"] = 0
        config["last_angle_filtered_line_count"] = 0
        config["last_dense_rejected_line_count"] = 0
        config["last_merged_line_count"] = 0
        return [], edges, None

    for line in lines:
        x1, y1, x2, y2 = line[0]

        dx = x2 - x1
        dy = y2 - y1
        length = math.sqrt(dx * dx + dy * dy)

        if length < config["hough_min_line_length"]:
            continue

        angle = math.degrees(math.atan2(dy, dx))
        angle = normalize_angle_90(angle)

        raw_lines_info.append({
            "x1": int(x1),
            "y1": int(y1),
            "x2": int(x2),
            "y2": int(y2),
            "angle": angle,
            "length": length,
            "type": "raw"
        })

    config["last_raw_line_count"] = len(raw_lines_info)

    dominant_angle = estimate_dominant_grid_angle(
        raw_lines_info,
        config,
        prior_angle=config.get("last_dominant_angle", None)
    )

    if dominant_angle is None:
        config["last_angle_filtered_line_count"] = 0
        config["last_dense_rejected_line_count"] = 0
        config["last_merged_line_count"] = 0
        return [], edges, None

    lines_info, dominant_angle = _postprocess_grid_lines(raw_lines_info, dominant_angle, config)
    config["last_dominant_angle"] = dominant_angle

    return lines_info, edges, dominant_angle

def estimate_grid_angle(dominant_angle):
    """
    dominant_angle 就是目前畫面中的主要格線角度。
    """
    return dominant_angle


def estimate_pixel_shift(prev_gray, curr_gray):
    """
    用 phase correlation 估計前後兩張影像的平移量。
    適合車子移動不太快、影像沒有大幅旋轉的情況。
    """
    if prev_gray is None or curr_gray is None:
        return 0.0, 0.0, 0.0

    prev_f = np.float32(prev_gray)
    curr_f = np.float32(curr_gray)

    shift, response = cv2.phaseCorrelate(prev_f, curr_f)

    dx_pixel = shift[0]
    dy_pixel = shift[1]

    return dx_pixel, dy_pixel, response



# -----------------------------------------------------------------------------
# Motion-state architecture
# -----------------------------------------------------------------------------
# The vision module has three states:
#   FORWARD: update map position from image translation.
#            Car forward is defined in image space as: the same ceiling object
#            moves upward from frame to frame. The current navigation heading
#            decides where this forward motion goes on the map.
#   TURN:    do not update x/y position. Track image rotation and convert it to
#            vehicle rotation. Ceiling-image rotation is opposite to vehicle turn.
#   STOP:    do not update x/y position or turn angle. Still detects lines/lights
#            for debug display.

MOTION_FORWARD = "FORWARD"
MOTION_TURN = "TURN"
MOTION_STOP = "STOP"


def normalize_motion_state(value):
    """Normalize English/Chinese state labels into FORWARD/TURN/STOP."""
    s = str(value if value is not None else "FORWARD").strip().upper()
    mapping = {
        "FORWARD": MOTION_FORWARD,
        "MOVE": MOTION_FORWARD,
        "MOVING": MOTION_FORWARD,
        "DRIVE": MOTION_FORWARD,
        "GO": MOTION_FORWARD,
        "前進": MOTION_FORWARD,
        "直走": MOTION_FORWARD,
        "TURN": MOTION_TURN,
        "TURNING": MOTION_TURN,
        "ROTATE": MOTION_TURN,
        "轉彎": MOTION_TURN,
        "旋轉": MOTION_TURN,
        "STOP": MOTION_STOP,
        "STOPPED": MOTION_STOP,
        "IDLE": MOTION_STOP,
        "WAIT": MOTION_STOP,
        "VERIFY": MOTION_STOP,
        "DELIVERY": MOTION_STOP,
        "停止": MOTION_STOP,
        "驗證": MOTION_STOP,
        "送貨": MOTION_STOP,
    }
    return mapping.get(s, MOTION_FORWARD)


def get_motion_state(config):
    """
    Get the current vision state.

    A future motor/controller layer can set any of these keys before each frame:
        control_motion_state / motion_state / vision_state / navigation_state
    """
    raw = config.get(
        "control_motion_state",
        config.get("motion_state", config.get("vision_state", config.get("navigation_state", "FORWARD"))),
    )
    state = normalize_motion_state(raw)
    previous = config.get("last_motion_state", None)
    config["active_motion_state"] = state
    config["motion_state_changed"] = previous is not None and previous != state
    config["last_motion_state"] = state
    return state


def get_current_nav_heading(config):
    return str(config.get("current_nav_heading", config.get("planned_heading", "UP"))).upper()


def normalize_turn_direction(value):
    s = str(value if value is not None else "NONE").strip().upper()
    mapping = {
        "RIGHT": "RIGHT",
        "R": "RIGHT",
        "CW": "RIGHT",
        "右": "RIGHT",
        "右轉": "RIGHT",
        "LEFT": "LEFT",
        "L": "LEFT",
        "CCW": "LEFT",
        "左": "LEFT",
        "左轉": "LEFT",
        "NONE": "NONE",
        "STRAIGHT": "NONE",
        "": "NONE",
    }
    return mapping.get(s, "NONE")




def _parse_cell_like(value):
    """Parse [x, y], (x, y), or "x,y" into a float tuple."""
    if value is None or value == "":
        return None
    try:
        if isinstance(value, str):
            parts = value.replace("(", "").replace(")", "").split(",")
            if len(parts) != 2:
                return None
            return float(parts[0]), float(parts[1])
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return float(value[0]), float(value[1])
    except Exception:
        return None
    return None


def update_turn_target_context(config, pos_x_cell, pos_y_cell):
    """
    Keep the router/control-provided turn coordinate in the runtime config.

    Priority is explicit controller keys first, then router keys, then manual
    test keys. The coordinate is only a reference for TURN/monitoring; TURN
    state still does not integrate x/y motion.
    """
    candidates = [
        "control_turn_cell",
        "turn_target_cell",
        "router_turn_cell",
        "current_turn_cell",
        "planned_turn_cell",
        "next_turn_cell",
        "current_segment_target_cell",
    ]

    selected = None
    selected_key = ""
    for key in candidates:
        cell = _parse_cell_like(config.get(key, None))
        if cell is not None:
            selected = cell
            selected_key = key
            break

    if selected is None:
        config["turn_target_x_cell"] = ""
        config["turn_target_y_cell"] = ""
        config["turn_cell_distance_cell"] = ""
        config["turn_target_source"] = "NO_TURN_CELL"
        return None

    tx, ty = selected
    config["turn_target_x_cell"] = tx
    config["turn_target_y_cell"] = ty
    config["turn_cell_distance_cell"] = math.hypot(tx - float(pos_x_cell), ty - float(pos_y_cell))
    config["turn_target_source"] = selected_key
    return selected


def angle_delta_90(current_angle, previous_angle):
    """
    Frame-to-frame rotation delta for a square ceiling grid.

    The grid has 90-degree symmetry, so the absolute angle is ambiguous after a
    90-degree turn. Tracking small frame-to-frame deltas lets us accumulate the
    full turn as long as the camera rotates gradually.
    """
    return normalize_angle_90(float(current_angle) - float(previous_angle))


def nearest_straight_reference_angle(angle_deg):
    """
    Reference straight angle for FORWARD mode.

    The robot is expected to be aligned with horizontal/vertical ceiling grid
    lines. If the measured grid is slightly off from 0 or +/-90 degrees, keep
    that measured error as current_yaw_error_deg for correction.
    """
    if angle_deg is None:
        return None, None

    candidates = [-90.0, 0.0, 90.0]
    angle = float(angle_deg)
    ref = min(candidates, key=lambda r: abs(normalize_angle_90(angle - r)))
    err = normalize_angle_90(angle - ref)
    return ref, err


def update_forward_angle_status(grid_angle, config):
    """
    In FORWARD mode, keep small measured yaw error instead of snapping to 0/90.
    """
    ref, err = nearest_straight_reference_angle(grid_angle)
    if ref is None:
        config["current_straight_reference_angle_deg"] = ""
        config["current_yaw_error_deg"] = ""
        config["current_angle_status"] = "NO_GRID_ANGLE"
        config["effective_grid_angle_deg"] = ""
        return

    tolerance = float(config.get("forward_angle_error_accept_deg", 15.0))
    config["current_straight_reference_angle_deg"] = ref
    config["current_yaw_error_deg"] = err
    config["effective_grid_angle_deg"] = float(grid_angle)

    if abs(err) <= tolerance:
        if abs(err) <= float(config.get("forward_angle_deadband_deg", 0.5)):
            config["current_angle_status"] = "STRAIGHT"
        else:
            config["current_angle_status"] = "STRAIGHT_WITH_YAW_ERROR"
    else:
        config["current_angle_status"] = "ANGLE_OUT_OF_RANGE"


def _robot_motion_to_map_delta(forward_cell, right_cell, config, apply_lateral_scale=True):
    """
    Convert robot-frame motion into map dx/dy.

    Robot frame:
        +forward = car moves forward
        +right   = car moves to its right

    Map frame:
        x = column, y = row
        UP    -> y decreases
        DOWN  -> y increases
        RIGHT -> x increases
        LEFT  -> x decreases
    """
    heading = get_current_nav_heading(config)

    forward_cell = float(forward_cell)
    right_cell = float(right_cell)

    if apply_lateral_scale:
        # For straight forward localization, side drift from phase correlation is
        # usually noise. Set this to 0.0 during early tests; increase only if you
        # actually want sideways motion to affect the map estimate.
        right_cell *= float(config.get("lateral_motion_scale", 0.0))

    if heading == "UP":
        dx_cell = right_cell
        dy_cell = -forward_cell
    elif heading == "DOWN":
        dx_cell = -right_cell
        dy_cell = forward_cell
    elif heading == "RIGHT":
        dx_cell = forward_cell
        dy_cell = right_cell
    elif heading == "LEFT":
        dx_cell = -forward_cell
        dy_cell = -right_cell
    else:
        dx_cell = right_cell
        dy_cell = -forward_cell

    if config.get("invert_grid_x", False):
        dx_cell = -dx_cell
    if config.get("invert_grid_y", False):
        dy_cell = -dy_cell

    return dx_cell, dy_cell


def reset_turn_tracking(config, grid_angle):
    """Initialize TURN-state rotation tracking."""
    direction = normalize_turn_direction(config.get("turn_direction", config.get("planned_turn_direction", "NONE")))
    start = float(grid_angle) if grid_angle is not None else 0.0
    config["turn_tracking_active"] = True
    config["turn_direction_runtime"] = direction
    config["turn_start_grid_angle_deg"] = start
    config["turn_prev_grid_angle_deg"] = start
    config["turn_image_delta_deg"] = 0.0
    config["turn_vehicle_delta_deg"] = 0.0
    config["turn_image_rotation_deg"] = 0.0
    config["turn_vehicle_rotation_deg"] = 0.0
    config["turn_progress_deg"] = 0.0
    config["turn_completed_by_vision"] = False
    config["turn_tracking_status"] = "TURN_TRACKING_INITIALIZED"


def update_turn_tracking(grid_angle, config):
    """
    Track how many degrees the robot has turned.

    Sign convention:
        vehicle RIGHT turn = positive vehicle rotation
        vehicle LEFT turn  = negative vehicle rotation

    The ceiling image rotates opposite to the vehicle, so:
        vehicle_delta = -image_delta
    """
    direction = normalize_turn_direction(config.get("turn_direction", config.get("planned_turn_direction", "NONE")))
    target = abs(float(config.get("turn_target_deg", 90.0)))
    tolerance = abs(float(config.get("turn_completion_tolerance_deg", 5.0)))

    if grid_angle is None:
        config["turn_tracking_status"] = "NO_GRID_ANGLE"
        return

    needs_reset = (
        not bool(config.get("turn_tracking_active", False))
        or config.get("turn_direction_runtime", "NONE") != direction
        or bool(config.pop("turn_reset_request", False))
    )
    if needs_reset:
        reset_turn_tracking(config, grid_angle)
        return

    prev_angle = float(config.get("turn_prev_grid_angle_deg", grid_angle))
    image_delta = angle_delta_90(grid_angle, prev_angle)

    max_delta = float(config.get("turn_max_delta_per_frame_deg", 25.0))
    if abs(image_delta) > max_delta:
        # One bad Hough frame should not destroy accumulated turn estimation.
        config["turn_image_delta_deg"] = image_delta
        config["turn_vehicle_delta_deg"] = -image_delta
        config["turn_tracking_status"] = "TURN_DELTA_REJECTED"
        return

    image_rotation = float(config.get("turn_image_rotation_deg", 0.0)) + image_delta
    vehicle_delta = -image_delta
    vehicle_rotation = -image_rotation

    # Sign convention after real camera test:
    #   LEFT turn  -> ceiling image rotation negative, vehicle_rotation positive
    #   RIGHT turn -> ceiling image rotation positive, vehicle_rotation negative
    # Therefore progress should be positive for the commanded direction.
    if direction == "LEFT":
        progress = vehicle_rotation
    elif direction == "RIGHT":
        progress = -vehicle_rotation
    else:
        progress = abs(vehicle_rotation)

    completed = progress >= max(0.0, target - tolerance)

    config["turn_prev_grid_angle_deg"] = float(grid_angle)
    config["turn_image_delta_deg"] = image_delta
    config["turn_vehicle_delta_deg"] = vehicle_delta
    config["turn_image_rotation_deg"] = image_rotation
    config["turn_vehicle_rotation_deg"] = vehicle_rotation
    config["turn_progress_deg"] = progress
    config["turn_completed_by_vision"] = bool(completed)

    if completed:
        config["turn_tracking_status"] = "TURN_TARGET_REACHED"
    elif direction == "LEFT" and vehicle_rotation < -tolerance:
        config["turn_tracking_status"] = "TURNING_WRONG_DIRECTION"
    elif direction == "RIGHT" and vehicle_rotation > tolerance:
        config["turn_tracking_status"] = "TURNING_WRONG_DIRECTION"
    else:
        config["turn_tracking_status"] = "TURNING"


def _axis_component(axis_name, grid_a_cell, grid_b_cell):
    """Return grid_a/grid_b component by config name."""
    axis = str(axis_name).lower()
    if axis in ("grid_a", "a", "x", "forward_a"):
        return grid_a_cell
    if axis in ("grid_b", "b", "y", "forward_b"):
        return grid_b_cell
    return grid_a_cell


def _detected_axes_to_map_delta(grid_a_cell, grid_b_cell, config):
    """
    Backward-compatible map conversion for the old grid-axis modes.

    New state-machine mode should normally use map_motion_mode="image_forward",
    where forward is defined by image-up motion rather than grid_a/grid_b.
    """
    mode = str(config.get("map_motion_mode", "image_forward")).lower()

    if mode in ("image_forward", "state_machine"):
        # This function receives grid components, so image_forward should not
        # normally enter here. Fall back to old heading_forward behavior rather
        # than failing if a caller still uses it.
        mode = "heading_forward"

    if mode != "heading_forward":
        dx_cell = grid_a_cell
        dy_cell = grid_b_cell
        if config.get("swap_grid_xy", False):
            dx_cell, dy_cell = dy_cell, dx_cell
    else:
        heading = get_current_nav_heading(config)
        forward_axis = config.get("camera_forward_axis", "grid_a")
        right_axis = config.get("camera_right_axis", "grid_b")
        forward = float(config.get("camera_forward_sign", -1.0)) * _axis_component(
            forward_axis, grid_a_cell, grid_b_cell
        )
        right = float(config.get("camera_right_sign", 1.0)) * _axis_component(
            right_axis, grid_a_cell, grid_b_cell
        )
        dx_cell, dy_cell = _robot_motion_to_map_delta(forward, right, config, apply_lateral_scale=True)

    if config.get("invert_grid_x", False):
        dx_cell = -dx_cell
    if config.get("invert_grid_y", False):
        dy_cell = -dy_cell

    return dx_cell, dy_cell


def pixel_shift_to_grid_cell_shift(dx_pixel, dy_pixel, grid_angle, config):
    """
    Convert phase-correlation image displacement into map-cell displacement.

    In map_motion_mode="image_forward" / "state_machine":
        Forward is defined only by the image observation:
            same ceiling object moves upward in the image.
        Image coordinates:
            +x = right, +y = down
        Therefore:
            forward = -dy_pixel
            right   = +dx_pixel

    Then current_nav_heading decides where forward goes on the map.
    """
    cell_per_pixel = float(config.get("cell_per_pixel", 0.0025))
    mode = str(config.get("map_motion_mode", "image_forward")).lower()

    if mode in ("image_forward", "state_machine"):
        forward_cell = (-float(dy_pixel)) * cell_per_pixel * float(config.get("image_forward_sign", 1.0))
        right_cell = (float(dx_pixel)) * cell_per_pixel * float(config.get("image_right_sign", 1.0))
        return _robot_motion_to_map_delta(
            forward_cell,
            right_cell,
            config,
            apply_lateral_scale=True,
        )

    # Old / fallback mode: project onto detected ceiling grid axes first.
    if grid_angle is None:
        grid_angle = float(config.get("fallback_grid_angle_deg", 0.0))

    angle = math.radians(float(grid_angle) + float(config.get("grid_angle_offset_deg", 0.0)))

    ax = math.cos(angle)
    ay = math.sin(angle)
    bx = -math.sin(angle)
    by = math.cos(angle)

    grid_a_cell = (dx_pixel * ax + dy_pixel * ay) * cell_per_pixel
    grid_b_cell = (dx_pixel * bx + dy_pixel * by) * cell_per_pixel

    return _detected_axes_to_map_delta(grid_a_cell, grid_b_cell, config)



# -----------------------------------------------------------------------------
# Forward-motion displacement estimators
# -----------------------------------------------------------------------------
# In FORWARD state, prefer actual object tracking over full-frame phase:
#   1. Same ceiling light center movement
#   2. Ceiling grid-line rho/position movement
#   3. Phase correlation fallback
#
# All estimators return image displacement in OpenCV image coordinates:
#   +x = right, +y = down.
# Since the car moving forward makes the ceiling move upward in the image,
# forward motion normally corresponds to dy_pixel < 0.

def _current_main_light(lights):
    """Choose the most stable light candidate for frame-to-frame tracking."""
    if not lights:
        return None
    return max(lights, key=lambda item: float(item.get("area", 0.0)))


def estimate_light_center_shift(lights, config):
    """
    Track the same visible light by its center position.

    Highest-priority FORWARD estimator:
        dx_pixel = current_cx - previous_cx
        dy_pixel = current_cy - previous_cy
    """
    if not config.get("light_center_track_enabled", True):
        config["light_track_status"] = "DISABLED"
        return None

    current = _current_main_light(lights)
    if current is None:
        config["light_track_status"] = "NO_LIGHT"
        return None

    min_area = float(config.get("light_center_track_min_area", config.get("light_min_area", 0.0)))
    if float(current.get("area", 0.0)) < min_area:
        config["light_track_status"] = "LIGHT_TOO_SMALL"
        return None

    cx = float(current["cx"])
    cy = float(current["cy"])
    area = float(current.get("area", 1.0))

    prev_cx = config.get("light_track_prev_cx", None)
    prev_cy = config.get("light_track_prev_cy", None)
    prev_area = config.get("light_track_prev_area", None)

    def store_current():
        config["light_track_prev_cx"] = cx
        config["light_track_prev_cy"] = cy
        config["light_track_prev_area"] = area

    if prev_cx is None or prev_cy is None:
        store_current()
        config["light_track_status"] = "INIT"
        return None

    prev_cx = float(prev_cx)
    prev_cy = float(prev_cy)
    dx = cx - prev_cx
    dy = cy - prev_cy
    dist = math.hypot(dx, dy)

    max_match = float(config.get("light_center_track_max_match_px", 500.0))
    if dist > max_match:
        store_current()
        config["light_track_status"] = "JUMP_REJECTED"
        config["light_track_match_distance_px"] = dist
        return None

    if prev_area is not None:
        prev_area = max(float(prev_area), 1.0)
        area_ratio = area / prev_area
        min_ratio = float(config.get("light_center_track_min_area_ratio", 0.25))
        max_ratio = float(config.get("light_center_track_max_area_ratio", 4.0))
        if area_ratio < min_ratio or area_ratio > max_ratio:
            store_current()
            config["light_track_status"] = "AREA_RATIO_REJECTED"
            config["light_track_match_distance_px"] = dist
            return None

    store_current()

    config["light_track_status"] = "OK"
    config["light_track_dx_pixel"] = dx
    config["light_track_dy_pixel"] = dy
    config["light_track_match_distance_px"] = dist

    confidence = float(config.get("light_center_track_confidence", 1.0))
    return {
        "source": "light_center",
        "dx_pixel": dx,
        "dy_pixel": dy,
        "response": confidence,
    }


def _extract_line_positions(lines_info, axis, min_abs_normal):
    """
    Convert merged grid lines into approximate image x/y positions.

    The line model is rho = p dot n.
    For mostly horizontal lines, y_position ~= rho / ny.
    For mostly vertical lines,   x_position ~= rho / nx.
    """
    positions = []

    for line in lines_info:
        angle = math.radians(float(line.get("angle", 0.0)))
        nx = -math.sin(angle)
        ny = math.cos(angle)
        rho = float(line.get("rho", 0.0))

        if axis == "y":
            if abs(ny) < min_abs_normal:
                continue
            positions.append(rho / ny)
        else:
            if abs(nx) < min_abs_normal:
                continue
            positions.append(rho / nx)

    return sorted(float(v) for v in positions if np.isfinite(v))


def _match_position_deltas(prev_positions, curr_positions, max_match_px):
    """
    Match repeated ceiling grid-line positions frame-to-frame and return median delta.
    """
    if not prev_positions or not curr_positions:
        return None, 0

    prev = [float(v) for v in prev_positions]
    deltas = []
    used = set()

    for curr in curr_positions:
        best_i = None
        best_dist = float("inf")
        for i, pv in enumerate(prev):
            if i in used:
                continue
            d = abs(float(curr) - pv)
            if d < best_dist:
                best_dist = d
                best_i = i

        if best_i is not None and best_dist <= max_match_px:
            used.add(best_i)
            deltas.append(float(curr) - prev[best_i])

    if not deltas:
        return None, 0

    return float(np.median(deltas)), len(deltas)


def estimate_grid_line_shift(lines_info, config):
    """
    Estimate image displacement from tracked ceiling grid-line positions.

    The y component comes from horizontal grid lines and is the main FORWARD
    measurement. The x component from vertical lines is optional.
    """
    if not config.get("grid_line_track_enabled", True):
        config["grid_track_status"] = "DISABLED"
        return None

    min_abs_normal = float(config.get("grid_line_track_min_abs_normal", 0.55))
    max_match_px = float(config.get("grid_line_track_max_match_px", 180.0))
    min_y_matches = int(config.get("grid_line_track_min_y_matches", 1))
    min_x_matches = int(config.get("grid_line_track_min_x_matches", 1))

    curr_y = _extract_line_positions(lines_info, "y", min_abs_normal)
    curr_x = _extract_line_positions(lines_info, "x", min_abs_normal)

    prev_y = config.get("grid_track_prev_y_positions", [])
    prev_x = config.get("grid_track_prev_x_positions", [])

    dy, y_count = _match_position_deltas(prev_y, curr_y, max_match_px)
    dx, x_count = _match_position_deltas(prev_x, curr_x, max_match_px)

    config["grid_track_prev_y_positions"] = curr_y
    config["grid_track_prev_x_positions"] = curr_x
    config["grid_track_y_match_count"] = y_count
    config["grid_track_x_match_count"] = x_count

    if dy is None or y_count < min_y_matches:
        if not curr_y:
            config["grid_track_status"] = "NO_HORIZONTAL_GRID_LINES"
        elif not prev_y:
            config["grid_track_status"] = "INIT"
        else:
            config["grid_track_status"] = "NO_Y_MATCH"
        return None

    if dx is None or x_count < min_x_matches:
        dx = 0.0
        x_count = 0

    max_abs_dy = float(config.get("grid_line_track_max_abs_dy_px", 220.0))
    max_abs_dx = float(config.get("grid_line_track_max_abs_dx_px", 220.0))
    if abs(dy) > max_abs_dy or abs(dx) > max_abs_dx:
        config["grid_track_status"] = "JUMP_REJECTED"
        config["grid_track_dx_pixel"] = dx
        config["grid_track_dy_pixel"] = dy
        return None

    confidence = min(
        1.0,
        float(config.get("grid_line_track_base_confidence", 0.55))
        + 0.1 * max(0, y_count - 1)
        + 0.05 * max(0, x_count - 1),
    )

    config["grid_track_status"] = "OK"
    config["grid_track_dx_pixel"] = dx
    config["grid_track_dy_pixel"] = dy

    return {
        "source": "grid_line",
        "dx_pixel": dx,
        "dy_pixel": dy,
        "response": confidence,
    }


def select_forward_motion_shift(lights, lines_info, phase_dx_pixel, phase_dy_pixel, phase_response, config):
    """
    Choose FORWARD displacement source by priority:
        light_center -> grid_line -> phase
    """
    light_est = estimate_light_center_shift(lights, config)
    grid_est = estimate_grid_line_shift(lines_info, config)

    phase_est = None
    if config.get("phase_fallback_enabled", True):
        phase_est = {
            "source": "phase",
            "dx_pixel": float(phase_dx_pixel),
            "dy_pixel": float(phase_dy_pixel),
            "response": float(phase_response),
        }

    priority = config.get("forward_motion_source_priority", ["light_center", "grid_line", "phase"])
    if isinstance(priority, str):
        priority = [item.strip() for item in priority.split(",") if item.strip()]

    choices = {
        "light": light_est,
        "light_center": light_est,
        "grid": grid_est,
        "grid_line": grid_est,
        "phase": phase_est,
    }

    selected = None
    for name in priority:
        candidate = choices.get(str(name).lower())
        if candidate is None:
            continue

        source = candidate["source"]
        min_response = float(config.get(
            f"{source}_min_response",
            config.get("motion_source_min_response", 0.03),
        ))

        if float(candidate.get("response", 0.0)) >= min_response:
            selected = candidate
            break

    if selected is None:
        selected = {
            "source": "none",
            "dx_pixel": 0.0,
            "dy_pixel": 0.0,
            "response": 0.0,
        }

    config["motion_source"] = selected["source"]
    config["selected_response"] = float(selected.get("response", 0.0))
    config["phase_dx_pixel"] = float(phase_dx_pixel)
    config["phase_dy_pixel"] = float(phase_dy_pixel)
    config["phase_response_raw"] = float(phase_response)

    return (
        float(selected.get("dx_pixel", 0.0)),
        float(selected.get("dy_pixel", 0.0)),
        float(selected.get("response", 0.0)),
    )


def load_navigation_map(config):
    """Load visualization/map_config.json so localization can check obstacle cells."""
    global MAP_CONFIG_CACHE
    if MAP_CONFIG_CACHE is not None:
        return MAP_CONFIG_CACHE

    map_path = Path(config.get("map_config_path", "../visualization/map_config.json"))
    if not map_path.is_absolute():
        map_path = VISION_DIR / map_path
    map_path = map_path.resolve()

    if not map_path.exists():
        MAP_CONFIG_CACHE = {}
        return MAP_CONFIG_CACHE

    try:
        with open(map_path, "r") as f:
            MAP_CONFIG_CACHE = json.load(f)
    except Exception:
        MAP_CONFIG_CACHE = {}

    return MAP_CONFIG_CACHE


def position_to_map_cell(x_cell, y_cell, config):
    """Convert continuous grid position to integer map cell index."""
    method = config.get("map_position_to_cell_method", "round")
    if method == "floor":
        return int(math.floor(x_cell)), int(math.floor(y_cell))
    return int(round(x_cell)), int(round(y_cell))


def get_map_cell_type_from_config(map_config, col, row):
    rows = map_config.get("occupancy_grid", {}).get("rows", [])
    if row < 0 or row >= len(rows):
        return "out_of_bounds"
    if col < 0 or col >= len(rows[row]):
        return "out_of_bounds"
    ch = rows[row][col]
    if ch == "#":
        return "obstacle"
    if ch == "L":
        return "light"
    if ch == ".":
        return "free"
    return "unknown"


def check_obstacle_at_position(x_cell, y_cell, config):
    """Return map_col, map_row, map_cell_type, is_obstacle."""
    if not config.get("obstacle_check_enabled", True):
        return None, None, "disabled", False

    map_config = load_navigation_map(config)
    col, row = position_to_map_cell(x_cell, y_cell, config)
    cell_type = get_map_cell_type_from_config(map_config, col, row)
    is_obstacle = cell_type in ("obstacle", "out_of_bounds")
    return col, row, cell_type, is_obstacle


def image_vector_to_grid_cell(dx_pixel, dy_pixel, grid_angle, config):
    """
    Convert an image vector, such as light center relative to image center,
    into map-cell vector.

    In image_forward/state_machine mode:
        image up = robot forward
        image right = robot right
    This uses the current heading to express that vector in map x/y.
    """
    cell_per_pixel = float(config.get("cell_per_pixel", 0.0025))
    mode = str(config.get("map_motion_mode", "image_forward")).lower()

    if mode in ("image_forward", "state_machine"):
        forward_cell = (-float(dy_pixel)) * cell_per_pixel * float(config.get("image_forward_sign", 1.0))
        right_cell = (float(dx_pixel)) * cell_per_pixel * float(config.get("image_right_sign", 1.0))

        vx_cell, vy_cell = _robot_motion_to_map_delta(
            forward_cell,
            right_cell,
            config,
            apply_lateral_scale=False,
        )

        vx_cell *= float(config.get("light_position_sign_x", 1.0))
        vy_cell *= float(config.get("light_position_sign_y", 1.0))
        return vx_cell, vy_cell

    # Old / fallback mode.
    if grid_angle is None:
        grid_angle = float(config.get("fallback_grid_angle_deg", 0.0))

    angle = math.radians(float(grid_angle) + float(config.get("grid_angle_offset_deg", 0.0)))

    ax = math.cos(angle)
    ay = math.sin(angle)
    bx = -math.sin(angle)
    by = math.cos(angle)

    grid_a_cell = (dx_pixel * ax + dy_pixel * ay) * cell_per_pixel
    grid_b_cell = (dx_pixel * bx + dy_pixel * by) * cell_per_pixel

    vx_cell, vy_cell = _detected_axes_to_map_delta(grid_a_cell, grid_b_cell, config)

    vx_cell *= float(config.get("light_position_sign_x", 1.0))
    vy_cell *= float(config.get("light_position_sign_y", 1.0))

    return vx_cell, vy_cell


def get_light_cells_from_map_config(map_config):
    """Return all known light landmarks from map_config.json."""
    lights = []

    # Preferred explicit list.
    for item in map_config.get("light_cells", []):
        try:
            lights.append({
                "name": item.get("name", f"LIGHT_R{int(item['y']):02d}_C{int(item['x']):02d}"),
                "x": float(item["x"]),
                "y": float(item["y"]),
            })
        except Exception:
            continue

    if lights:
        return lights

    # Fallback: scan occupancy grid for L cells.
    rows = map_config.get("occupancy_grid", {}).get("rows", [])
    for row_idx, row in enumerate(rows):
        for col_idx, ch in enumerate(row):
            if ch == "L":
                lights.append({
                    "name": f"LIGHT_R{row_idx:02d}_C{col_idx:02d}",
                    "x": float(col_idx),
                    "y": float(row_idx),
                })

    return lights


def identify_lights_on_map(lights, frame_shape, pos_x_cell, pos_y_cell, grid_angle, config):
    """
    Identify visible ceiling lights and use them as absolute landmarks.

    For each detected image light:
      1. Convert image offset from camera center into ceiling-grid-cell vector.
      2. For each known map light, infer where the robot/camera should be:
             robot_from_light = known_light_position - light_vector_from_robot
      3. Compare robot_from_light with the current grid/odometry estimate.
      4. Choose the known light that gives the smallest robot-position error.

    This makes the light useful not only as an ID, but also as an independent
    absolute-position check for the grid/phase-correlation localization.
    """
    if not config.get("light_identification_enabled", True):
        return []

    map_config = load_navigation_map(config)
    known_lights = get_light_cells_from_map_config(map_config)
    if not known_lights:
        return []

    h, w = frame_shape[:2]
    origin_x = float(config.get("light_image_origin_x", w / 2.0))
    origin_y = float(config.get("light_image_origin_y", h / 2.0))

    max_match_dist = float(config.get("light_match_max_distance_cells", 4.0))
    max_validation_error = float(config.get("light_validation_max_error_cells", 2.0))
    offset_x = float(config.get("camera_light_offset_x_cells", 0.0))
    offset_y = float(config.get("camera_light_offset_y_cells", 0.0))

    results = []
    for idx, light in enumerate(lights):
        dx_img = float(light["cx"]) - origin_x
        dy_img = float(light["cy"]) - origin_y

        # Vector from robot/camera position to the visible light, in map-cell basis.
        light_vec_x, light_vec_y = image_vector_to_grid_cell(dx_img, dy_img, grid_angle, config)

        # The old estimate is still useful for debugging: where this light would be
        # if current grid/odometry position were perfect.
        est_light_x = float(pos_x_cell) + light_vec_x + offset_x
        est_light_y = float(pos_y_cell) + light_vec_y + offset_y

        best = None
        best_light_dist = float("inf")
        best_robot_error = float("inf")
        best_robot_x = None
        best_robot_y = None

        for known in known_lights:
            # Candidate robot position inferred from this known light landmark.
            robot_x_from_light = float(known["x"]) - light_vec_x - offset_x
            robot_y_from_light = float(known["y"]) - light_vec_y - offset_y

            robot_error = math.hypot(robot_x_from_light - float(pos_x_cell),
                                     robot_y_from_light - float(pos_y_cell))
            light_dist = math.hypot(est_light_x - float(known["x"]),
                                    est_light_y - float(known["y"]))

            # Primary criterion: which known light gives the closest robot pose
            # to the current grid/odometry estimate. This equals light_dist when
            # offset/signs are consistent, but is clearer for validation.
            if robot_error < best_robot_error:
                best = known
                best_robot_error = robot_error
                best_light_dist = light_dist
                best_robot_x = robot_x_from_light
                best_robot_y = robot_y_from_light

        matched = best is not None and best_light_dist <= max_match_dist
        validated = matched and best_robot_error <= max_validation_error

        if not matched:
            status = "UNKNOWN_LIGHT"
        elif validated:
            status = "LIGHT_VALIDATED"
        else:
            status = "LIGHT_MISMATCH"

        results.append({
            "detected_index": idx,
            "estimated_x_cell": est_light_x,
            "estimated_y_cell": est_light_y,
            "matched": matched,
            "validated": validated,
            "validation_status": status,
            "match_name": best["name"] if matched else "UNKNOWN_LIGHT",
            "match_x_cell": best["x"] if best is not None else None,
            "match_y_cell": best["y"] if best is not None else None,
            "match_distance_cell": best_light_dist if best is not None else None,
            "light_based_x_cell": best_robot_x if matched else None,
            "light_based_y_cell": best_robot_y if matched else None,
            "light_position_error_cell": best_robot_error if best is not None else None,
        })

    return results

def select_primary_light_match(light_matches):
    """Choose the best matched light among detected image lights."""
    if not light_matches:
        return {
            "match_name": "NO_LIGHT",
            "estimated_x_cell": "",
            "estimated_y_cell": "",
            "match_distance_cell": "",
            "matched": False,
            "validated": False,
            "validation_status": "NO_LIGHT",
            "light_based_x_cell": "",
            "light_based_y_cell": "",
            "light_position_error_cell": "",
        }

    matched = [m for m in light_matches if m.get("matched")]
    candidates = matched if matched else light_matches
    best = min(candidates, key=lambda m: float("inf") if m.get("match_distance_cell") is None else m["match_distance_cell"])
    return best


def draw_debug(frame_rgb, lights, lines_info, zone, grid_angle, dx_pixel, dy_pixel, dx_cell, dy_cell, response, pos_x_cell, pos_y_cell, config, map_col=None, map_row=None, map_cell_type="unknown", is_obstacle=False, primary_light_match=None):
    """
    在畫面上畫出偵測結果。
    """
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    debug = frame_bgr.copy()

    # draw lights
    for i, light in enumerate(lights):
        x = int(light["x"])
        y = int(light["y"])
        w = int(light["w"])
        h = int(light["h"])
        cx = int(light["cx"])
        cy = int(light["cy"])

        cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.circle(debug, (cx, cy), 5, (0, 0, 255), -1)
        light_label = f"Light {i}"

        # If this detected bright region has been matched to a known map light,
        # show the map landmark name directly on the yellow box.
        if primary_light_match is not None:
            if primary_light_match.get("detected_index", None) == i:
                match_name = primary_light_match.get("match_name", "")
                status = primary_light_match.get("validation_status", "")
                if match_name and match_name not in ("NO_LIGHT", "UNKNOWN_LIGHT"):
                    light_label = match_name
                    if status == "LIGHT_VALIDATED":
                        light_label += " OK"

        cv2.putText(
            debug,
            light_label,
            (x, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

    # draw grid lines
    for line in lines_info:
        x1 = line["x1"]
        y1 = line["y1"]
        x2 = line["x2"]
        y2 = line["y2"]

        if line["type"] == "grid_a":
            color = (255, 0, 0)
        elif line["type"] == "grid_b":
            color = (0, 255, 0)
        else:
            color = (128, 128, 128)

        cv2.line(debug, (x1, y1), (x2, y2), color, 1)

    text_lines = [
        f"Mode: {'camera' if config.get('use_camera', True) else 'test image'}",
        f"Zone: {zone}",
        f"Lights: {len(lights)}",
        f"Grid angle: {grid_angle:.2f} deg" if grid_angle is not None else "Grid angle: N/A",
        f"State: {config.get('active_motion_state', config.get('motion_state', 'FORWARD'))}, "
        f"Heading: {config.get('current_nav_heading', config.get('planned_heading', 'N/A'))}, motion={config.get('map_motion_mode', 'image_forward')}",
        f"Forward def: image_up -> {config.get('current_nav_heading', config.get('planned_heading', 'N/A'))}, "
        f"yaw_err={config.get('current_yaw_error_deg', '')}, status={config.get('current_angle_status', '')}",
        f"Turn: dir={config.get('turn_direction', config.get('planned_turn_direction', 'NONE'))}, "
        f"img_rot={config.get('turn_image_rotation_deg', 0.0)}, veh_rot={config.get('turn_vehicle_rotation_deg', 0.0)}, "
        f"progress={config.get('turn_progress_deg', 0.0)}/{config.get('turn_target_deg', 90.0)}, done={config.get('turn_completed_by_vision', False)}",
        f"Turn cell: ({config.get('turn_target_x_cell', '')},{config.get('turn_target_y_cell', '')}), "
        f"dist={config.get('turn_cell_distance_cell', '')}, source={config.get('turn_target_source', '')}",
        f"Angle update: det={config.get('last_detected_angle_update_deg', 0.0):.2f} deg, "
        f"support={config.get('last_detected_angle_support_count', 0)}",
        f"Lines: raw={config.get('last_raw_line_count', 0)}, angle={config.get('last_angle_filtered_line_count', 0)}, "
        f"dense_rej={config.get('last_dense_rejected_line_count', 0)}, merged={config.get('last_merged_line_count', 0)}",
        f"Motion source: {config.get('motion_source', 'N/A')}, selected_resp={config.get('selected_response', response):.3f}, "
        f"phase=({config.get('phase_dx_pixel', dx_pixel):.2f},{config.get('phase_dy_pixel', dy_pixel):.2f}) r={config.get('phase_response_raw', response):.3f}",
        f"Track: light={config.get('light_track_status', '')} dy={config.get('light_track_dy_pixel', '')}, "
        f"grid={config.get('grid_track_status', '')} dy={config.get('grid_track_dy_pixel', '')}",
        f"dx, dy pixel selected: {dx_pixel:.2f}, {dy_pixel:.2f}",
        f"dx, dy cell: {dx_cell:.4f}, {dy_cell:.4f}",
        f"motion response: {response:.3f}",
        f"cell_per_pixel: {config['cell_per_pixel']:.6f}",
        f"Estimated grid position: x={pos_x_cell:.3f} cell, y={pos_y_cell:.3f} cell",
        f"Map cell: col={map_col}, row={map_row}, type={map_cell_type}",
        f"Detected light: {(primary_light_match or {}).get('match_name', 'NO_LIGHT')}",
        f"Light est: x={(primary_light_match or {}).get('estimated_x_cell', '')}, y={(primary_light_match or {}).get('estimated_y_cell', '')}, dist={(primary_light_match or {}).get('match_distance_cell', '')}",
        f"Light-based robot pos: x={(primary_light_match or {}).get('light_based_x_cell', '')}, y={(primary_light_match or {}).get('light_based_y_cell', '')}",
        f"Light validation: {(primary_light_match or {}).get('validation_status', 'NO_LIGHT')}, err={(primary_light_match or {}).get('light_position_error_cell', '')}",
        "OBSTACLE WARNING" if is_obstacle and config.get("show_obstacle_warning", True) else "Map status: free/light/unknown",
        "Keys: q quit | +/- cell scale | [/] light threshold | r reset pos | s save config"
    ]

    y0 = 30
    for text in text_lines:
        cv2.putText(
            debug,
            text,
            (20, y0),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2
        )
        y0 += 28

    return debug


def draw_lines_only_debug(frame_rgb, lines_info, dominant_angle=None):
    """
    單獨顯示 Hough line 偵測結果。
    藍色：主要格線方向 A
    綠色：主要格線方向 B，也就是 A + 90 度
    """
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    debug = np.zeros_like(frame_bgr)

    for line in lines_info:
        x1 = line["x1"]
        y1 = line["y1"]
        x2 = line["x2"]
        y2 = line["y2"]

        if line["type"] == "grid_a":
            color = (255, 0, 0)
        elif line["type"] == "grid_b":
            color = (0, 255, 0)
        else:
            color = (128, 128, 128)

        cv2.line(debug, (x1, y1), (x2, y2), color, 2)

    text = "Lines only: blue=grid A, green=grid B"
    if dominant_angle is not None:
        text += f", dominant={dominant_angle:.2f} deg"

    cv2.putText(
        debug,
        text,
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    return debug


def save_debug_outputs(
    frame_count,
    frame_rgb,
    gray,
    light_raw_mask,
    light_clean_mask,
    edges,
    debug,
    lines_only_debug
):
    """
    將中間步驟存成圖片，方便事後檢查。
    """
    output_dir = VISION_DIR / "debug_output"
    output_dir.mkdir(exist_ok=True)

    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    cv2.imwrite(str(output_dir / f"{frame_count:05d}_raw.jpg"), frame_bgr)
    cv2.imwrite(str(output_dir / f"{frame_count:05d}_gray.jpg"), gray)
    cv2.imwrite(str(output_dir / f"{frame_count:05d}_light_raw_mask.jpg"), light_raw_mask)
    cv2.imwrite(str(output_dir / f"{frame_count:05d}_light_clean_mask.jpg"), light_clean_mask)
    cv2.imwrite(str(output_dir / f"{frame_count:05d}_edges.jpg"), edges)
    cv2.imwrite(str(output_dir / f"{frame_count:05d}_lines_only.jpg"), lines_only_debug)
    cv2.imwrite(str(output_dir / f"{frame_count:05d}_final_debug.jpg"), debug)


def init_trajectory_log(config):
    """
    Trajectory log is still stored at project_root/data/trajectory_log.csv.
    It can be disabled during single-image debugging by setting log_trajectory=false.
    """
    data_dir = PROJECT_ROOT / "data"
    data_dir.mkdir(exist_ok=True)

    trajectory_path = data_dir / "trajectory_log.csv"

    if not trajectory_path.exists():
        with open(trajectory_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "frame",
                "motion_state",
                "current_heading",
                "x_cell",
                "y_cell",
                "dx_pixel",
                "dy_pixel",
                "dx_cell",
                "dy_cell",
                "zone",
                "grid_angle",
                "phase_response",
                "yaw_error_deg",
                "angle_status",
                "turn_direction",
                "turn_image_rotation_deg",
                "turn_vehicle_rotation_deg",
                "turn_progress_deg",
                "turn_completed_by_vision",
                "map_col",
                "map_row",
                "map_cell_type",
                "is_obstacle",
                "detected_light_id",
                "detected_light_x_cell",
                "detected_light_y_cell",
                "detected_light_distance_cell",
                "light_based_x_cell",
                "light_based_y_cell",
                "light_position_error_cell",
                "light_validation_status",
                "motion_source",
                "selected_response",
                "light_track_status",
                "grid_track_status",
                "turn_target_x_cell",
                "turn_target_y_cell",
                "turn_cell_distance_cell",
                "turn_target_source"
            ])

    return trajectory_path


def show_debug_windows(config, debug, gray, light_raw_mask, light_clean_mask, edges, lines_only_debug):
    if not config.get("show_debug", True):
        return

    cv2.imshow("Ceiling Locator Debug", debug)

    if config.get("debug_show_gray", False):
        cv2.imshow("01 Gray", gray)

    if config.get("debug_show_light_raw_mask", False):
        cv2.imshow("02 Light Raw Mask", light_raw_mask)

    if config.get("debug_show_light_clean_mask", False):
        cv2.imshow("03 Light Clean Mask", light_clean_mask)

    if config.get("debug_show_edges", False):
        cv2.imshow("04 Grid Edges", edges)

    if config.get("debug_show_lines_only", False):
        cv2.imshow("05 Lines Only", lines_only_debug)


def process_frame(frame_rgb, config, prev_gray, pos_x_cell, pos_y_cell, frame_count, trajectory_path, return_debug=False):
    frame_rgb = apply_gamma_correction(frame_rgb, config)
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

    lights, light_raw_mask, light_clean_mask = detect_lights(frame_rgb, config)
    zone = estimate_light_zone(lights, frame_rgb.shape)

    lines_info, edges, dominant_angle = detect_grid_lines(frame_rgb, config)
    grid_angle = dominant_angle

    # On the first frame, optionally start from a known absolute map cell.
    if frame_count == 0 and "initial_map_cell" in config:
        try:
            pos_x_cell = float(config["initial_map_cell"][0])
            pos_y_cell = float(config["initial_map_cell"][1])
        except Exception:
            pass

    motion_state = get_motion_state(config)

    # State-specific angle handling.
    if motion_state == MOTION_FORWARD:
        update_forward_angle_status(grid_angle, config)
        # Leaving turn mode: keep accumulated debug values, but mark tracker inactive.
        config["turn_tracking_active"] = False
    elif motion_state == MOTION_TURN:
        update_turn_tracking(grid_angle, config)
        # During a turn, yaw relative to a straight grid is not a controller error.
        config["current_angle_status"] = "TURNING"
    elif motion_state == MOTION_STOP:
        config["turn_tracking_status"] = "STOPPED"
        config["current_angle_status"] = "STOPPED"

    phase_dx_pixel, phase_dy_pixel, phase_response = estimate_pixel_shift(prev_gray, gray_blur)

    # State-specific displacement source selection.
    if motion_state == MOTION_FORWARD:
        dx_pixel, dy_pixel, response = select_forward_motion_shift(
            lights,
            lines_info,
            phase_dx_pixel,
            phase_dy_pixel,
            phase_response,
            config,
        )
        dx_cell, dy_cell = pixel_shift_to_grid_cell_shift(dx_pixel, dy_pixel, grid_angle, config)
    else:
        # TURN and STOP do not integrate translation into map x/y.
        # Keep raw phase values for debugging, but do not use them as map motion.
        dx_pixel, dy_pixel, response = phase_dx_pixel, phase_dy_pixel, phase_response
        config["motion_source"] = motion_state.lower()
        config["selected_response"] = float(response)
        config["phase_dx_pixel"] = float(phase_dx_pixel)
        config["phase_dy_pixel"] = float(phase_dy_pixel)
        config["phase_response_raw"] = float(phase_response)
        # Still update trackers so FORWARD can resume smoothly after STOP/TURN.
        estimate_light_center_shift(lights, config)
        estimate_grid_line_shift(lines_info, config)
        dx_cell, dy_cell = 0.0, 0.0

    proposed_x_cell = pos_x_cell + dx_cell
    proposed_y_cell = pos_y_cell + dy_cell
    map_col, map_row, map_cell_type, is_obstacle = check_obstacle_at_position(
        proposed_x_cell, proposed_y_cell, config
    )

    # Identify which configured ceiling light is visible in the current image.
    # In TURN/STOP this is only for debug unless correction is explicitly enabled.
    light_matches = identify_lights_on_map(
        lights,
        frame_rgb.shape,
        proposed_x_cell,
        proposed_y_cell,
        grid_angle,
        config
    )
    primary_light_match = select_primary_light_match(light_matches)

    # Only FORWARD integrates the selected displacement source.
    if motion_state == MOTION_FORWARD:
        update_min_response = float(config.get(
            "selected_motion_response_min",
            config.get("phase_response_min", 0.15),
        ))
        if response > update_min_response:
            if not (is_obstacle and config.get("stop_update_when_obstacle_predicted", False)):
                pos_x_cell = proposed_x_cell
                pos_y_cell = proposed_y_cell
                map_col, map_row, map_cell_type, is_obstacle = check_obstacle_at_position(
                    pos_x_cell, pos_y_cell, config
                )
                light_matches = identify_lights_on_map(
                    lights,
                    frame_rgb.shape,
                    pos_x_cell,
                    pos_y_cell,
                    grid_angle,
                    config
                )
                primary_light_match = select_primary_light_match(light_matches)
    else:
        # Keep reported map cell consistent with the unchanged pose.
        map_col, map_row, map_cell_type, is_obstacle = check_obstacle_at_position(
            pos_x_cell, pos_y_cell, config
        )

    # Optional landmark correction only during FORWARD. Do not pull pose while
    # stopped for validation/delivery, and do not correct x/y during a rotation.
    if (motion_state == MOTION_FORWARD
            and config.get("light_position_correction_enabled", False)
            and primary_light_match.get("validated", False)):
        try:
            alpha = float(config.get("light_position_correction_alpha", 0.25))
            alpha = max(0.0, min(1.0, alpha))
            lx = float(primary_light_match["light_based_x_cell"])
            ly = float(primary_light_match["light_based_y_cell"])
            pos_x_cell = (1.0 - alpha) * pos_x_cell + alpha * lx
            pos_y_cell = (1.0 - alpha) * pos_y_cell + alpha * ly
            map_col, map_row, map_cell_type, is_obstacle = check_obstacle_at_position(
                pos_x_cell, pos_y_cell, config
            )
        except Exception:
            pass

    # Keep turn coordinate/distance available to debug, log, map visualizer, and controller.
    update_turn_target_context(config, pos_x_cell, pos_y_cell)

    if config.get("log_trajectory", True):
        with open(trajectory_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                time.time(),
                frame_count,
                motion_state,
                get_current_nav_heading(config),
                pos_x_cell,
                pos_y_cell,
                dx_pixel,
                dy_pixel,
                dx_cell,
                dy_cell,
                zone,
                grid_angle if grid_angle is not None else "",
                response,
                config.get("current_yaw_error_deg", ""),
                config.get("current_angle_status", ""),
                config.get("turn_direction", config.get("planned_turn_direction", "NONE")),
                config.get("turn_image_rotation_deg", ""),
                config.get("turn_vehicle_rotation_deg", ""),
                config.get("turn_progress_deg", ""),
                str(bool(config.get("turn_completed_by_vision", False))).lower(),
                map_col if map_col is not None else "",
                map_row if map_row is not None else "",
                map_cell_type,
                str(bool(is_obstacle)).lower(),
                primary_light_match.get("match_name", "NO_LIGHT"),
                primary_light_match.get("estimated_x_cell", ""),
                primary_light_match.get("estimated_y_cell", ""),
                primary_light_match.get("match_distance_cell", ""),
                primary_light_match.get("light_based_x_cell", ""),
                primary_light_match.get("light_based_y_cell", ""),
                primary_light_match.get("light_position_error_cell", ""),
                primary_light_match.get("validation_status", "NO_LIGHT"),
                config.get("motion_source", ""),
                config.get("selected_response", ""),
                config.get("light_track_status", ""),
                config.get("grid_track_status", ""),
                config.get("turn_target_x_cell", ""),
                config.get("turn_target_y_cell", ""),
                config.get("turn_cell_distance_cell", ""),
                config.get("turn_target_source", "")
            ])

    debug = draw_debug(
        frame_rgb,
        lights,
        lines_info,
        zone,
        grid_angle,
        dx_pixel,
        dy_pixel,
        dx_cell,
        dy_cell,
        response,
        pos_x_cell,
        pos_y_cell,
        config,
        map_col,
        map_row,
        map_cell_type,
        is_obstacle,
        primary_light_match
    )

    lines_only_debug = draw_lines_only_debug(frame_rgb, lines_info, dominant_angle)

    show_debug_windows(
        config,
        debug,
        gray,
        light_raw_mask,
        light_clean_mask,
        edges,
        lines_only_debug
    )

    if config.get("debug_save_frames", False):
        save_interval = int(config.get("debug_save_interval", 10))
        if frame_count % save_interval == 0:
            save_debug_outputs(
                frame_count,
                frame_rgb,
                gray,
                light_raw_mask,
                light_clean_mask,
                edges,
                debug,
                lines_only_debug
            )

    if return_debug:
        return gray_blur.copy(), pos_x_cell, pos_y_cell, debug, lines_only_debug

    return gray_blur.copy(), pos_x_cell, pos_y_cell

def resize_frame_if_needed(frame_rgb, config):
    if not config.get("resize_input", False):
        return frame_rgb

    target_width = int(config.get("resize_width", frame_rgb.shape[1]))
    h, w = frame_rgb.shape[:2]

    if w <= target_width:
        return frame_rgb

    scale = target_width / w
    target_height = int(h * scale)

    return cv2.resize(frame_rgb, (target_width, target_height))


def main():
    config = load_config()

    use_camera = config.get("use_camera", True)
    sample_hz = config["sample_hz"]
    sample_period = 1.0 / sample_hz

    picam2 = None
    test_frame_rgb = None

    if use_camera:
        picam2 = setup_camera(
            config["frame_width"],
            config["frame_height"],
	    config
        )
    else:
        test_frame_rgb = load_test_image(config)

    prev_gray = None

    initial_map_cell = config.get("initial_map_cell", [0.0, 0.0])
    pos_x_cell = float(initial_map_cell[0])
    pos_y_cell = float(initial_map_cell[1])

    frame_count = 0
    trajectory_path = init_trajectory_log(config)

    print("Ceiling locator started.")
    print(f"Mode: {'camera' if use_camera else 'test image'}")
    print("Press q to quit.")
    print("Press + / - to tune cell_per_pixel.")
    print("Press [ / ] to tune light_threshold.")
    print("Press r to reset position.")
    print("Press s to save config.")

    try:
        while True:
            start_time = time.time()

            if use_camera:
                frame_rgb = picam2.capture_array()
            else:
                frame_rgb = test_frame_rgb.copy()
            
            frame_rgb = resize_frame_if_needed(frame_rgb, config)

            prev_gray, pos_x_cell, pos_y_cell = process_frame(
                frame_rgb,
                config,
                prev_gray,
                pos_x_cell,
                pos_y_cell,
                frame_count,
                trajectory_path
            )

            wait_ms = 1 if use_camera or config.get("test_image_loop", True) else 0
            key = cv2.waitKey(wait_ms) & 0xFF

            if key == ord("q"):
                break

            elif key == ord("+") or key == ord("="):
                config["cell_per_pixel"] *= 1.05
                print(f"cell_per_pixel = {config['cell_per_pixel']:.7f}")

            elif key == ord("-") or key == ord("_"):
                config["cell_per_pixel"] /= 1.05
                print(f"cell_per_pixel = {config['cell_per_pixel']:.7f}")

            elif key == ord("["):
                config["light_threshold"] = max(0, config["light_threshold"] - 5)
                print(f"light_threshold = {config['light_threshold']}")

            elif key == ord("]"):
                config["light_threshold"] = min(255, config["light_threshold"] + 5)
                print(f"light_threshold = {config['light_threshold']}")

            elif key == ord("r"):
                initial_map_cell = config.get("initial_map_cell", [0.0, 0.0])
                pos_x_cell = float(initial_map_cell[0])
                pos_y_cell = float(initial_map_cell[1])
                prev_gray = None
                print(f"Position reset to initial_map_cell={initial_map_cell}.")

            elif key == ord("s"):
                save_config(config)
                print("Config saved.")

            frame_count += 1

            if not use_camera and not config.get("test_image_loop", True):
                print("Single test image processed. Press q in an OpenCV window to quit.")
                while True:
                    key = cv2.waitKey(0) & 0xFF
                    if key == ord("q"):
                        return

            elapsed = time.time() - start_time
            sleep_time = sample_period - elapsed

            if sleep_time > 0:
                time.sleep(sleep_time)

    finally:
        save_config(config)

        if picam2 is not None:
            picam2.stop()

        cv2.destroyAllWindows()
        print("Ceiling locator stopped.")
        print("Config saved.")


if __name__ == "__main__":
    main()
