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
    config_to_save.pop("last_dominant_angle", None)
    with open(path, "w") as f:
        json.dump(config_to_save, f, indent=2)


def setup_camera(width, height, config):
    picam2 = Picamera2()

    camera_config = picam2.create_video_configuration(
        main={"size": (width, height), "format": "RGB888"}
    )

    picam2.configure(camera_config)
    picam2.start()
    time.sleep(2)  # camera warm up

    if not config.get("camera_auto_exposure", True):
        picam2.set_controls({
            "AeEnable": False,
            "ExposureTime": int(config.get("camera_exposure_time", 20000)),
            "AnalogueGain": float(config.get("camera_analogue_gain", 4.0))
        })
    else:
        picam2.set_controls({
            "AeEnable": True
        })

    time.sleep(1)

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

def detect_grid_lines(frame_rgb, config):
    """
    偵測天花板格線。
    先用 HoughLinesP 找所有線段，再估計 dominant_angle。
    保留 dominant_angle 和 dominant_angle + 90 度兩組主要格線方向。
    
    回傳：
        lines_info: 篩選後的格線資訊
        edges: Canny edge image
        dominant_angle: 目前畫面中的主要格線角度
    """
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)

    # 可選：CLAHE 局部對比增強
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
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "angle": angle,
            "length": length,
            "type": "raw"
        })

    dominant_angle = estimate_dominant_grid_angle(
    raw_lines_info,
    config,
    prior_angle=config.get("last_dominant_angle", None)
)

    if dominant_angle is None:
        return [], edges, None
    
    config["last_dominant_angle"] = dominant_angle

    angle_tolerance = config.get("line_angle_tolerance", 12)

    direction_a = dominant_angle
    direction_b = normalize_angle_90(dominant_angle + 90)

    lines_info = []

    for line in raw_lines_info:
        angle = line["angle"]

        dist_a = angle_distance_180(angle, direction_a)
        dist_b = angle_distance_180(angle, direction_b)

        if dist_a < angle_tolerance:
            line["type"] = "grid_a"
            lines_info.append(line)
        elif dist_b < angle_tolerance:
            line["type"] = "grid_b"
            lines_info.append(line)
        else:
            # 不屬於兩組主要格線方向的線，視為雜訊
            continue

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


def pixel_shift_to_grid_cell_shift(dx_pixel, dy_pixel, grid_angle, config):
    """
    Convert image displacement into the ceiling-grid coordinate basis.

    Coordinate definition:
        +x_cell: along the detected ceiling grid direction A
        +y_cell: along the detected ceiling grid direction B, perpendicular to A
        1.0 unit = one ceiling grid cell

    dx_pixel/dy_pixel are measured in image coordinates:
        +x = right, +y = down.
    grid_angle is the angle of grid direction A in the image.
    """
    if grid_angle is None:
        grid_angle = float(config.get("fallback_grid_angle_deg", 0.0))

    angle = math.radians(float(grid_angle) + float(config.get("grid_angle_offset_deg", 0.0)))

    # Grid basis vectors in image coordinates.
    ax = math.cos(angle)
    ay = math.sin(angle)
    bx = -math.sin(angle)
    by = math.cos(angle)

    # Project image displacement onto the two ceiling-grid axes.
    dx_cell = (dx_pixel * ax + dy_pixel * ay) * float(config.get("cell_per_pixel", 0.0025))
    dy_cell = (dx_pixel * bx + dy_pixel * by) * float(config.get("cell_per_pixel", 0.0025))

    if config.get("invert_grid_x", False):
        dx_cell = -dx_cell
    if config.get("invert_grid_y", False):
        dy_cell = -dy_cell

    return dx_cell, dy_cell




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
    into the ceiling-grid coordinate basis.

    This uses the same grid-angle basis as pixel_shift_to_grid_cell_shift(),
    but has separate sign controls because optical image displacement and
    vehicle-motion displacement may need opposite signs in practice.
    """
    if grid_angle is None:
        grid_angle = float(config.get("fallback_grid_angle_deg", 0.0))

    angle = math.radians(float(grid_angle) + float(config.get("grid_angle_offset_deg", 0.0)))

    ax = math.cos(angle)
    ay = math.sin(angle)
    bx = -math.sin(angle)
    by = math.cos(angle)

    vx_cell = (dx_pixel * ax + dy_pixel * ay) * float(config.get("cell_per_pixel", 0.0025))
    vy_cell = (dx_pixel * bx + dy_pixel * by) * float(config.get("cell_per_pixel", 0.0025))

    # Separate from invert_grid_x/y so you can tune light landmark matching
    # without changing motion integration.
    vx_cell *= float(config.get("light_position_sign_x", 1.0))
    vy_cell *= float(config.get("light_position_sign_y", 1.0))

    if config.get("invert_grid_x", False):
        vx_cell = -vx_cell
    if config.get("invert_grid_y", False):
        vy_cell = -vy_cell

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
        cv2.putText(
            debug,
            f"Light {i}",
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
        f"dx, dy pixel: {dx_pixel:.2f}, {dy_pixel:.2f}",
        f"dx, dy cell: {dx_cell:.4f}, {dy_cell:.4f}",
        f"phase response: {response:.3f}",
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
                "x_cell",
                "y_cell",
                "dx_pixel",
                "dy_pixel",
                "dx_cell",
                "dy_cell",
                "zone",
                "grid_angle",
                "phase_response",
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
                "light_validation_status"
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

    dx_pixel, dy_pixel, response = estimate_pixel_shift(prev_gray, gray_blur)
    dx_cell, dy_cell = pixel_shift_to_grid_cell_shift(dx_pixel, dy_pixel, grid_angle, config)

    proposed_x_cell = pos_x_cell + dx_cell
    proposed_y_cell = pos_y_cell + dy_cell
    map_col, map_row, map_cell_type, is_obstacle = check_obstacle_at_position(
        proposed_x_cell, proposed_y_cell, config
    )

    # Identify which configured ceiling light is visible in the current image.
    light_matches = identify_lights_on_map(
        lights,
        frame_rgb.shape,
        proposed_x_cell,
        proposed_y_cell,
        grid_angle,
        config
    )
    primary_light_match = select_primary_light_match(light_matches)

    # If response is too low, the shift estimate is unreliable.
    # In that case, log the measurement but do not update the accumulated position.
    # If proposed cell is an obstacle, either warn-only or block the update according to config.
    if response > float(config.get("phase_response_min", 0.15)):
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

    # Optional landmark correction: use the light-inferred robot position to
    # gently pull the grid/phase-correlation position back toward an absolute
    # map landmark. Disabled by default; enable after signs/scale are calibrated.
    if (config.get("light_position_correction_enabled", False)
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

    if config.get("log_trajectory", True):
        with open(trajectory_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                time.time(),
                frame_count,
                pos_x_cell,
                pos_y_cell,
                dx_pixel,
                dy_pixel,
                dx_cell,
                dy_cell,
                zone,
                grid_angle if grid_angle is not None else "",
                response,
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
                primary_light_match.get("validation_status", "NO_LIGHT")
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
