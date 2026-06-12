
import cv2
import csv
import time
import math
from pathlib import Path
import numpy as np
import json
import sys

PROJECT_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT_BOOTSTRAP))

import navigation_router as nav_router

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAJECTORY_PATH = PROJECT_ROOT / "data" / "trajectory_log.csv"
CONFIG_PATH = Path(__file__).resolve().parent / "map_config.json"


def load_map_config():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


MAP_CONFIG = load_map_config()
MAP_CONFIG_MAP = MAP_CONFIG["map"]
MAP_WIDTH_CELLS = int(MAP_CONFIG_MAP.get("width_cells", 48))
MAP_HEIGHT_CELLS = int(MAP_CONFIG_MAP.get("height_cells", 22))
GRID_STEP_CELLS = float(MAP_CONFIG_MAP.get("grid_step_cells", 1))
UNIT_LABEL = MAP_CONFIG_MAP.get("unit_label", "cell")

CANVAS_WIDTH = int(MAP_CONFIG["canvas"]["width_px"])
CANVAS_HEIGHT = int(MAP_CONFIG["canvas"]["height_px"])

OCCUPANCY_CONFIG = MAP_CONFIG.get("occupancy_grid", {})
OCCUPANCY_ROWS = OCCUPANCY_CONFIG.get("rows", [])
OUTPUT_CONFIG = MAP_CONFIG.get("output", {})
VIEW_CONFIG = MAP_CONFIG.get("view", {})
TRAJECTORY_CONFIG = MAP_CONFIG.get("trajectory", {})
COORDINATE_CONFIG = MAP_CONFIG.get("coordinate_basis", {})
COLORS = MAP_CONFIG.get("colors", {})
NAVIGATION_CONFIG = MAP_CONFIG.get("navigation", {})
ROUTE_PATH = (PROJECT_ROOT / "data" / "route_path.json")
STATUS_PATH = (PROJECT_ROOT / "data" / "navigation_status.json")
if NAVIGATION_CONFIG.get("route_path"):
    p = Path(NAVIGATION_CONFIG.get("route_path"))
    ROUTE_PATH = (Path(__file__).resolve().parent / p).resolve() if not p.is_absolute() else p
if NAVIGATION_CONFIG.get("navigation_status_path"):
    p = Path(NAVIGATION_CONFIG.get("navigation_status_path"))
    STATUS_PATH = (Path(__file__).resolve().parent / p).resolve() if not p.is_absolute() else p
Y_AXIS_DOWN = COORDINATE_CONFIG.get("y_axis_down", True)
GRID_CELL_SIZE_CM = float(COORDINATE_CONFIG.get("grid_cell_size_cm", 60.0))


def color_bgr(name, default):
    rgb = COLORS.get(name, default)
    return (int(rgb[2]), int(rgb[1]), int(rgb[0]))


def get_cell_type(x_cell, y_cell):
    """Return obstacle/free/light/out_of_bounds for integer map cell coordinates."""
    x = int(x_cell)
    y = int(y_cell)
    if y < 0 or y >= len(OCCUPANCY_ROWS):
        return "out_of_bounds"
    if x < 0 or x >= len(OCCUPANCY_ROWS[y]):
        return "out_of_bounds"
    ch = OCCUPANCY_ROWS[y][x]
    if ch == "#":
        return "obstacle"
    if ch == "L":
        return "light"
    if ch == ".":
        return "free"
    return "unknown"


def _row_float(row, key, default=None):
    value = row.get(key, "")
    if value == "" or value is None:
        if default is None:
            raise ValueError(f"missing {key}")
        return default
    return float(value)


def load_trajectory():
    """
    Read trajectory_log.csv.

    Supports both:
      1. old 23-column logs
      2. new state-machine 32-column logs
      3. the common bad case where run_vision_map.py wrote the old header
         but ceiling_vision_locator.py appended the new 32-column rows.
    """
    points = []
    if not TRAJECTORY_PATH.exists():
        return points

    old_header = [
        "timestamp", "frame", "x_cell", "y_cell", "dx_pixel", "dy_pixel",
        "dx_cell", "dy_cell", "zone", "grid_angle", "phase_response",
        "map_col", "map_row", "map_cell_type", "is_obstacle",
        "detected_light_id", "detected_light_x_cell", "detected_light_y_cell",
        "detected_light_distance_cell", "light_based_x_cell",
        "light_based_y_cell", "light_position_error_cell",
        "light_validation_status",
    ]

    state_header = [
        "timestamp", "frame", "motion_state", "current_heading",
        "x_cell", "y_cell", "dx_pixel", "dy_pixel", "dx_cell", "dy_cell",
        "zone", "grid_angle", "phase_response", "yaw_error_deg",
        "angle_status", "turn_direction", "turn_image_rotation_deg",
        "turn_vehicle_rotation_deg", "turn_progress_deg",
        "turn_completed_by_vision", "map_col", "map_row", "map_cell_type",
        "is_obstacle", "detected_light_id", "detected_light_x_cell",
        "detected_light_y_cell", "detected_light_distance_cell",
        "light_based_x_cell", "light_based_y_cell",
        "light_position_error_cell", "light_validation_status",
        "motion_source", "selected_response", "light_track_status",
        "grid_track_status", "turn_target_x_cell", "turn_target_y_cell",
        "turn_cell_distance_cell", "turn_target_source",
    ]

    def to_float(value, default=None):
        if value == "" or value is None:
            return default
        try:
            return float(value)
        except Exception:
            return default

    def to_int(value, default=None):
        v = to_float(value, None)
        if v is None:
            return default
        return int(v)

    def row_to_dict(header, row):
        # Bad mixed-header case:
        # old header length=23, but row length=32 and row[2] is FORWARD/TURN/STOP.
        if len(row) >= 32 and len(header) < 32 and str(row[2]).upper() in ("FORWARD", "TURN", "STOP"):
            return dict(zip(state_header, row[:32]))

        if "motion_state" in header:
            # Normal state-machine log. Prefer the real header so extra columns
            # such as motion_source / track_status are preserved.
            return dict(zip(header, row[:len(header)]))

        # Normal old log.
        return dict(zip(header, row[:len(header)]))

    with open(TRAJECTORY_PATH, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, [])
        if not header:
            return points

        for row in reader:
            try:
                r = row_to_dict(header, row)

                x_cell = to_float(r.get("x_cell", ""))
                y_cell = to_float(r.get("y_cell", ""))

                if x_cell is None or y_cell is None:
                    x_cm = to_float(r.get("x_cm", ""))
                    y_cm = to_float(r.get("y_cm", ""))
                    if x_cm is None or y_cm is None:
                        continue
                    x_cell = x_cm / GRID_CELL_SIZE_CM
                    y_cell = y_cm / GRID_CELL_SIZE_CM

                grid_angle = to_float(r.get("grid_angle", ""), None)
                map_col = to_int(r.get("map_col", ""), None)
                map_row = to_int(r.get("map_row", ""), None)
                cell_type = r.get("map_cell_type", "")
                if cell_type == "":
                    cell_type = get_cell_type(round(x_cell), round(y_cell))

                points.append({
                    "timestamp": to_float(r.get("timestamp", ""), 0.0),
                    "frame": to_int(r.get("frame", ""), None),
                    "motion_state": r.get("motion_state", ""),
                    "current_heading": r.get("current_heading", ""),
                    "x_cell": x_cell,
                    "y_cell": y_cell,
                    "dx_pixel": to_float(r.get("dx_pixel", ""), 0.0),
                    "dy_pixel": to_float(r.get("dy_pixel", ""), 0.0),
                    "dx_cell": to_float(r.get("dx_cell", ""), 0.0),
                    "dy_cell": to_float(r.get("dy_cell", ""), 0.0),
                    "zone": r.get("zone", ""),
                    "grid_angle": grid_angle,
                    "yaw_error_deg": to_float(r.get("yaw_error_deg", ""), None),
                    "angle_status": r.get("angle_status", ""),
                    "turn_direction": r.get("turn_direction", ""),
                    "turn_progress_deg": to_float(r.get("turn_progress_deg", ""), None),
                    "turn_completed_by_vision": str(r.get("turn_completed_by_vision", "false")).lower() == "true",
                    "response": to_float(r.get("phase_response", ""), 0.0),
                    "map_col": map_col,
                    "map_row": map_row,
                    "map_cell_type": cell_type,
                    "is_obstacle": str(r.get("is_obstacle", "false")).lower() == "true",
                    "detected_light_id": r.get("detected_light_id", ""),
                    "detected_light_x_cell": to_float(r.get("detected_light_x_cell", ""), None),
                    "detected_light_y_cell": to_float(r.get("detected_light_y_cell", ""), None),
                    "detected_light_distance_cell": to_float(r.get("detected_light_distance_cell", ""), None),
                    "light_based_x_cell": to_float(r.get("light_based_x_cell", ""), None),
                    "light_based_y_cell": to_float(r.get("light_based_y_cell", ""), None),
                    "light_position_error_cell": to_float(r.get("light_position_error_cell", ""), None),
                    "light_validation_status": r.get("light_validation_status", ""),
                    "motion_source": r.get("motion_source", ""),
                    "selected_response": to_float(r.get("selected_response", ""), None),
                    "light_track_status": r.get("light_track_status", ""),
                    "grid_track_status": r.get("grid_track_status", ""),
                    "turn_target_x_cell": to_float(r.get("turn_target_x_cell", ""), None),
                    "turn_target_y_cell": to_float(r.get("turn_target_y_cell", ""), None),
                    "turn_cell_distance_cell": to_float(r.get("turn_cell_distance_cell", ""), None),
                    "turn_target_source": r.get("turn_target_source", ""),
                })
            except Exception:
                continue

    return points



def normalize_points_to_first_point(points):
    # Occupancy map uses absolute map cell coordinates, so do not normalize by default.
    return points


def compute_view(points):
    margin_px = int(VIEW_CONFIG.get("margin_px", 45))
    auto_zoom = VIEW_CONFIG.get("auto_zoom", False)

    if not auto_zoom or not points:
        view = {
            "min_x": -0.5,
            "max_x": MAP_WIDTH_CELLS - 0.5,
            "min_y": -0.5,
            "max_y": MAP_HEIGHT_CELLS - 0.5,
            "mode": "fixed_occupancy_map",
        }
    else:
        xs = [p["x_cell"] for p in points]
        ys = [p["y_cell"] for p in points]
        padding = float(VIEW_CONFIG.get("padding_cells", 0.5))
        min_x = min(xs) - padding
        max_x = max(xs) + padding
        min_y = min(ys) - padding
        max_y = max(ys) + padding
        min_view_w = float(VIEW_CONFIG.get("min_view_width_cells", 4.0))
        min_view_h = float(VIEW_CONFIG.get("min_view_height_cells", 3.0))
        cx = (min_x + max_x) / 2
        cy = (min_y + max_y) / 2
        width = max(max_x - min_x, min_view_w)
        height = max(max_y - min_y, min_view_h)
        view = {
            "min_x": max(-0.5, cx - width / 2),
            "max_x": min(MAP_WIDTH_CELLS - 0.5, cx + width / 2),
            "min_y": max(-0.5, cy - height / 2),
            "max_y": min(MAP_HEIGHT_CELLS - 0.5, cy + height / 2),
            "mode": "auto_zoom",
        }

    view_w = max(view["max_x"] - view["min_x"], 1e-6)
    view_h = max(view["max_y"] - view["min_y"], 1e-6)
    usable_w = max(CANVAS_WIDTH - 2 * margin_px, 1)
    usable_h = max(CANVAS_HEIGHT - 2 * margin_px, 1)
    view["scale"] = min(usable_w / view_w, usable_h / view_h)
    view["center_x"] = (view["min_x"] + view["max_x"]) / 2
    view["center_y"] = (view["min_y"] + view["max_y"]) / 2
    view["margin_px"] = margin_px
    return view


def cell_to_canvas(x_cell, y_cell, view=None):
    if view is None:
        view = compute_view([])
    px = int(CANVAS_WIDTH / 2 + (x_cell - view["center_x"]) * view["scale"])
    if Y_AXIS_DOWN:
        py = int(CANVAS_HEIGHT / 2 + (y_cell - view["center_y"]) * view["scale"])
    else:
        py = int(CANVAS_HEIGHT / 2 - (y_cell - view["center_y"]) * view["scale"])
    return px, py


def draw_occupancy_grid(canvas, view):
    free_color = color_bgr("free", [205, 205, 205])
    obs_color = color_bgr("obstacle", [0, 0, 0])
    light_color = color_bgr("light", [255, 245, 195])
    unknown_color = color_bgr("unknown", [240, 240, 240])
    grid_color = color_bgr("grid_line", [170, 170, 170])

    for y, row in enumerate(OCCUPANCY_ROWS):
        for x, ch in enumerate(row):
            if ch == "#":
                color = obs_color
            elif ch == "L":
                color = light_color
            elif ch == ".":
                color = free_color
            else:
                color = unknown_color

            x1, y1 = cell_to_canvas(x - 0.5, y - 0.5, view)
            x2, y2 = cell_to_canvas(x + 0.5, y + 0.5, view)
            left, right = min(x1, x2), max(x1, x2)
            top, bottom = min(y1, y2), max(y1, y2)
            cv2.rectangle(canvas, (left, top), (right, bottom), color, -1)
            cv2.rectangle(canvas, (left, top), (right, bottom), grid_color, 1)


def draw_axes_and_labels(canvas, view):
    text_color = (80, 80, 80)
    # Labels on top and left: cell indices.
    for x in range(MAP_WIDTH_CELLS):
        px, py = cell_to_canvas(x, -0.85, view)
        if 0 <= px < CANVAS_WIDTH:
            cv2.putText(canvas, str(x), (px - 7, max(15, py)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1)
    for y in range(MAP_HEIGHT_CELLS):
        px, py = cell_to_canvas(-0.85, y, view)
        if 0 <= py < CANVAS_HEIGHT:
            cv2.putText(canvas, str(y), (max(2, px - 8), py + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1)


def load_route():
    try:
        return nav_router.load_route(ROUTE_PATH)
    except Exception:
        return None


def load_navigation_status():
    if not STATUS_PATH.exists():
        return {}
    try:
        with open(STATUS_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def draw_route(canvas, route, status, view):
    if not route or not NAVIGATION_CONFIG.get("draw_route", True):
        return

    path = route.get("path", [])
    if len(path) >= 2:
        pts = [cell_to_canvas(float(p[0]), float(p[1]), view) for p in path]
        route_color = color_bgr("route", [80, 140, 255])
        for i in range(1, len(pts)):
            cv2.line(canvas, pts[i - 1], pts[i], route_color, 2)

    # Start and goal markers.
    if route.get("start_cell"):
        sx, sy = cell_to_canvas(route["start_cell"][0], route["start_cell"][1], view)
        cv2.circle(canvas, (sx, sy), 10, (0, 180, 0), 2)
        cv2.putText(canvas, "ROUTE START", (sx + 8, sy + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 120, 0), 2)

    if route.get("goal_cell"):
        gx, gy = cell_to_canvas(route["goal_cell"][0], route["goal_cell"][1], view)
        goal_color = color_bgr("route_goal", [80, 80, 255])
        cv2.circle(canvas, (gx, gy), 11, goal_color, 3)
        cv2.putText(canvas, "GOAL", (gx + 8, gy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, goal_color, 2)

    if NAVIGATION_CONFIG.get("draw_turn_points", True):
        turn_color = color_bgr("route_turn", [255, 120, 80])
        for t in route.get("turn_points", []):
            cell = t.get("cell", [0, 0])
            tx, ty = cell_to_canvas(cell[0], cell[1], view)
            cv2.circle(canvas, (tx, ty), 9, turn_color, 2)
            label = f"TURN {t.get('turn', '')}"
            cv2.putText(canvas, label, (tx + 8, ty + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, turn_color, 2)

    # Current navigation target.
    target = status.get("next_target_cell")
    if target:
        tx, ty = cell_to_canvas(target[0], target[1], view)
        cv2.drawMarker(canvas, (tx, ty), (255, 0, 255),
                       markerType=cv2.MARKER_CROSS, markerSize=18, thickness=2)
        cv2.putText(canvas, "NEXT TARGET", (tx + 8, ty - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 0, 180), 2)


def draw_navigation_overlay(canvas, route, status):
    if not route:
        cv2.putText(canvas, "Navigation: no route_path.json yet", (20, CANVAS_HEIGHT - 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 180), 2)
        return

    lines = [
        f"Route: start={route.get('start_cell')} goal={route.get('goal_cell')} initial_heading={route.get('initial_heading')}",
        f"Nav: seg={status.get('current_segment_index', 'N/A')}/{max(len(route.get('segments', []))-1, 0)}, "
        f"heading={status.get('current_heading', 'N/A')} -> {status.get('target_heading', 'N/A')}, "
        f"turn_flag={status.get('turn_flag', False)} {status.get('turn_direction', '')}",
        f"Next target={status.get('next_target_cell', 'N/A')}, dist={status.get('distance_to_next_target_cell', 'N/A')}, done={status.get('route_done', False)}",
    ]

    y = CANVAS_HEIGHT - 78
    for text in lines:
        cv2.putText(canvas, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
        y += 24

def draw_trajectory(canvas, points, view):
    if len(points) == 0:
        cv2.putText(canvas, "Waiting for trajectory_log.csv...", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return

    canvas_points = [cell_to_canvas(p["x_cell"], p["y_cell"], view) for p in points]
    line_thickness = int(TRAJECTORY_CONFIG.get("line_thickness", 3))
    point_radius = int(TRAJECTORY_CONFIG.get("point_radius", 3))
    draw_points = TRAJECTORY_CONFIG.get("draw_points", True)

    for i in range(1, len(canvas_points)):
        cv2.line(canvas, canvas_points[i - 1], canvas_points[i], (0, 120, 255), line_thickness)

    if draw_points:
        step = max(1, len(canvas_points) // 200)
        for pxy in canvas_points[::step]:
            cv2.circle(canvas, pxy, point_radius, (0, 100, 255), -1)

    start = canvas_points[0]
    cv2.circle(canvas, start, 7, (0, 200, 0), -1)
    cv2.putText(canvas, "START", (start[0] + 8, start[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 0), 2)

    current = canvas_points[-1]
    last = points[-1]
    current_color = (0, 0, 255) if not last.get("is_obstacle", False) else (0, 0, 180)
    cv2.circle(canvas, current, 9, current_color, -1)

    # Router/controller-provided turn coordinate reference.
    if last.get("turn_target_x_cell") is not None and last.get("turn_target_y_cell") is not None:
        tx, ty = cell_to_canvas(last["turn_target_x_cell"], last["turn_target_y_cell"], view)
        cv2.drawMarker(canvas, (tx, ty), (255, 0, 255),
                       markerType=cv2.MARKER_TILTED_CROSS, markerSize=22, thickness=2)
        cv2.putText(canvas, "TURN CELL", (tx + 8, ty + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 0, 180), 2)

    # Independent robot position inferred from the matched ceiling light.
    if last.get("light_based_x_cell") is not None and last.get("light_based_y_cell") is not None:
        rx, ry = cell_to_canvas(last["light_based_x_cell"], last["light_based_y_cell"], view)
        cv2.circle(canvas, (rx, ry), 8, (255, 255, 0), 2)
        cv2.line(canvas, current, (rx, ry), (255, 180, 0), 2)
        cv2.putText(canvas, "light-based robot pos", (rx + 8, ry + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140, 100, 0), 2)

    # Estimated position of the visible light, if provided by the locator.
    if last.get("detected_light_x_cell") is not None and last.get("detected_light_y_cell") is not None:
        lx, ly = cell_to_canvas(last["detected_light_x_cell"], last["detected_light_y_cell"], view)
        cv2.circle(canvas, (lx, ly), 8, (0, 255, 255), 2)
        label = last.get("detected_light_id", "") or "detected light"
        cv2.putText(canvas, label, (lx + 8, ly - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 120, 120), 2)

    grid_angle = last["grid_angle"]
    if grid_angle is not None:
        angle_rad = math.radians(grid_angle)
        heading_len = 45
        x2 = int(current[0] + heading_len * math.sin(angle_rad))
        y2 = int(current[1] - heading_len * math.cos(angle_rad))
        cv2.arrowedLine(canvas, current, (x2, y2), (255, 0, 0), 3, tipLength=0.3)

    xs = [p["x_cell"] for p in points]
    ys = [p["y_cell"] for p in points]
    range_x = max(xs) - min(xs) if xs else 0
    range_y = max(ys) - min(ys) if ys else 0

    map_col = last.get("map_col")
    map_row = last.get("map_row")
    cell_type = last.get("map_cell_type", get_cell_type(round(last["x_cell"]), round(last["y_cell"])))

    info = [
        f"Current grid position: x={last['x_cell']:.3f}, y={last['y_cell']:.3f} cell",
        f"State: {last.get('motion_state', 'N/A') or 'N/A'}, heading={last.get('current_heading', 'N/A') or 'N/A'}, yaw_err={last.get('yaw_error_deg') if last.get('yaw_error_deg') is not None else 'N/A'}",
        f"Turn: dir={last.get('turn_direction', 'N/A') or 'N/A'}, progress={last.get('turn_progress_deg') if last.get('turn_progress_deg') is not None else 'N/A'}, done={last.get('turn_completed_by_vision', False)}",
        f"Turn cell: x={last.get('turn_target_x_cell') if last.get('turn_target_x_cell') is not None else 'N/A'}, y={last.get('turn_target_y_cell') if last.get('turn_target_y_cell') is not None else 'N/A'}, dist={last.get('turn_cell_distance_cell') if last.get('turn_cell_distance_cell') is not None else 'N/A'}",
        f"Map cell: col={map_col}, row={map_row}, type={cell_type}",
        f"Detected light: {last.get('detected_light_id', '') or 'NO_LIGHT'}",
        f"Light match dist: {last.get('detected_light_distance_cell') if last.get('detected_light_distance_cell') is not None else 'N/A'} cell",
        f"Light validation: {last.get('light_validation_status', '') or 'N/A'}, error={last.get('light_position_error_cell') if last.get('light_position_error_cell') is not None else 'N/A'} cell",
        f"Light-based robot: x={last.get('light_based_x_cell') if last.get('light_based_x_cell') is not None else 'N/A'}, y={last.get('light_based_y_cell') if last.get('light_based_y_cell') is not None else 'N/A'}",
        f"Zone: {last['zone']}",
        f"Grid angle: {grid_angle:.2f} deg" if grid_angle is not None else "Grid angle: N/A",
        f"Motion source: {last.get('motion_source', '') or 'N/A'}, selected_resp={last.get('selected_response') if last.get('selected_response') is not None else 'N/A'}",
        f"Track status: light={last.get('light_track_status', '') or 'N/A'}, grid={last.get('grid_track_status', '') or 'N/A'}",
        f"Phase response: {last['response']:.3f}",
        f"Points: {len(points)}",
        f"Trajectory range: dx={range_x:.3f}, dy={range_y:.3f} cell",
        "Legend: black=obstacle, gray=free, yellow=light",
    ]

    y0 = 28
    for text in info:
        cv2.putText(canvas, text, (20, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 2)
        y0 += 24

    if cell_type == "obstacle":
        cv2.putText(canvas, "WARNING: robot is in / predicted to obstacle cell", (20, y0 + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)


def render_map(raw_points=None):
    if raw_points is None:
        raw_points = load_trajectory()
    points = normalize_points_to_first_point(raw_points)
    view = compute_view(points)

    canvas = np.ones((CANVAS_HEIGHT, CANVAS_WIDTH, 3), dtype=np.uint8) * 255
    draw_occupancy_grid(canvas, view)
    draw_axes_and_labels(canvas, view)

    route = load_route()
    status = load_navigation_status()
    draw_route(canvas, route, status, view)

    draw_trajectory(canvas, points, view)
    draw_navigation_overlay(canvas, route, status)
    return canvas


def save_map(canvas, output_dir, frame_count):
    output_filename = OUTPUT_CONFIG.get("output_filename", "latest_map.jpg")
    save_sequence = OUTPUT_CONFIG.get("save_sequence", True)
    latest_path = output_dir / output_filename
    cv2.imwrite(str(latest_path), canvas)
    if save_sequence:
        seq_path = output_dir / f"map_{frame_count:05d}.jpg"
        cv2.imwrite(str(seq_path), canvas)
    return latest_path


def main():
    print("Map visualizer started.")
    print(f"Reading: {TRAJECTORY_PATH}")
    print("Coordinate unit: 1 ceiling grid cell")
    print("Map legend: black=obstacle, gray=free, yellow=light")

    headless = OUTPUT_CONFIG.get("headless", True)
    display_window = OUTPUT_CONFIG.get("display_window", False)
    save_map_image = OUTPUT_CONFIG.get("save_map_image", True)
    save_interval_frames = int(OUTPUT_CONFIG.get("save_interval_frames", 5))
    loop_sleep_sec = float(OUTPUT_CONFIG.get("loop_sleep_sec", 0.5))
    output_dir = Path(__file__).resolve().parent / OUTPUT_CONFIG.get("output_dir", "map_output")
    output_dir.mkdir(exist_ok=True)

    if headless or not display_window:
        print("Headless mode: saving map images instead of showing window.")
        print(f"Output dir: {output_dir}")
        print("Press Ctrl+C to quit.")
    else:
        print("Display mode: press q in the OpenCV window to quit.")

    frame_count = 0
    last_points_count = -1
    try:
        while True:
            raw_points = load_trajectory()
            if len(raw_points) == last_points_count:
                time.sleep(loop_sleep_sec)
                continue
            last_points_count = len(raw_points)
            canvas = render_map(raw_points)

            if save_map_image and frame_count % save_interval_frames == 0:
                latest_path = save_map(canvas, output_dir, frame_count)
                print(f"Saved map: {latest_path}, points={len(raw_points)}")

            if not headless and display_window:
                cv2.imshow("Robot Map Visualizer", canvas)
                key = cv2.waitKey(200) & 0xFF
                if key == ord("q"):
                    break
            else:
                time.sleep(loop_sleep_sec)
            frame_count += 1

    except KeyboardInterrupt:
        print("Map visualizer stopped.")
    finally:
        if not headless and display_window:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
