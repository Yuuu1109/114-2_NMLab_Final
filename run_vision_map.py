"""
Integrated runner for the RPi car ceiling vision + occupancy-grid map output.

Place this file at the project root:
    ~/Final/run_vision_map.py

Expected project layout:
    Final/
    ├── run_vision_map.py
    ├── vision/
    │   ├── ceiling_vision_locator.py
    │   └── vision_config.json
    ├── visualization/
    │   ├── map_visualizer.py
    │   └── map_config.json
    └── data/
        └── trajectory_log.csv

This version matches the obstacle-map / ceiling-grid-cell files:
    - ceiling_vision_locator.py writes x_cell, y_cell, dx_cell, dy_cell
    - map_visualizer.py draws occupancy_grid from map_config.json
    - map_config.json uses x=column, y=row, 1 unit=one ceiling grid cell
    - vision_config.json defines initial_map_cell and obstacle checking

Loop:
    capture/process one frame
    -> append trajectory_log.csv with map_col/map_row/map_cell_type/is_obstacle
    -> every N frames, render and save/display the occupancy map
    -> optionally show the latest Ceiling Locator final_debug frame

Headless mode:
    Saves map images only. Works over SSH without display.

Display mode:
    Set visualization/map_config.json:
        "headless": false,
        "display_window": true
    or pass --force-display.
    Requires HDMI desktop, VNC, or SSH X11 forwarding.
    Press q in an OpenCV window to stop.

Stop headless mode with Ctrl+C.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

import json


PROJECT_ROOT = Path(__file__).resolve().parent
VISION_DIR = PROJECT_ROOT / "vision"
VISUALIZATION_DIR = PROJECT_ROOT / "visualization"
DATA_DIR = PROJECT_ROOT / "data"

# Allow importing modules from project subfolders.
sys.path.insert(0, str(VISION_DIR))
sys.path.insert(0, str(VISUALIZATION_DIR))

import ceiling_vision_locator as vision  # noqa: E402
import map_visualizer as mapper  # noqa: E402
import navigation_router as nav_router  # noqa: E402


TRAJECTORY_HEADER = [
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
    "turn_target_source",
]


def write_empty_trajectory_log(path: Path) -> None:
    """Create a fresh trajectory CSV with the obstacle-map / grid-cell / light-validation header."""
    path.parent.mkdir(exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(TRAJECTORY_HEADER)


def ensure_grid_cell_trajectory_log(path: Path, reset_log: bool = False) -> Path:
    """
    Ensure trajectory_log.csv uses the new x_cell/y_cell + obstacle columns.

    If reset_log=True, overwrite it.
    If an old x_cm/y_cm header is found, back it up and create a new file.
    """
    path.parent.mkdir(exist_ok=True)

    if reset_log or not path.exists():
        write_empty_trajectory_log(path)
        print(f"Reset trajectory log: {path}")
        return path

    try:
        with open(path, "r", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, [])
    except Exception:
        header = []

    required = {"motion_state", "current_heading", "x_cell", "y_cell", "dx_cell", "dy_cell", "yaw_error_deg", "angle_status", "turn_direction", "turn_completed_by_vision", "map_col", "map_row", "map_cell_type", "is_obstacle", "detected_light_id", "light_based_x_cell", "light_validation_status", "turn_target_x_cell", "turn_target_y_cell", "turn_cell_distance_cell", "turn_target_source"}
    if not required.issubset(set(header)):
        backup = path.with_name(f"{path.stem}_old_{int(time.time())}{path.suffix}")
        shutil.move(str(path), str(backup))
        write_empty_trajectory_log(path)
        print(f"Old trajectory format detected. Backed up to: {backup}")
        print(f"Created new trajectory log: {path}")

    return path


def apply_gamma_correction(frame_rgb: np.ndarray, gamma: float) -> np.ndarray:
    """
    Optional brightness correction for dark RPi camera frames.
    gamma < 1 brightens dark regions; gamma > 1 darkens.
    """
    gamma = max(float(gamma), 0.01)
    table = np.array([
        ((i / 255.0) ** gamma) * 255 for i in range(256)
    ]).astype("uint8")
    return cv2.LUT(frame_rgb, table)


def render_and_save_map(map_frame_count: int) -> tuple[np.ndarray, Path | None]:
    """Render the current trajectory CSV to an occupancy map image and optionally save it."""
    canvas = mapper.render_map()

    output_config = mapper.OUTPUT_CONFIG
    save_map_image = bool(output_config.get("save_map_image", True))

    latest_path = None
    if save_map_image:
        output_dir = VISUALIZATION_DIR / output_config.get("output_dir", "map_output")
        output_dir.mkdir(exist_ok=True)
        latest_path = mapper.save_map(canvas, output_dir, map_frame_count)

    return canvas, latest_path



def _route_segment_target_cell(route: dict | None, segment_index: int):
    """Return the current segment target cell from route_path.json, if available."""
    if not route:
        return None
    segments = route.get("segments", [])
    if not segments:
        return None
    idx = max(0, min(int(segment_index), len(segments) - 1))
    target = segments[idx].get("to")
    if isinstance(target, (list, tuple)) and len(target) == 2:
        return [float(target[0]), float(target[1])]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run ceiling-grid localization and occupancy-map rendering in one synchronized loop."
    )
    parser.add_argument(
        "--reset-log",
        action="store_true",
        help="Delete old trajectory data and start a new trajectory_log.csv.",
    )
    parser.add_argument(
        "--reset-map-output",
        action="store_true",
        help="Delete old visualization/map_output images before running.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional max frame count for testing. Omit for continuous run.",
    )
    parser.add_argument(
        "--map-every",
        type=int,
        default=None,
        help="Render one map every N vision frames. Defaults to map_config output.save_interval_frames.",
    )
    parser.add_argument(
        "--force-headless",
        action="store_true",
        help="Override map_config and disable all OpenCV display windows.",
    )
    parser.add_argument(
        "--force-display",
        action="store_true",
        help="Override map_config and enable map/final-debug OpenCV display windows. Requires a display.",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Override navigation start cell as x,y. Example: --start 31,21",
    )
    parser.add_argument(
        "--goal",
        type=str,
        default=None,
        help="Override navigation goal cell as x,y. Example: --goal 43,14",
    )
    parser.add_argument(
        "--heading",
        type=str,
        default=None,
        choices=["UP", "DOWN", "LEFT", "RIGHT"],
        help="Optional manual override. Usually omit this; heading is inferred from the first route segment.",
    )
    parser.add_argument(
        "--no-navigation",
        action="store_true",
        help="Disable route planning and navigation_status.json output.",
    )
    args = parser.parse_args()

    config = vision.load_config()

    map_config = mapper.MAP_CONFIG
    nav_config = map_config.get("navigation", {})
    navigation_enabled = bool(nav_config.get("enabled", True)) and not args.no_navigation
    route = None
    navigation_status_path = nav_router.DEFAULT_STATUS_PATH
    current_segment_index = 0
    # Usually AUTO: do not force an initial heading from the command line.
    # The router will infer it from the first planned route segment.
    requested_heading = args.heading if args.heading is not None else nav_config.get("initial_heading", "AUTO")
    logical_heading = "AUTO"

    if navigation_enabled:
        try:
            start_cell = nav_router.parse_cell(args.start, "start") if args.start else nav_router.parse_cell(nav_config.get("start_cell", config.get("initial_map_cell", [0, 0])), "start_cell")
            goal_cell = nav_router.parse_cell(args.goal, "goal") if args.goal else nav_router.parse_cell(nav_config.get("goal_cell", [0, 0]), "goal_cell")
            route = nav_router.plan_route(
                map_config,
                start=start_cell,
                goal=goal_cell,
                initial_heading=requested_heading,
            )
            route_path = nav_config.get("route_path", "../data/route_path.json")
            route_path = Path(route_path)
            if not route_path.is_absolute():
                route_path = (VISUALIZATION_DIR / route_path).resolve()
            nav_router.save_route(route, route_path)

            status_path = nav_config.get("navigation_status_path", "../data/navigation_status.json")
            status_path = Path(status_path)
            if not status_path.is_absolute():
                status_path = (VISUALIZATION_DIR / status_path).resolve()
            navigation_status_path = status_path

            # Route start is also the localization baseline.
            config["initial_map_cell"] = [float(start_cell[0]), float(start_cell[1])]
            # After planning, use the first route segment as the logical initial heading.
            logical_heading = nav_router.normalize_heading(route.get("initial_heading", "UP"))
            print(f"Navigation route planned: {route_path}")
            print(f"  start={route['start_cell']} goal={route['goal_cell']} heading={route['initial_heading']} ({route.get('initial_heading_source', 'unknown')})")
            print(f"  path cells={len(route['path'])}, segments={len(route['segments'])}, turns={len(route['turn_points'])}")
        except Exception as exc:
            navigation_enabled = False
            print(f"Navigation disabled: failed to plan route: {exc}")


    # Map rendering depends on the CSV written by vision.process_frame().
    config["log_trajectory"] = True

    output_config = mapper.OUTPUT_CONFIG
    headless = bool(output_config.get("headless", True))
    display_window = bool(output_config.get("display_window", False))

    if args.force_headless:
        headless = True
        display_window = False
        config["show_debug"] = False

    if args.force_display:
        headless = False
        display_window = True

    # Avoid Qt/xcb errors when running in SSH/headless mode.
    if headless:
        config["show_debug"] = False

    map_display_enabled = (not headless) and display_window
    vision_display_enabled = bool(config.get("show_debug", False))
    final_debug_display_enabled = (not headless) and bool(config.get("display_final_debug_window", False))
    final_debug_every = max(1, int(config.get("display_final_debug_every", 1)))
    any_display_enabled = map_display_enabled or vision_display_enabled or final_debug_display_enabled

    sample_hz = float(config.get("sample_hz", 5))
    sample_period = 1.0 / sample_hz if sample_hz > 0 else 0.0
    use_camera = bool(config.get("use_camera", True))

    trajectory_path = ensure_grid_cell_trajectory_log(
        DATA_DIR / "trajectory_log.csv",
        reset_log=args.reset_log,
    )

    # Also call the locator's initializer for compatibility. It will not rewrite
    # the file if it already exists.
    vision.init_trajectory_log(config)

    map_output_dir = VISUALIZATION_DIR / output_config.get("output_dir", "map_output")
    if args.reset_map_output and map_output_dir.exists():
        shutil.rmtree(map_output_dir)
        print(f"Reset map output: {map_output_dir}")
    map_output_dir.mkdir(parents=True, exist_ok=True)

    map_every = args.map_every
    if map_every is None:
        map_every = int(output_config.get("save_interval_frames", 1))
    map_every = max(1, int(map_every))

    picam2 = None
    test_frame_rgb = None

    if use_camera:
        if vision.Picamera2 is None:
            raise RuntimeError("Picamera2 is not available. Set use_camera=false for test-image mode.")
        picam2 = vision.setup_camera(
            int(config["frame_width"]),
            int(config["frame_height"]),
            config,
        )
    else:
        test_frame_rgb = vision.load_test_image(config)

    prev_gray = None

    initial_map_cell = config.get("initial_map_cell", [0.0, 0.0])
    try:
        pos_x_cell = float(initial_map_cell[0])
        pos_y_cell = float(initial_map_cell[1])
    except Exception:
        pos_x_cell = 0.0
        pos_y_cell = 0.0

    frame_count = 0
    map_frame_count = 0

    print("Integrated vision-map runner started.")
    print(f"Mode: {'camera' if use_camera else 'test image'}")
    print("Coordinate system: x=column, y=row, 1 unit=one ceiling grid cell")
    print(f"Initial map cell: ({pos_x_cell:.3f}, {pos_y_cell:.3f})")
    print(f"Trajectory: {trajectory_path}")
    print(f"Map output dir: {map_output_dir}")
    print(f"Map render interval: every {map_every} vision frame(s)")
    print(f"Map display window: {map_display_enabled}")
    print(f"Vision debug windows: {vision_display_enabled}")
    print(f"Final debug playback window: {final_debug_display_enabled}")
    print(f"Navigation enabled: {navigation_enabled}")
    if route is not None:
        print(f"Navigation status: {navigation_status_path}")

    if any_display_enabled:
        print("Display mode enabled. Press q in an OpenCV window to stop.")
    else:
        print("Headless mode. Press Ctrl+C to stop.")

    try:
        while True:
            loop_start = time.time()

            if use_camera:
                frame_rgb = picam2.capture_array()
            else:
                frame_rgb = test_frame_rgb.copy()

            frame_rgb = vision.resize_frame_if_needed(frame_rgb, config)

            # Provide the current planned navigation heading to the locator.
            # The locator can map its forward/lateral motion estimate to map
            # x/y according to this heading instead of using a fixed swap.
            if navigation_enabled and route is not None:
                config["current_nav_heading"] = logical_heading
                segment_target = _route_segment_target_cell(route, current_segment_index)
                if segment_target is not None:
                    # This is the router-provided coordinate of the current segment endpoint.
                    # During TURN state it is the turn cell reference; during FORWARD it is
                    # also useful as the next target reference for monitoring.
                    config["router_turn_cell"] = segment_target
                    config["current_segment_target_cell"] = segment_target
            else:
                config["current_nav_heading"] = config.get("planned_heading", "UP")

            process_result = vision.process_frame(
                frame_rgb,
                config,
                prev_gray,
                pos_x_cell,
                pos_y_cell,
                frame_count,
                trajectory_path,
                return_debug=final_debug_display_enabled,
            )

            final_debug_frame = None
            if final_debug_display_enabled:
                prev_gray, pos_x_cell, pos_y_cell, final_debug_frame, _lines_only_debug = process_result
            else:
                prev_gray, pos_x_cell, pos_y_cell = process_result

            if navigation_enabled and route is not None:
                nav_status, current_segment_index, logical_heading = nav_router.navigation_status(
                    route,
                    pos_x_cell,
                    pos_y_cell,
                    current_segment_index,
                    logical_heading,
                )
                # Add light/localization fields from the latest trajectory row if available.
                nav_status["position_source"] = "vision_grid_with_optional_light_correction"
                nav_status["turn_flag_for_controller"] = bool(nav_status.get("turn_flag", False))
                nav_status["turn_target_cell"] = config.get("router_turn_cell", config.get("turn_target_cell", None))
                nav_status["turn_cell_distance_cell"] = config.get("turn_cell_distance_cell", None)
                nav_status["turn_completed_by_vision"] = bool(config.get("turn_completed_by_vision", False))
                nav_router.save_navigation_status(nav_status, navigation_status_path)

            if final_debug_display_enabled and frame_count % final_debug_every == 0:
                cv2.imshow("Ceiling Final Debug", final_debug_frame)

            latest_map = None
            if frame_count % map_every == 0:
                map_canvas, latest_map = render_and_save_map(map_frame_count)

                if map_display_enabled:
                    cv2.imshow("Integrated Robot Map", map_canvas)

                latest_text = str(latest_map) if latest_map is not None else "not saved"
                if navigation_enabled and route is not None:
                    nav_short = f", nav_seg={current_segment_index}, heading={logical_heading}"
                else:
                    nav_short = ""
                print(
                    f"frame={frame_count:06d}, "
                    f"pos=({pos_x_cell:.3f}, {pos_y_cell:.3f}) cell, "
                    f"map={latest_text}"
                    f"{nav_short}"
                )
                map_frame_count += 1

            # Required for OpenCV windows to update and for q to stop.
            if any_display_enabled:
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("q pressed. Stopping.")
                    break

            frame_count += 1

            if args.max_frames is not None and frame_count >= args.max_frames:
                print("Reached max frame count. Stopping.")
                break

            elapsed = time.time() - loop_start
            sleep_time = sample_period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("Stopping integrated runner.")

    finally:
        if picam2 is not None:
            picam2.stop()
        if any_display_enabled:
            cv2.destroyAllWindows()
        print("Integrated runner stopped.")


if __name__ == "__main__":
    main()
