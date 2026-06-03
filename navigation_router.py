"""
Simple occupancy-grid router for the ceiling-grid RPi car project.

Coordinate convention:
    x = column index, increases to the right
    y = row index, increases downward
    1 unit = one ceiling grid cell

Router:
    - 4-neighbor A* only: UP/DOWN/LEFT/RIGHT, no diagonal movement
    - avoids obstacle/out_of_bounds cells
    - adds turn penalty to prefer fewer turns
    - optionally prefers trunk rows, e.g. row 16/17
    - default mode searches usable trunk-row entry/exit cells:
        start -> reachable entry on row 16/17 -> horizontal trunk segment
        -> exit on row 16/17 near the goal column -> goal
      then falls back to global A* only if no valid trunk-entry route is found
      and fallback_to_global_astar is enabled.
"""

from __future__ import annotations

import heapq
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


Cell = Tuple[int, int]

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MAP_CONFIG_PATH = PROJECT_ROOT / "visualization" / "map_config.json"
DEFAULT_ROUTE_PATH = PROJECT_ROOT / "data" / "route_path.json"
DEFAULT_STATUS_PATH = PROJECT_ROOT / "data" / "navigation_status.json"


DIRS = {
    "UP": (0, -1),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
    "RIGHT": (1, 0),
}

OPPOSITE = {
    "UP": "DOWN",
    "DOWN": "UP",
    "LEFT": "RIGHT",
    "RIGHT": "LEFT",
}


def load_map_config(path: Path | str = DEFAULT_MAP_CONFIG_PATH) -> dict:
    path = Path(path)
    with open(path, "r") as f:
        return json.load(f)


def navigation_config(map_config: dict) -> dict:
    return map_config.get("navigation", {})


def is_auto_heading(heading: str | None) -> bool:
    return heading is None or str(heading).upper() in ("", "AUTO", "NONE")


def normalize_heading(heading: str | None, default: str = "UP") -> str:
    if is_auto_heading(heading):
        return default
    h = str(heading).upper()
    return h if h in DIRS else default


def parse_cell(value, name: str = "cell") -> Cell:
    if isinstance(value, str):
        parts = value.replace("(", "").replace(")", "").split(",")
        if len(parts) != 2:
            raise ValueError(f"{name} must be x,y")
        return int(round(float(parts[0]))), int(round(float(parts[1])))
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(round(float(value[0]))), int(round(float(value[1])))
    raise ValueError(f"{name} must be [x, y] or x,y")


def occupancy_rows(map_config: dict) -> List[str]:
    return map_config.get("occupancy_grid", {}).get("rows", [])


def map_size(map_config: dict) -> tuple[int, int]:
    map_info = map_config.get("map", {})
    rows = occupancy_rows(map_config)
    height = int(map_info.get("height_cells", len(rows)))
    width = int(map_info.get("width_cells", max((len(r) for r in rows), default=0)))
    return width, height


def cell_type(map_config: dict, cell: Cell) -> str:
    x, y = cell
    rows = occupancy_rows(map_config)
    if y < 0 or y >= len(rows):
        return "out_of_bounds"
    if x < 0 or x >= len(rows[y]):
        return "out_of_bounds"
    ch = rows[y][x]
    if ch == "#":
        return "obstacle"
    if ch == ".":
        return "free"
    if ch == "L":
        return "light"
    return "unknown"


def is_walkable(map_config: dict, cell: Cell) -> bool:
    return cell_type(map_config, cell) in ("free", "light")


def heuristic(a: Cell, b: Cell) -> float:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def direction_between(a: Cell, b: Cell) -> Optional[str]:
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    for name, vec in DIRS.items():
        if vec == (dx, dy):
            return name
    return None


def step_neighbors(cell: Cell) -> Iterable[tuple[Cell, str]]:
    x, y = cell
    for heading, (dx, dy) in DIRS.items():
        yield (x + dx, y + dy), heading


def astar(
    map_config: dict,
    start: Cell,
    goal: Cell,
    start_heading: str | None = "UP",
    turn_penalty: float = 3.0,
    trunk_rows: Optional[List[int]] = None,
    trunk_row_penalty: float = 0.0,
    prefer_axis_aligned_to_goal: bool = True,
    allowed_rows: Optional[List[int]] = None,
) -> Optional[List[Cell]]:
    """
    A* over state=(x, y, heading) so turn penalty is included in the optimal cost.

    If start_heading is AUTO/None, the first step does not pay a turn penalty.
    If allowed_rows is given, all cells in this path must stay on those rows.
    """
    start_heading_state = "AUTO" if is_auto_heading(start_heading) else normalize_heading(start_heading)
    if not is_walkable(map_config, start) or not is_walkable(map_config, goal):
        return None

    trunk_rows = list(trunk_rows or [])
    allowed_rows_set = None if allowed_rows is None else set(int(r) for r in allowed_rows)
    if allowed_rows_set is not None:
        if start[1] not in allowed_rows_set or goal[1] not in allowed_rows_set:
            return None

    def extra_cost(cell: Cell) -> float:
        c = 0.0
        if trunk_rows and trunk_row_penalty > 0:
            # Small soft bias toward trunk rows. Keep it small so vertical legs
            # can still leave/reach the trunk.
            c += min(abs(cell[1] - r) for r in trunk_rows) * trunk_row_penalty
        return c

    start_state = (start[0], start[1], start_heading_state)
    pq = []
    heapq.heappush(pq, (heuristic(start, goal), 0.0, start_state))
    came_from: Dict[tuple[int, int, str], tuple[int, int, str]] = {}
    best_cost: Dict[tuple[int, int, str], float] = {start_state: 0.0}

    goal_state = None

    while pq:
        _priority, cost_so_far, state = heapq.heappop(pq)
        x, y, heading = state
        cell = (x, y)

        if cell == goal:
            goal_state = state
            break

        if cost_so_far > best_cost.get(state, float("inf")) + 1e-9:
            continue

        for nxt, next_heading in step_neighbors(cell):
            if allowed_rows_set is not None and nxt[1] not in allowed_rows_set:
                continue
            if not is_walkable(map_config, nxt):
                continue

            step_cost = 1.0
            if heading in DIRS and heading != next_heading:
                step_cost += float(turn_penalty)

            if prefer_axis_aligned_to_goal:
                # Tiny tie-break: when not requiring a turn, prefer steps that
                # reduce Manhattan distance to the current goal.
                if heuristic(nxt, goal) < heuristic(cell, goal):
                    step_cost -= 1e-3

            new_cost = cost_so_far + step_cost + extra_cost(nxt)
            new_state = (nxt[0], nxt[1], next_heading)

            if new_cost < best_cost.get(new_state, float("inf")):
                best_cost[new_state] = new_cost
                came_from[new_state] = state
                priority = new_cost + heuristic(nxt, goal)
                heapq.heappush(pq, (priority, new_cost, new_state))

    if goal_state is None:
        return None

    path = []
    state = goal_state
    while True:
        path.append((state[0], state[1]))
        if state == start_state:
            break
        state = came_from[state]
    path.reverse()
    return path

def path_heading_at_end(path: List[Cell], fallback: str) -> str:
    if len(path) >= 2:
        h = direction_between(path[-2], path[-1])
        if h:
            return h
    return normalize_heading(fallback)


def join_paths(parts: List[List[Cell]]) -> List[Cell]:
    out: List[Cell] = []
    for part in parts:
        if not part:
            continue
        if not out:
            out.extend(part)
        else:
            out.extend(part[1:] if out[-1] == part[0] else part)
    return out


def trunk_cells(map_config: dict, trunk_rows: List[int]) -> List[Cell]:
    """Return all walkable cells on the configured trunk rows."""
    width, _height = map_size(map_config)
    cells: List[Cell] = []
    for row in trunk_rows:
        for x in range(width):
            cell = (x, int(row))
            if is_walkable(map_config, cell):
                cells.append(cell)
    return cells


def count_turns_in_path(path: List[Cell], initial_heading: str = "AUTO") -> int:
    """Count heading changes in a cell path."""
    return max(0, len(compress_segments(path, initial_heading)) - 1)


def path_uses_only_rows(path: List[Cell], rows: List[int]) -> bool:
    allowed = set(int(r) for r in rows)
    return all(cell[1] in allowed for cell in path)


def choose_waypoint_route(map_config: dict, start: Cell, goal: Cell, nav: dict) -> Optional[List[Cell]]:
    """
    Plan through trunk-row entry/exit cells instead of fixed (start_x, trunk_row).

    New route shape:
        start -> entry_on_trunk -> exit_on_same_trunk_row -> goal

    The middle segment is hard-constrained to one trunk row, so the main
    horizontal movement stays on row 16 or row 17. The approach/exit segments
    are allowed to move normally so the robot can find a reachable opening.
    """
    trunk_rows = [int(r) for r in nav.get("trunk_rows", [16, 17])]
    turn_penalty = float(nav.get("turn_penalty", 3.0))
    trunk_row_penalty = float(nav.get("trunk_row_penalty", 0.0))
    initial_heading = nav.get("initial_heading", "AUTO")
    allow_fallback = bool(nav.get("fallback_to_global_astar", True))

    candidate_limit = int(nav.get("trunk_candidate_limit", 24))
    candidate_limit = max(1, candidate_limit)

    # Scoring weights. Increase trunk_goal_column_weight if you want the exit
    # point to be closer to goal_x before leaving the trunk.
    entry_dist_weight = float(nav.get("trunk_entry_distance_weight", 0.15))
    exit_dist_weight = float(nav.get("trunk_exit_distance_weight", 0.15))
    goal_column_weight = float(nav.get("trunk_goal_column_weight", 0.25))
    trunk_horizontal_bonus = float(nav.get("trunk_horizontal_bonus", 0.01))

    candidates = []

    for trunk_row in trunk_rows:
        row_cells = trunk_cells(map_config, [trunk_row])
        if not row_cells:
            continue

        # Entry candidates: reachable openings on row16/17 near the start.
        entry_candidates = sorted(
            row_cells,
            key=lambda c: (heuristic(start, c), abs(c[0] - start[0]), abs(c[0] - goal[0]))
        )[:candidate_limit]

        # Exit candidates: same row, preferably near the goal column.
        exit_candidates = sorted(
            row_cells,
            key=lambda c: (abs(c[0] - goal[0]), heuristic(c, goal), abs(c[0] - start[0]))
        )[:candidate_limit]

        for entry in entry_candidates:
            p1 = astar(map_config, start, entry, initial_heading, turn_penalty, [], 0.0)
            if not p1:
                continue
            heading_after_p1 = path_heading_at_end(p1, initial_heading)

            for exit_cell in exit_candidates:
                p2 = astar(
                    map_config,
                    entry,
                    exit_cell,
                    heading_after_p1,
                    turn_penalty,
                    [trunk_row],
                    0.0,
                    allowed_rows=[trunk_row],
                )
                if not p2:
                    continue
                heading_after_p2 = path_heading_at_end(p2, heading_after_p1)

                p3 = astar(map_config, exit_cell, goal, heading_after_p2, turn_penalty, [], 0.0)
                if not p3:
                    continue

                path = join_paths([p1, p2, p3])
                turns = count_turns_in_path(path, initial_heading)
                trunk_horizontal_len = abs(exit_cell[0] - entry[0])
                score = (
                    len(path)
                    + turns * turn_penalty
                    + heuristic(start, entry) * entry_dist_weight
                    + heuristic(exit_cell, goal) * exit_dist_weight
                    + abs(exit_cell[0] - goal[0]) * goal_column_weight
                    - trunk_horizontal_len * trunk_horizontal_bonus
                    + abs(start[1] - trunk_row) * 0.01
                    + abs(goal[1] - trunk_row) * 0.01
                )
                candidates.append((score, path, entry, exit_cell, trunk_row))

    if candidates:
        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]

    if allow_fallback:
        return astar(
            map_config,
            start,
            goal,
            initial_heading,
            turn_penalty=turn_penalty,
            trunk_rows=trunk_rows,
            trunk_row_penalty=trunk_row_penalty,
        )

    return None

def compress_segments(path: List[Cell], initial_heading: str = "UP") -> List[dict]:
    if not path:
        return []
    if len(path) == 1:
        return [{
            "from": list(path[0]),
            "to": list(path[0]),
            "heading": normalize_heading(initial_heading),
            "length_cells": 0,
            "action": "arrived",
        }]

    segments = []
    seg_start = path[0]
    current_heading = direction_between(path[0], path[1]) or normalize_heading(initial_heading)
    length = 0

    for i in range(1, len(path)):
        h = direction_between(path[i - 1], path[i])
        if h != current_heading:
            segments.append({
                "from": list(seg_start),
                "to": list(path[i - 1]),
                "heading": current_heading,
                "length_cells": length,
                "action": "go_straight",
            })
            seg_start = path[i - 1]
            current_heading = h or current_heading
            length = 1
        else:
            length += 1

    segments.append({
        "from": list(seg_start),
        "to": list(path[-1]),
        "heading": current_heading,
        "length_cells": length,
        "action": "go_straight",
    })
    return segments


def turn_direction(from_heading: str, to_heading: str) -> str:
    from_heading = normalize_heading(from_heading)
    to_heading = normalize_heading(to_heading)
    if from_heading == to_heading:
        return "STRAIGHT"
    if OPPOSITE[from_heading] == to_heading:
        return "UTURN"

    right_turns = {
        "UP": "RIGHT",
        "RIGHT": "DOWN",
        "DOWN": "LEFT",
        "LEFT": "UP",
    }
    if right_turns[from_heading] == to_heading:
        return "RIGHT"
    return "LEFT"


def compute_turn_points(segments: List[dict]) -> List[dict]:
    turns = []
    for i in range(len(segments) - 1):
        a = segments[i]
        b = segments[i + 1]
        cell = a["to"]
        turns.append({
            "segment_index": i,
            "cell": cell,
            "from_heading": a["heading"],
            "to_heading": b["heading"],
            "turn": turn_direction(a["heading"], b["heading"]),
        })
    return turns



def infer_initial_heading_from_path(path: List[Cell], default: str = "UP") -> str:
    if len(path) >= 2:
        return direction_between(path[0], path[1]) or normalize_heading(default)
    return normalize_heading(default)

def plan_route(map_config: dict, start: Cell | None = None, goal: Cell | None = None, initial_heading: str | None = None) -> dict:
    nav = navigation_config(map_config)
    if start is None:
        start = parse_cell(nav.get("start_cell", [0, 0]), "start_cell")
    if goal is None:
        goal = parse_cell(nav.get("goal_cell", [0, 0]), "goal_cell")

    requested_heading = initial_heading if initial_heading is not None else nav.get("initial_heading", "AUTO")
    planning_heading = "AUTO" if is_auto_heading(requested_heading) else normalize_heading(requested_heading)
    path = choose_waypoint_route(map_config, start, goal, {**nav, "initial_heading": planning_heading})
    if not path:
        raise RuntimeError(f"No route found from {start} to {goal}")

    inferred_heading = infer_initial_heading_from_path(path, default="UP")
    route_initial_heading = inferred_heading if is_auto_heading(requested_heading) else normalize_heading(requested_heading)
    segments = compress_segments(path, route_initial_heading)
    turn_points = compute_turn_points(segments)

    return {
        "schema": "ceiling_grid_route_v1",
        "coordinate_definition": "x=column, y=row, y increases downward, 1 unit=one ceiling grid cell",
        "start_cell": list(start),
        "goal_cell": list(goal),
        "initial_heading": route_initial_heading,
        "initial_heading_source": "auto_from_first_segment" if is_auto_heading(requested_heading) else "manual_override",
        "path": [list(p) for p in path],
        "segments": segments,
        "turn_points": turn_points,
        "turn_arrival_radius_cells": float(nav.get("turn_arrival_radius_cells", 0.6)),
        "goal_arrival_radius_cells": float(nav.get("goal_arrival_radius_cells", 0.8)),
        "trunk_rows": nav.get("trunk_rows", [16, 17]),
        "router_mode": "trunk_entry_exit_waypoint_astar",
    }

def save_route(route: dict, path: Path | str = DEFAULT_ROUTE_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(exist_ok=True)
    with open(path, "w") as f:
        json.dump(route, f, indent=2)
    return path


def load_route(path: Path | str = DEFAULT_ROUTE_PATH) -> Optional[dict]:
    path = Path(path)
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)


def distance_to_cell(x: float, y: float, cell: Cell | list) -> float:
    return math.hypot(float(cell[0]) - float(x), float(cell[1]) - float(y))


def navigation_status(
    route: dict,
    x_cell: float,
    y_cell: float,
    current_segment_index: int,
    current_heading: str,
) -> tuple[dict, int, str]:
    """
    Decide whether the car is near a turn point or final goal.

    Return:
        status dict
        updated segment index
        updated logical heading
    """
    segments = route.get("segments", [])
    if not segments:
        return {
            "navigation_enabled": False,
            "route_done": True,
            "turn_flag": False,
            "reason": "no_segments",
        }, current_segment_index, current_heading

    current_segment_index = max(0, min(int(current_segment_index), len(segments) - 1))
    current = segments[current_segment_index]
    target = current["to"]

    turn_radius = float(route.get("turn_arrival_radius_cells", 0.6))
    goal_radius = float(route.get("goal_arrival_radius_cells", 0.8))
    dist_to_target = distance_to_cell(x_cell, y_cell, target)
    is_last = current_segment_index >= len(segments) - 1

    turn_flag = False
    turn = "NONE"
    target_heading = current["heading"]
    next_turn_cell = target
    route_done = False

    if is_last:
        if dist_to_target <= goal_radius:
            route_done = True
    else:
        next_seg = segments[current_segment_index + 1]
        if dist_to_target <= turn_radius:
            turn_flag = True
            target_heading = next_seg["heading"]
            turn = turn_direction(current["heading"], target_heading)

            # Once in the turn radius, advance the segment for the next iteration.
            current_segment_index += 1
            current_heading = target_heading

    status = {
        "navigation_enabled": True,
        "route_done": route_done,
        "current_position": [float(x_cell), float(y_cell)],
        "current_segment_index": current_segment_index,
        "current_heading": current_heading,
        "target_heading": target_heading,
        "turn_flag": bool(turn_flag),
        "turn_direction": turn,
        "next_target_cell": next_turn_cell,
        "distance_to_next_target_cell": dist_to_target,
        "goal_cell": route.get("goal_cell"),
        "segments_total": len(segments),
        "route_path_length_cells": len(route.get("path", [])),
    }
    return status, current_segment_index, current_heading


def save_navigation_status(status: dict, path: Path | str = DEFAULT_STATUS_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(exist_ok=True)
    status = dict(status)
    status["timestamp"] = __import__("time").time()
    with open(path, "w") as f:
        json.dump(status, f, indent=2)
    return path


def _save_json_file(path: Path | str, data: dict) -> Path:
    """Write JSON with project-friendly formatting."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    return path


def apply_cli_start_goal_to_configs(
    map_config: dict,
    map_config_path: Path | str,
    start: Cell | None = None,
    goal: Cell | None = None,
    heading: str | None = None,
    vision_config_path: Path | str | None = None,
    persist: bool = True,
) -> tuple[dict, list[str]]:
    """Persist CLI route overrides into map_config and vision_config."""
    changed: list[str] = []
    nav = map_config.setdefault("navigation", {})

    if start is not None:
        nav["start_cell"] = [int(start[0]), int(start[1])]
        changed.append(f"map_config.navigation.start_cell={nav['start_cell']}")

    if goal is not None:
        nav["goal_cell"] = [int(goal[0]), int(goal[1])]
        changed.append(f"map_config.navigation.goal_cell={nav['goal_cell']}")

    if heading is not None:
        nav["initial_heading"] = normalize_heading(heading)
        changed.append(f"map_config.navigation.initial_heading={nav['initial_heading']}")
    else:
        nav["initial_heading"] = nav.get("initial_heading", "AUTO")

    # Useful defaults for the new trunk-entry router. Existing config values
    # are preserved if already present.
    nav.setdefault("trunk_rows", [16, 17])
    nav.setdefault("trunk_candidate_limit", 24)
    nav.setdefault("fallback_to_global_astar", True)

    if persist and changed:
        _save_json_file(map_config_path, map_config)

    if start is not None and vision_config_path is not None:
        vision_path = Path(vision_config_path)
        if vision_path.exists():
            try:
                with open(vision_path, "r") as f:
                    vision_config = json.load(f)
                vision_config["initial_map_cell"] = [float(start[0]), float(start[1])]
                if persist:
                    _save_json_file(vision_path, vision_config)
                changed.append(f"vision_config.initial_map_cell={vision_config['initial_map_cell']}")
            except Exception as exc:
                changed.append(f"vision_config update skipped: {exc}")
        else:
            changed.append(f"vision_config update skipped: not found at {vision_path}")

    return map_config, changed


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plan an obstacle-avoiding ceiling-grid route.")
    parser.add_argument("--map-config", default=str(DEFAULT_MAP_CONFIG_PATH))
    parser.add_argument("--vision-config", default=str(PROJECT_ROOT / "vision" / "vision_config.json"))
    parser.add_argument("--start", default=None, help="Start cell as x,y. Also updates config by default.")
    parser.add_argument("--goal", default=None, help="Goal cell as x,y. Also updates config by default.")
    parser.add_argument("--heading", default=None, choices=list(DIRS.keys()), help="Optional manual override. Omit it to infer heading from the first route segment.")
    parser.add_argument("--output", default=str(DEFAULT_ROUTE_PATH))
    parser.add_argument("--no-write-config", action="store_true", help="Only generate route_path.json; do not persist --start/--goal into configs.")
    args = parser.parse_args()

    map_config_path = Path(args.map_config)
    cfg = load_map_config(map_config_path)
    start = parse_cell(args.start, "start") if args.start else None
    goal = parse_cell(args.goal, "goal") if args.goal else None

    cfg, config_changes = apply_cli_start_goal_to_configs(
        cfg,
        map_config_path,
        start=start,
        goal=goal,
        heading=args.heading,
        vision_config_path=Path(args.vision_config),
        persist=not args.no_write_config,
    )

    nav = navigation_config(cfg)
    if start is None:
        start = parse_cell(nav.get("start_cell", [0, 0]), "start_cell")
    if goal is None:
        goal = parse_cell(nav.get("goal_cell", [0, 0]), "goal_cell")

    route = plan_route(cfg, start=start, goal=goal, initial_heading=args.heading)
    out = save_route(route, args.output)

    print(f"Route saved: {out}")
    if config_changes:
        mode = "updated" if not args.no_write_config else "would update"
        print(f"Config {mode}:")
        for item in config_changes:
            print(f"  - {item}")
    print(f"start={route['start_cell']} goal={route['goal_cell']} heading={route['initial_heading']} ({route.get('initial_heading_source', 'unknown')})")
    print(f"router_mode={route.get('router_mode')}, trunk_rows={route.get('trunk_rows')}")
    print(f"path length={len(route['path'])}, segments={len(route['segments'])}, turns={len(route['turn_points'])}")
    for i, seg in enumerate(route["segments"]):
        print(f"  segment {i}: {seg['from']} -> {seg['to']} heading={seg['heading']} length={seg['length_cells']}")
