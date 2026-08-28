"""
visualization.py
-----------------
Drawing helpers: bounding boxes, track trails, the counting line, and a
live counts overlay panel. Kept separate from detection/counting logic
so the pipeline stays easy to test headlessly (no display needed).
"""

from typing import List, Dict, Tuple

import cv2
import numpy as np

# Consistent color per vehicle class (BGR)
CLASS_COLORS = {
    "car": (66, 135, 245),
    "truck": (36, 199, 255),
    "bus": (52, 235, 131),
    "motorcycle": (235, 64, 191),
}
DEFAULT_COLOR = (200, 200, 200)


def _color_for(class_name: str) -> Tuple[int, int, int]:
    return CLASS_COLORS.get(class_name, DEFAULT_COLOR)


def draw_tracks(frame: np.ndarray, tracks: List, box_thickness: int = 2,
                 font_scale: float = 0.55) -> np.ndarray:
    for t in tracks:
        color = _color_for(t.class_name)
        cv2.rectangle(frame, (t.x1, t.y1), (t.x2, t.y2), color, box_thickness)

        label = f"#{t.track_id} {t.class_name} {t.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(frame, (t.x1, t.y1 - th - 8), (t.x1 + tw + 4, t.y1), color, -1)
        cv2.putText(frame, label, (t.x1 + 2, t.y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1, cv2.LINE_AA)

        cx, cy = t.centroid
        cv2.circle(frame, (cx, cy), 3, color, -1)
    return frame


def draw_trails(frame: np.ndarray, trails: Dict[int, "deque"]) -> np.ndarray:
    for track_id, pts in trails.items():
        pts_list = list(pts)
        for i in range(1, len(pts_list)):
            cv2.line(frame, pts_list[i - 1], pts_list[i], (120, 120, 120), 1, cv2.LINE_AA)
    return frame


def draw_line(frame: np.ndarray, point1, point2, thickness: int = 3) -> np.ndarray:
    cv2.line(frame, tuple(point1), tuple(point2), (0, 0, 255), thickness, cv2.LINE_AA)
    for p in (point1, point2):
        cv2.circle(frame, tuple(p), 6, (0, 0, 255), -1)
    return frame


def draw_counts_panel(frame: np.ndarray, breakdown: Dict[str, Dict[str, int]],
                       font_scale: float = 0.6) -> np.ndarray:
    """Draws a semi-transparent panel in the top-left with live counts."""
    directions = list(breakdown.keys())
    all_classes = sorted({c for d in breakdown.values() for c in d.keys()})

    lines = []
    for direction in directions:
        total = sum(breakdown[direction].values())
        lines.append(f"{direction}: {total}")
        for c in all_classes:
            n = breakdown[direction].get(c, 0)
            if n:
                lines.append(f"  {c}: {n}")

    if not lines:
        lines = ["No crossings yet"]

    padding = 10
    line_h = 22
    panel_w = 220
    panel_h = padding * 2 + line_h * len(lines)

    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), (30, 30, 30), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    y = 10 + padding + 16
    for line in lines:
        cv2.putText(frame, line, (10 + padding, y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
        y += line_h

    return frame
