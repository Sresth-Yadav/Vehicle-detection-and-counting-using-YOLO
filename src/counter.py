"""
counter.py
----------
Implements line-crossing counting for tracked vehicles.

A vehicle is counted the moment its centroid crosses from one side of a
user-defined line to the other. We determine "side" with the sign of the
2D cross product of (line vector) x (point vector) — a standard, cheap
way to test which side of a line a point sits on. Each track's previous
side is cached; a sign flip between frames = a crossing event.
"""

from collections import defaultdict, deque
from typing import Dict, Tuple, List

import numpy as np


def _side_of_line(p1, p2, point) -> float:
    """
    Returns a signed value: positive if `point` is on one side of the
    line p1->p2, negative if on the other, ~0 if on the line.
    """
    x1, y1 = p1
    x2, y2 = p2
    px, py = point
    return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)


class LineCounter:
    """
    Tracks per-vehicle side-of-line state and tallies IN/OUT counts,
    broken down by vehicle class, whenever a tracked centroid crosses
    the configured line.
    """

    def __init__(
        self,
        point1: Tuple[int, int],
        point2: Tuple[int, int],
        direction_labels: Dict[str, str] = None,
        trail_length: int = 30,
    ):
        self.point1 = tuple(point1)
        self.point2 = tuple(point2)
        self.labels = direction_labels or {"positive": "IN", "negative": "OUT"}

        # track_id -> last known side (+1 / -1)
        self._last_side: Dict[int, int] = {}
        # track_id -> set of frame indices already counted (avoid double count)
        self._counted: set = set()

        # counts[label][class_name] = int
        self.counts: Dict[str, Dict[str, int]] = {
            self.labels["positive"]: defaultdict(int),
            self.labels["negative"]: defaultdict(int),
        }

        # per-track centroid trail, for optional visualization
        self.trails: Dict[int, deque] = defaultdict(lambda: deque(maxlen=trail_length))

        # log of every counting event: (track_id, class_name, direction, frame_idx)
        self.event_log: List[dict] = []

    def total(self) -> int:
        return sum(sum(d.values()) for d in self.counts.values())

    def update(self, track_id: int, class_name: str, centroid: Tuple[int, int], frame_idx: int):
        """
        Feed one tracked object's current centroid in. Updates trail and,
        if a crossing is detected, increments the appropriate counter.
        """
        self.trails[track_id].append(centroid)

        side_val = _side_of_line(self.point1, self.point2, centroid)
        side = 1 if side_val > 0 else -1

        prev_side = self._last_side.get(track_id)
        self._last_side[track_id] = side

        if prev_side is None:
            return  # first time we've seen this track — nothing to compare yet

        if prev_side != side and track_id not in self._counted:
            direction = self.labels["positive"] if side > 0 else self.labels["negative"]
            self.counts[direction][class_name] += 1
            self._counted.add(track_id)
            self.event_log.append(
                {
                    "track_id": track_id,
                    "class_name": class_name,
                    "direction": direction,
                    "frame": frame_idx,
                }
            )

    def summary(self) -> Dict[str, int]:
        """Flat {direction: total_count} summary."""
        return {label: sum(d.values()) for label, d in self.counts.items()}

    def breakdown(self) -> Dict[str, Dict[str, int]]:
        """Full {direction: {class_name: count}} breakdown."""
        return {label: dict(d) for label, d in self.counts.items()}
