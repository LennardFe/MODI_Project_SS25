from calculations.bearing_calc import get_bearings
from calculations.distance_calc import get_distance_changes
import json
import numpy as np
import math
import sqlite3

CALIBRATION_ANCHOR = "DC0F"  # globally set at beginning
THETA = 121  # globally set by another thread


def select_target(gesture_start, gesture_end):
    print("Selecting Target")
    bearings = get_bearings(
        anchors,
        CALIBRATION_ANCHOR,
        get_initial_position(),
        math.radians(THETA),
        get_current_position(gesture_start),
    )
    distance_changes = get_distance_changes(gesture_start, gesture_end)
    anchor_min_bearing = min(bearings)
    anchor_min_distance_change = min(distance_changes)
    if anchor_min_bearing == anchor_min_distance_change:
        print("SUCCESS. CONCURRING OPINIONS.")
        print(f"Selected Target: {anchor_min_bearing}")
    else:
        # What do we do here?
        print()

    # TODO Reset gesture recognition


def read_anchor_config():
    with open("assets/anchor_config.json", "r") as f:
        config = json.load(f)
    anchors = {}
    for anchor in config:
        anchors[anchor["id"]] = np.array([anchor["x"], anchor["y"], anchor["z"]])
    return anchors


def get_initial_position():
    conn = sqlite3.connect("assets/test_data.db")
    initial_position = conn.execute(
        """SELECT est_position FROM location_data ORDER BY timestamp ASC LIMIT 1"""
    ).fetchone()[0]
    conn.close()
    initial_position = str(initial_position).replace("[", "").replace("]", "")
    initial_position = np.fromstring(initial_position, dtype=float, sep=",")
    return initial_position[:3]


def get_current_position(gesture_end):
    conn = sqlite3.connect("assets/test_data.db")
    current_position = conn.execute(
        """SELECT est_position FROM location_data WHERE timestamp < ? ORDER BY timestamp DESC LIMIT 1""",
        (gesture_end,),
    ).fetchone()[0]
    conn.close()
    current_position = str(current_position).replace("[", "").replace("]", "")
    current_position = np.fromstring(current_position, dtype=float, sep=",")
    return current_position[:3]


if __name__ == "__main__":
    anchors = read_anchor_config()
    print(anchors)
