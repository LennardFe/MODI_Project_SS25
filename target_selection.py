from bearing_calc import get_bearings
from distance_calc import get_distance_changes
import json
import numpy as np
import math

CALIBRATION_ANCHOR = "DC0F"  # globally set at beginning
THETA = 121  # globally set by another thread
CURRENT_POSITION = np.array(
    [0, 0, 0]
)  # globally set by another thread or queried from db
INITIAL_POSITION = np.array([0, 0, 0])  # globally set at beginning


def select_target(gesture_start, gesture_end):
    print("Selecting Target")
    bearings = get_bearings(
        anchors,
        CALIBRATION_ANCHOR,
        INITIAL_POSITION,
        math.radians(THETA),
        CURRENT_POSITION,
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
    with open("anchor_config.json", "r") as f:
        config = json.load(f)
    anchors = {}
    for anchor in config:
        anchors[anchor["id"]] = np.array([anchor["x"], anchor["y"], anchor["z"]])
    return anchors


if __name__ == "__main__":
    anchors = read_anchor_config()
    print(anchors)
