from calculations.bearing_calc import get_bearings
from calculations.distance_calc import get_distance_changes
import json
import numpy as np
import math
import sqlite3

CALIBRATION_ANCHOR = "DC0F"  # globally set at beginning
THETA = 121  # globally set by another thread (Initial Richtung + Theta) ~= Heading


def select_target(gesture_start, gesture_end):
    print("Selecting Target")

    # Return python dictionary with ids and angle (bearing) of anchors
    bearings = get_bearings(
        read_anchor_config(),
        CALIBRATION_ANCHOR,
        get_initial_position(),
        math.radians(THETA),
        get_current_position(gesture_start),
    )

    # Get the distance changes from the gesture start to the gesture end
    distance_changes = get_distance_changes(gesture_start, gesture_end)

    # Get anchor with minimum bearing
    anchor_min_bearing = min(bearings)

    # Get anchor with minimum distance change (works because decrease is negative)
    anchor_min_distance_change = min(distance_changes)


    if anchor_min_bearing == anchor_min_distance_change:
        print("SUCCESS. CONCURRING OPINIONS.")
        print(f"Selected Target: {anchor_min_bearing}")
    else:
        pass # Do more complex score calculation based on relative changes


def read_anchor_config():
    with open("assets/anchor_config.json", "r") as f:
        config = json.load(f)
    anchors = {}
    for anchor in config:
        anchors[anchor["id"]] = np.array([anchor["x"], anchor["y"], anchor["z"]])
    return anchors

# First ever position of the tag, this is where we calibrated the tag to
def get_initial_position():
    conn = sqlite3.connect("assets/MODI.db")
    initial_position = conn.execute(
        """SELECT est_position_x, est_position_y FROM location_data WHERE est_position_x IS NOT NULL AND est_position_y IS NOT NULL ORDER BY timestamp ASC LIMIT 1"""
    ).fetchone()
    conn.close()
    return np.array(initial_position)

# Get last known position of the tag before the gesture started
def get_current_position(gesture_start):
    conn = sqlite3.connect("assets/MODI.db")
    current_position = conn.execute(
        """SELECT est_position_x, est_position_y FROM location_data WHERE timestamp < ? AND est_position_x IS NOT NULL AND est_position_y IS NOT NULL ORDER BY timestamp DESC LIMIT 1""",
        (gesture_start,),
    ).fetchone()
    conn.close()
    return np.array(current_position)


if __name__ == "__main__":
    anchors = read_anchor_config()
    print(anchors)
