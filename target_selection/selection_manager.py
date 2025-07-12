from target_selection.calculations.bearing_calc import get_bearings
from target_selection.calculations.distance_calc import get_distance_changesv2
from target_selection.calculations.theta_calc import get_theta
from target_selection.calculations.score_calc import get_best_scoring_anchor
import json
import numpy as np
import math
import sqlite3
import matplotlib
import matplotlib.pyplot as plt
import threading
import time

THETA = 0  # globally set by another thread (Initial Richtung + Theta) ~= Heading


def select_target(gesture_start, gesture_end, CALIBRATION_ANCHOR, database_name="MODI"):
    print("Selecting Target")
    print(f"Duration of gesture: {(gesture_end - gesture_start) * 1.0e-6}")

    theta = -get_theta(database_name)
    print(f"Theta: {theta}")

    # Return python dictionary with ids and angle (bearing) of anchors
    bearings = get_bearings(
        read_anchor_config(),
        CALIBRATION_ANCHOR,
        get_initial_position(database_name),
        math.radians(theta),
        get_current_position(gesture_end, database_name),
    )

    # Get the distance changes from the gesture start to the gesture end
    distance_changes = get_distance_changesv2(gesture_start, gesture_end, database_name)

    print("Bearings: {}".format(bearings))
    print("Distance changes: {}".format(distance_changes))
    # Get anchor with minimum bearing
    anchor_min_bearing = min(bearings, key=bearings.get)

    # Get anchor with minimum distance change (works because decrease is negative)
    anchor_min_distance_change = min(distance_changes, key=distance_changes.get)
    plot_distance_change(gesture_start, gesture_end, anchor_min_distance_change, database_name)

    if anchor_min_bearing == anchor_min_distance_change:
        print("SUCCESS. CONCURRING OPINIONS.")
        print(f"Selected Target: {anchor_min_bearing}")
    else:
        print("DISAGREEMENT. SELECTING BEST SCORING ANCHOR.")
        anchor = get_best_scoring_anchor(distance_changes, bearings, method="Ole")
        print(f"Selected Target: {anchor}")


def plot_distance_change(gesture_start, gesture_end, anchor_id, database_name="MODI"):
    is_main_thread = threading.current_thread() is threading.main_thread()
    
    if not is_main_thread:
        matplotlib.use('Agg')
    
    conn = sqlite3.connect(f'assets/{database_name}.db', check_same_thread=False)
    cur = conn.cursor()
    plot_start = gesture_start - 2e9
    plot_end = gesture_end + 2e9

    cur.execute(
        """SELECT timestamp, est_position_x, est_position_y FROM location_data 
                   WHERE timestamp > ? AND timestamp < ? 
                   AND est_position_x IS NOT NULL 
                   AND est_position_y IS NOT NULL 
                   ORDER BY timestamp ASC""",
        (plot_start, plot_end),
    )
    data = cur.fetchall()
    conn.close()



    with open("assets/anchor_config.json", "r") as f:
        anchor_config = json.load(f)

    anchor_position = None
    for anchor in anchor_config:
        if anchor["id"] == anchor_id:
            anchor_position = np.array([anchor["x"], anchor["y"]])
            break


    timestamps = [row[0] for row in data]
    distances = []

    for row in data:
        tag_position = np.array([row[1], row[2]])  # est_position_x, est_position_y
        distance = np.linalg.norm(anchor_position - tag_position)
        distances.append(distance)

    timestamps_relative = [(ts - timestamps[0]) / 1e9 for ts in timestamps]
    gesture_start_relative = (gesture_start - timestamps[0]) / 1e9
    gesture_end_relative = (gesture_end - timestamps[0]) / 1e9

    initial_distance = distances[0]
    distance_changes = [(d - initial_distance) for d in distances]

    try:
        plt.figure(figsize=(12, 6))
        plt.plot(
            timestamps_relative,
            distance_changes,
            "b-",
            linewidth=1,
            label=f"Distance Change (Anchor {anchor_id})",
        )

        plt.axvline(
            x=gesture_start_relative,
            color="green",
            linestyle="--",
            linewidth=2,
            label="Gesture Start",
        )
        plt.axvline(
            x=gesture_end_relative,
            color="red",
            linestyle="--",
            linewidth=2,
            label="Gesture End",
        )

        plt.axhline(y=0, color="gray", linestyle="-", alpha=0.3)

        plt.xlabel("Time (seconds)")
        plt.ylabel("Distance Change (meters)")
        plt.title(f"Distance Change Over Time - Anchor {anchor_id} (Calculated from X,Y)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if is_main_thread:
            plt.show()
        else:
            timestamp_str = time.strftime("%Y%m%d_%H%M%S")
            filename = f"plots/distance_change_{anchor_id}_{timestamp_str}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
    except Exception as e:
        plt.close()

    return data


def read_anchor_config():
    with open("assets/anchor_config.json", "r") as f:
        config = json.load(f)
    anchors = {}
    for anchor in config:
        anchors[anchor["id"]] = np.array([anchor["x"], anchor["y"]])
    return anchors


# First ever position of the tag, this is where we calibrated the tag to
def get_initial_position(database_name="MODI"):
    conn = sqlite3.connect(f'assets/{database_name}.db', check_same_thread=False)
    initial_position = conn.execute(
        """SELECT est_position_x, est_position_y FROM location_data WHERE est_position_x IS NOT NULL AND est_position_y IS NOT NULL ORDER BY timestamp ASC LIMIT 1"""
    ).fetchone()
    conn.close()
    return np.array(initial_position)


# Get last known position of the tag before the gesture started
def get_current_position(gesture_start, database_name="MODI"):
    conn = sqlite3.connect(f'assets/{database_name}.db', check_same_thread=False)
    current_position = conn.execute(
        """SELECT est_position_x, est_position_y FROM location_data WHERE timestamp < ? AND est_position_x IS NOT NULL AND est_position_y IS NOT NULL ORDER BY timestamp DESC LIMIT 1""",
        (gesture_start,),
    ).fetchone()
    conn.close()
    return np.array(current_position)


if __name__ == "__main__":
    anchors = read_anchor_config()
    print(anchors)
