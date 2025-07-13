from target_selection.calculations.bearing_calc import get_bearings
from target_selection.calculations.distance_calc import get_distance_changesv2
from target_selection.calculations.theta_calc import get_theta
from target_selection.calculations.score_calc import get_best_scoring_anchor
import sqlite3, json, math, time
import numpy as np

# Main function to select target based on thr given arguments
def select_target(gesture_start, gesture_end, CALIBRATION_ANCHOR, database_name="MODI"):
    print("Selecting Target")
    print(f"Duration of gesture: {(gesture_end - gesture_start) * 1.0e-6}")

    theta = get_theta(database_name)
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

    selected_target = None
    if anchor_min_bearing == anchor_min_distance_change:
        print("SUCCESS. CONCURRING OPINIONS.")
        selected_target = anchor_min_bearing
        print(f"Selected Target: {selected_target}")
    else:
        print("DISAGREEMENT. SELECTING BEST SCORING ANCHOR.")
        selected_target = get_best_scoring_anchor(distance_changes, bearings, method="Ole")
        print(f"Selected Target: {selected_target}")
    try:
        import os
        os.makedirs("plots", exist_ok=True)
        with open("plots/last_selected_target.txt", "w") as f:
            f.write(f"{time.time_ns()},{selected_target}")
        print(f"📍 Target selection saved for animation: {selected_target}")
    except Exception as e:
        print(f"Error saving target selection: {e}")

    return selected_target

# Read anchor configuration from JSON file
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