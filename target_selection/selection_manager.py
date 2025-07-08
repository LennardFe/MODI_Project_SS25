from target_selection.calculations.bearing_calc import get_bearings
from target_selection.calculations.distance_calc import get_distance_changesv2
from target_selection.calculations.theta_calc import get_theta
from target_selection.calculations.score_calc import get_best_scoring_anchor
import json
import numpy as np
import math
import sqlite3
import matplotlib.pyplot as plt

THETA = 0  # globally set by another thread (Initial Richtung + Theta) ~= Heading


def select_target(gesture_start, gesture_end, CALIBRATION_ANCHOR):
    print("Selecting Target")
    print(f"Duration of gesture: {(gesture_end - gesture_start) * 1.0e-6}")

    theta = -get_theta()
    print(f"Theta: {theta}")

    # Return python dictionary with ids and angle (bearing) of anchors
    bearings = get_bearings(
        read_anchor_config(),
        CALIBRATION_ANCHOR,
        get_initial_position(),
        math.radians(theta),
        get_current_position(gesture_end),
    )

    # Get the distance changes from the gesture start to the gesture end
    distance_changes = get_distance_changesv2(gesture_start, gesture_end)

    print("Bearings: {}".format(bearings))
    print("Distance changes: {}".format(distance_changes))
    # Get anchor with minimum bearing
    anchor_min_bearing = min(bearings, key=bearings.get)

    # Get anchor with minimum distance change (works because decrease is negative)
    anchor_min_distance_change = min(distance_changes, key=distance_changes.get)
    plot_distance_change(gesture_start, gesture_end, anchor_min_distance_change)

    if anchor_min_bearing == anchor_min_distance_change:
        print("SUCCESS. CONCURRING OPINIONS.")
        print(f"Selected Target: {anchor_min_bearing}")
    else:
        print("DISAGREEMENT. SELECTING BEST SCORING ANCHOR.")
        anchor = get_best_scoring_anchor(distance_changes, bearings, method="Ole")
        print(f"Selected Target: {anchor}")


def plot_distance_change(gesture_start, gesture_end, anchor_id):
    conn = sqlite3.connect("assets/MODI.db", check_same_thread=False)
    cur = conn.cursor()
    plot_start = gesture_start - 2e+9
    plot_end = gesture_end + 2e+9
    
    # Query position data for the tag
    cur.execute("""SELECT timestamp, est_position_x, est_position_y FROM location_data 
                   WHERE timestamp > ? AND timestamp < ? 
                   AND est_position_x IS NOT NULL 
                   AND est_position_y IS NOT NULL 
                   ORDER BY timestamp ASC""", (plot_start, plot_end))
    data = cur.fetchall()
    conn.close()
    
    
    # Load anchor configuration to get anchor position
    with open("assets/anchor_config.json", "r") as f:
        anchor_config = json.load(f)
    
    # Find the specific anchor position
    anchor_position = None
    for anchor in anchor_config:
        if anchor["id"] == anchor_id:
            anchor_position = np.array([anchor["x"], anchor["y"]])
            break

    
    # Extract timestamps and calculate distances
    timestamps = [row[0] for row in data]
    distances = []
    
    for row in data:
        tag_position = np.array([row[1], row[2]])  # est_position_x, est_position_y
        distance = np.linalg.norm(anchor_position - tag_position)
        distances.append(distance)
    
    # Convert timestamps to relative time (in seconds) for better readability
    timestamps_relative = [(ts - timestamps[0]) / 1e9 for ts in timestamps]
    gesture_start_relative = (gesture_start - timestamps[0]) / 1e9
    gesture_end_relative = (gesture_end - timestamps[0]) / 1e9
    
    # Calculate distance changes relative to the first measurement
    initial_distance = distances[0]
    distance_changes = [(d - initial_distance) for d in distances]
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps_relative, distance_changes, 'b-', linewidth=1, label=f'Distance Change (Anchor {anchor_id})')
    
    # Add vertical lines for gesture start and end
    plt.axvline(x=gesture_start_relative, color='green', linestyle='--', linewidth=2, label='Gesture Start')
    plt.axvline(x=gesture_end_relative, color='red', linestyle='--', linewidth=2, label='Gesture End')
    
    # Add horizontal line at y=0 for reference
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    
    # Customize the plot
    plt.xlabel('Time (seconds)')
    plt.ylabel('Distance Change (meters)')
    plt.title(f'Distance Change Over Time - Anchor {anchor_id} (Calculated from X,Y)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Show the plot
    plt.show()
    
    return data

def read_anchor_config():
    with open("assets/anchor_config.json", "r") as f:
        config = json.load(f)
    anchors = {}
    for anchor in config:
        anchors[anchor["id"]] = np.array([anchor["x"], anchor["y"]])
    return anchors


# First ever position of the tag, this is where we calibrated the tag to
def get_initial_position():
    conn = sqlite3.connect("assets/MODI.db", check_same_thread=False)
    initial_position = conn.execute(
        """SELECT est_position_x, est_position_y FROM location_data WHERE est_position_x IS NOT NULL AND est_position_y IS NOT NULL ORDER BY timestamp ASC LIMIT 1"""
    ).fetchone()
    conn.close()
    return np.array(initial_position)


# Get last known position of the tag before the gesture started
def get_current_position(gesture_start):
    conn = sqlite3.connect("assets/MODI.db", check_same_thread=False)
    current_position = conn.execute(
        """SELECT est_position_x, est_position_y FROM location_data WHERE timestamp < ? AND est_position_x IS NOT NULL AND est_position_y IS NOT NULL ORDER BY timestamp DESC LIMIT 1""",
        (gesture_start,),
    ).fetchone()
    conn.close()
    return np.array(current_position)


if __name__ == "__main__":
    anchors = read_anchor_config()
    print(anchors)
