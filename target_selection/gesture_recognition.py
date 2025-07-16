from target_selection.selection_manager import select_target
from kivy.clock import Clock
import time, sqlite3, os

# Connect to the SQLite database
def connect(database_name="MODI"):
    conn = sqlite3.connect(f"assets/{database_name}.db", check_same_thread=False)
    return conn

# For live top down visualization
def save_gesture_state(state):
    try:
        os.makedirs("plots", exist_ok=True)
        with open("plots/gesture_state.txt", "w") as f:
            f.write(f"{time.time_ns()},{state}")
    except Exception as e:
        print(f"Error saving gesture state: {e}")

# Monitor the gesture, if motion is detected (via db call) get data from db and call select target
def monitor_gesture(CALIBRATION_ANCHOR, kivy_instance, database_name="MODI"):
    conn = connect(database_name)
    while True:
        if check_last_axis_acceleration(
            conn,
            "z",
        ):
            save_gesture_state("ARM_UP")
            gesture_end = time.time_ns()
            cur = conn.cursor()
            gesture_start = cur.execute(
                """SELECT timestamp FROM accel_data WHERE abs(z) < 0.2 AND abs(x) > 0.9 ORDER BY timestamp DESC LIMIT 1"""
            ).fetchone()[0]
            select_target(gesture_start, gesture_end, CALIBRATION_ANCHOR, kivy_instance, database_name)
            break

    monitor_arm_down(CALIBRATION_ANCHOR, kivy_instance, database_name)

# If gesture was monitored, then monitor when arm down gesture is performed and jump back to monitor_gesture
def monitor_arm_down(CALIBRATION_ANCHOR, kivy_instance, database_name="MODI"):
    conn = connect(database_name)
    while True:
        if check_last_axis_acceleration(conn, "x"):
            save_gesture_state("ARM_DOWN")
            print("Arm down gesture recognized.")
            if kivy_instance is not None:
                Clock.schedule_once(lambda _: kivy_instance.set_all_off(), 0)
            break

    monitor_gesture(CALIBRATION_ANCHOR, kivy_instance, database_name)

# Function to check if the arm movement was enough to be considered a gesture
def check_last_axis_acceleration(conn, axis):
    cur = conn.cursor()
    if axis == "x":
        last_axis_accelerations = cur.execute("""SELECT abs(x)
                                              FROM accel_data
                                              ORDER BY timestamp DESC
                                              LIMIT 10""").fetchall()
    else:
        last_axis_accelerations = cur.execute("""SELECT abs(z)
                                                 FROM accel_data
                                                 ORDER BY timestamp DESC
                                                 LIMIT 10""").fetchall()

    if len(last_axis_accelerations) > 0:
        for a in last_axis_accelerations:
            if a[0] > 1.1 or a[0] < 0.9:
                return False
        return True
    else:
        return False

# Test function
if __name__ == "__main__":
    monitor_gesture("5C19")