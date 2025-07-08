from target_selection.selection_manager import select_target
import time
import sqlite3


def connect():
    conn = sqlite3.connect("assets/MODI.db", check_same_thread=False)
    return conn


def monitor_gesture(CALIBRATION_ANCHOR):
    conn = connect()
    while True:
        if check_last_axis_acceleration(conn, "z"):
            gesture_end = time.time_ns()
            cur = conn.cursor()
            gesture_start = cur.execute(
                """SELECT timestamp FROM accel_data WHERE abs(z) < 0.2 AND abs(x) > 0.9 ORDER BY timestamp DESC LIMIT 1"""
            ).fetchone()[0]

            diff_in_milliseconds = gesture_end - gesture_start
            diff_in_seconds = diff_in_milliseconds / 1000
            print(f"Gesture recognized. Difference between start and end in seconds: {diff_in_seconds} s.")
            conn.close()

            select_target(gesture_start, gesture_end, CALIBRATION_ANCHOR)
            break

    monitor_arm_down(CALIBRATION_ANCHOR)
            

def monitor_arm_down(CALIBRATION_ANCHOR):
    conn = connect()
    while True:
        if check_last_axis_acceleration(conn, "x"):
            print("Arm down gesture recognized.")
            conn.close()
            break

    monitor_gesture(CALIBRATION_ANCHOR)


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

if __name__ == "__main__":
    monitor_gesture()