from target_selection.selection_manager import select_target
import time
import sqlite3


def connect(database_name="MODI"):
    conn = sqlite3.connect(f'assets/{database_name}.db', check_same_thread=False)
    return conn


def monitor_gesture(CALIBRATION_ANCHOR, database_name="MODI"):
    while True:
        if check_last_axis_acceleration("z", database_name):
            gesture_end = time.time_ns()
            conn = connect(database_name)
            cur = conn.cursor()
            gesture_start = cur.execute(
                """SELECT timestamp FROM accel_data WHERE abs(z) < 0.2 AND abs(x) > 0.9 ORDER BY timestamp DESC LIMIT 1"""
            ).fetchone()[0]
            conn.close()

            select_target(gesture_start, gesture_end, CALIBRATION_ANCHOR, database_name)
            break

    monitor_arm_down(CALIBRATION_ANCHOR, database_name)


def monitor_arm_down(CALIBRATION_ANCHOR, database_name="MODI"):
    while True:
        if check_last_axis_acceleration("x", database_name):
            print("Arm down gesture recognized.")
            break

    monitor_gesture(CALIBRATION_ANCHOR, database_name)


def check_last_axis_acceleration(axis, database_name="MODI"):
    conn = connect(database_name)
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
    
    conn.close()
    
    if len(last_axis_accelerations) > 0:
        for a in last_axis_accelerations:
            if a[0] > 1.1 or a[0] < 0.9:
                return False
        return True
    else:
        return False


if __name__ == "__main__":
    monitor_gesture("5C19")
