from selection_manager import select_target
import time
import sqlite3


def connect():
    conn = sqlite3.connect("test_data.db")
    return conn


def monitor_gesture():
    conn = connect()
    while True:
        if check_last_axis_acceleration(conn, "z"):
            gesture_end = time.time_ns()
            cur = conn.cursor()
            gesture_start = cur.execute(
                """SELECT timestamp FROM accel_data WHERE abs(z) < 0.2 AND abs(x) > 0.9 ORDER BY timestamp DESC LIMIT 1"""
            ).fetchone()[0]
            print("Gesture recognized.")
            conn.close()
            select_target(gesture_start, gesture_end)
            break

    monitor_arm_down()
            

def monitor_arm_down():
    conn = connect()
    while True:
        if check_last_axis_acceleration(conn, "x"):
            print("Arm down gesture recognized.")
            conn.close()
            break

    monitor_gesture()


def check_last_axis_acceleration(conn, axis):
    cur = conn.cursor()
    last_axis_accelerations = cur.execute("""SELECT abs(?)
                                          FROM accel_data
                                          ORDER BY timestamp DESC
                                          LIMIT 10""", (axis,),).fetchall()
    if len(last_axis_accelerations) > 0:
        for a in last_axis_accelerations:
            print(a[0])
            if a[0] > 1.05 or a[0] < 0.9:
                return False
        return True
    else:
        return False
    