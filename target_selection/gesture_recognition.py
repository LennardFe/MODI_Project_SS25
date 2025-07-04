from selection_manager import select_target
import time
import sqlite3


def connect():
    conn = sqlite3.connect("test_data.db")
    return conn


def monitor_gesture():
    conn = connect()
    while True:
        if check_last_z_accelerations(conn):
            gesture_end = time.time_ns()
            cur = conn.cursor()
            cur.execute(
                """SELECT timestamp FROM accel_data WHERE abs(z) < 0.9 AND abs(x) > 0.9 ORDER BY timestamp DESC LIMIT 1"""
            )
            gesture_start = cur.execute(
                """SELECT timestamp FROM accel_data WHERE abs(z) < 0.9 AND abs(x) > 0.9 ORDER BY timestamp DESC LIMIT 1"""
            ).fetchone()[0]
            print("Gesture recognized.")
            select_target(gesture_start, gesture_end)
            conn.close()
            break


def check_last_z_accelerations(conn):
    cur = conn.cursor()
    last_z_accelerations = cur.execute("""SELECT abs(z)
                                          FROM accel_data
                                          ORDER BY timestamp DESC
                                          LIMIT 10""").fetchall()
    if len(last_z_accelerations) > 0:
        for z in last_z_accelerations:
            print(z[0])
            if z[0] > 1.05 or z[0] < 0.9:
                return False
        print("Gesture recognized.")
        return True
    else:
        return False
