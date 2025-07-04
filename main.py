from target_selection.gesture_recognition import monitor_gesture
from sensor_data_handler.data_handler_imu import handle_imu_data
from sensor_data_handler.data_handler_dwm import handle_uwb_data
from threading import Thread
import time
import sqlite3


def setup_db():
    conn = sqlite3.connect("assets/MODI.db")
    cur = conn.cursor()
    cur.execute("""DROP TABLE IF EXISTS gyro_data""")
    cur.execute("""DROP TABLE IF EXISTS accel_data""")
    cur.execute("""DROP TABLE IF EXISTS location_data""")

    cur.execute("""
                CREATE TABLE IF NOT EXISTS gyro_data
                (
                    id        INTEGER PRIMARY KEY,
                    timestamp INTEGER,
                    x         REAL,
                    y         REAL,
                    z         REAL
                )
                """)
    cur.execute("""
                CREATE TABLE IF NOT EXISTS accel_data
                (
                    id        INTEGER PRIMARY KEY,
                    timestamp INTEGER,
                    x         REAL,
                    y         REAL,
                    z         REAL
                )
                """)
    cur.execute("""
                CREATE TABLE IF NOT EXISTS location_data
                (
                    id           INTEGER PRIMARY KEY,
                    timestamp    INTEGER,
                    anchor_id    TEXT,
                    position     TEXT,
                    distance     REAL,
                    time_taken   INTEGER,
                    est_position TEXT
                )
                """)
    conn.commit()
    conn.close()


Thread(target=handle_imu_data).start()
#Thread(target=handle_uwb_data).start()
