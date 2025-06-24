import threading
import sqlite3
import time

# This file tests the threading capabilities of Python with SQLite


def connect():
    con = sqlite3.connect("tutorial.db")

    return con


def setup():
    con = connect()
    cur = con.cursor()

    cur.execute("""
                CREATE TABLE IF NOT EXISTS location_data 
                (id INTEGER PRIMARY KEY, 
                timestamp INTEGER,
                anchor_id TEXT,
                position TEXT,
                distance REAL,
                time_taken INTEGER,
                est_position TEXT)
                """)

    # Truncate the table if it exists
    cur.execute("DELETE FROM location_data")

    con.commit()
    con.close()


def insert_data(timestamp, anchor_id, position, distance, time_taken, est_position):
    con = connect()
    cur = con.cursor()

    cur.execute(
        """
                INSERT INTO location_data 
                (timestamp, anchor_id, position, distance, time_taken, est_position) 
                VALUES (?, ?, ?, ?, ?, ?)
                """,
        (timestamp, anchor_id, position, distance, time_taken, est_position),
    )
    con.commit()


def insert_demo_data():
    while True:
        timestamp = round(time.perf_counter_ns() / 1e6)
        anchor_id = "A1"
        position = "[1.0, 2.0, 3.0]"
        distance = 4.5
        time_taken = 100
        est_position = "[1.1, 2.1, 3.1]"

        insert_data(timestamp, anchor_id, position, distance, time_taken, est_position)

        timestamp_after_insert = round(time.perf_counter_ns() / 1e6)
        time.sleep(0.1 - (timestamp_after_insert - timestamp) / 1.0e3)


def read_from_demo_data():
    con = connect()
    cur = con.cursor()

    while True:
        time.sleep(2)
        rows = cur.execute(
            "SELECT timestamp FROM location_data ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()

        current_time = round(time.perf_counter_ns() / 1e6)

        difference = current_time - rows[0] if rows else None
        print(
            f"Current time: {current_time}, Last timestamp in DB: {rows[0] if rows else 'None'}, Difference: {difference} ms"
        )


setup()
threading.Thread(target=insert_demo_data, daemon=False).start()
threading.Thread(target=read_from_demo_data, daemon=False).start()
