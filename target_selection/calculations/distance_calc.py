import sqlite3
import time
import json
import numpy as np


def connect(database_name="MODI"):
    conn = sqlite3.connect(f"assets/{database_name}.db")
    return conn


def get_distance_changes(start, end=0, database_name="MODI"):
    if end == 0:
        end = time.time_ns()
    conn = connect(database_name)
    cur = conn.cursor()
    anchor_ids = cur.execute("SELECT DISTINCT anchor_id FROM location_data").fetchall()
    anchor_ids = [id[0] for id in anchor_ids]
    distance_changes = {}
    for anchor_id in anchor_ids:
        start_distance = cur.execute(
            """SELECT distance
                                        FROM location_data
                                        WHERE timestamp > ?
                                        AND anchor_id = ?
                                        ORDER BY timestamp
                                        LIMIT 1""",
            (
                start,
                anchor_id,
            ),
        ).fetchone()
        end_distance = cur.execute(
            """SELECT distance
                                        FROM location_data
                                        WHERE timestamp < ?
                                        AND anchor_id = ?
                                        ORDER BY timestamp DESC
                                        LIMIT 1""",
            (
                end,
                anchor_id,
            ),
        ).fetchone()
        delta_distance = end_distance[0] - start_distance[0]
        distance_change = delta_distance / start_distance[0]
        distance_changes[anchor_id] = distance_change
    conn.close()
    return distance_changes


def get_distance_changesv2(start, end=0, database_name="MODI"):
    if end == 0:
        end = time.time_ns()

    conn = connect(database_name)
    cur = conn.cursor()
    start_position = cur.execute(
        """SELECT est_position_x, est_position_y
               FROM location_data
               WHERE timestamp > ?
               AND est_position_x IS NOT NULL
               AND est_position_y IS NOT NULL
               ORDER BY timestamp
               LIMIT 1""",
        (start,),
    ).fetchone()
    
     # This can throw errors, this means we did a pointing gesture outside of the triangulation area
    if start_position is None:
        print("No position found after start timestamp, using last known position before start.")
        # Get the first position before timestamp
        start_position = cur.execute(
            """SELECT est_position_x, est_position_y
               FROM location_data
               WHERE timestamp < ?
               AND est_position_x IS NOT NULL
               AND est_position_y IS NOT NULL
               ORDER BY timestamp DESC
               LIMIT 1""",
            (start,),
        ).fetchone()

    start_position = np.array([start_position[0], start_position[1]])
    end_position = cur.execute(
        """SELECT est_position_x, est_position_y
               FROM location_data
               WHERE timestamp < ?
               AND est_position_x IS NOT NULL
               AND est_position_y IS NOT NULL
               ORDER BY timestamp DESC
               LIMIT 1""",
        (end,),
    ).fetchone()
    conn.close()

    end_position = np.array([end_position[0], end_position[1]])
    with open("assets/anchor_config.json", "r") as f:
        anchors = json.load(f)
    distance_changes = {}
    for anchor in anchors:
        anchor_position = np.array([anchor["x"], anchor["y"]])
        start_distance = np.linalg.norm(anchor_position - start_position)
        end_distance = np.linalg.norm(anchor_position - end_position)
        distance_change = (end_distance - start_distance) / float(start_distance)
        distance_changes[anchor["id"]] = distance_change
    return distance_changes


if __name__ == "__main__":
    start = 15867624888400
    end = 15872223826700
    distance_changes = get_distance_changes(start, end)
    min_anchor = min(distance_changes, key=distance_changes.get)
    print(min_anchor)
    min_distance = distance_changes[min_anchor]
    print(min_distance)
