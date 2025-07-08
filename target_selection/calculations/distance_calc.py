import sqlite3
import time


def connect():
    conn = sqlite3.connect("assets/MODI.db")
    return conn


def get_distance_changes(start, end=0):
    if end == 0:
        end = time.time_ns()
    conn = connect()
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
    return distance_changes


if __name__ == "__main__":
    start = 15867624888400
    end = 15872223826700
    distance_changes = get_distance_changes(start, end)
    min_anchor = min(distance_changes, key=distance_changes.get)
    print(min_anchor)
    min_distance = distance_changes[min_anchor]
    print(min_distance)
