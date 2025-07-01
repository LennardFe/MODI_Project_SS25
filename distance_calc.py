import sqlite3
import time

def connect():
    conn = sqlite3.connect("test_data.db")
    return conn

def distance_calc(start, end=0):
    if end == 0:
        end = time.perf_counter_ns()
    conn = connect()
    cur = conn.cursor()
    anchor_ids = cur.execute('SELECT DISTINCT anchor_id FROM location_data').fetchall()
    anchor_ids = [id[0] for id in anchor_ids]
    print(anchor_ids)
    distance_changes = {}
    for anchor_id in anchor_ids:
        start_distance = cur.execute("""SELECT distance
                                        FROM location_data
                                        WHERE timestamp > ?
                                        AND anchor_id = ?
                                        ORDER BY timestamp
                                        LIMIT 1""", (start, anchor_id,)).fetchone()
        end_distance = cur.execute("""SELECT distance
                                        FROM location_data
                                        WHERE timestamp < ?
                                        AND anchor_id = ?
                                        ORDER BY timestamp DESC
                                        LIMIT 1""", (end, anchor_id,)).fetchone()
        delta_distance = end_distance[0] - start_distance[0]
        distance_change = delta_distance / start_distance[0]
        distance_changes[anchor_id] = distance_change
        print(delta_distance)
    return distance_changes

def main():
    start = 15867624888400
    end = 15872223826700
    distance_changes = distance_calc(start, end)
    min_anchor = min(distance_changes, key=distance_changes.get)
    min_distance = distance_changes[min_anchor]
    print(min_distance)

start_time = time.perf_counter_ns()
main()
print((time.perf_counter_ns() - start_time)/1.0e6)