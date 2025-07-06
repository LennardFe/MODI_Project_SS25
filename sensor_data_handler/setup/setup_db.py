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
                    id              INTEGER PRIMARY KEY,
                    timestamp       INTEGER,
                    anchor_id       TEXT,
                    distance        INTEGER,
                    distance_qf     INTEGER,
                    est_position_x  REAL,
                    est_position_y  REAL,
                    est_position_z  REAL,
                    est_position_qf INTEGER
                )
                """)
    conn.commit()
    conn.close()