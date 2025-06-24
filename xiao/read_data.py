import asyncio
from bleak import BleakClient, BleakScanner
import struct
import sqlite3
import time

DEVICE_NAME = "MODI_SW_IMU"
GYRO_SERVICE_UUID = "09451b74-8500-4b3d-9090-bdf3187a98dd"
ACCEL_SERVICE_UUID = "cc55d02b-0890-43ff-9c6b-c078d26a7d3f"


def connect():
    conn = sqlite3.connect("../test_data.db")
    return conn


def setup_db():
    conn = connect()
    cur = conn.cursor()
    cur.execute("""
                        CREATE TABLE IF NOT EXISTS gyro_data
                        (
                            id           INTEGER PRIMARY KEY,
                            timestamp    INTEGER,
                            x            REAL,
                            y            REAL,
                            z            REAL
                        )
                        """)
    # Truncate the table if it exists
    cur.execute("DELETE FROM gyro_data")
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
    # Truncate the table if it exists
    cur.execute("DELETE FROM accel_data")
    conn.commit()
    conn.close()


def insert_data(table, timestamp, x, y, z):
    conn = connect()
    cur = conn.cursor()
    if str(table).lower() == "gyro":
        cur.execute(
            """
                    INSERT INTO gyro_data
                    (timestamp, x, y, z) 
                    VALUES (?, ?, ?, ?)
                    """,
            (timestamp, x, y, z),
        )
    elif str(table).lower() == "accel":
        cur.execute(
            """
            INSERT INTO accel_data
                (timestamp, x, y, z)
            VALUES (?, ?, ?, ?)
            """,
            (timestamp, x, y, z),
        )
    conn.commit()
    conn.close()


def make_handler(type):
    def handler(sender, data):
        value = struct.unpack("<fff", data)
        insert_data(type, time.perf_counter_ns(),*value)
        print(f"{type}: {value}")

    return handler


async def main():
    setup_db()
    device = await BleakScanner.find_device_by_name(DEVICE_NAME)

    async with BleakClient(device) as client:
        await client.start_notify(GYRO_SERVICE_UUID, make_handler("Gyro"))
        await client.start_notify(ACCEL_SERVICE_UUID, make_handler("Accel"))

        await asyncio.Event().wait()


asyncio.run(main())
