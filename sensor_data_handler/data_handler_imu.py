import asyncio
from bleak import BleakClient, BleakScanner
import struct
import sqlite3
import time
from queue import Queue
from threading import Thread

DEVICE_NAME = "MODI_SW_IMU"
GYRO_SERVICE_UUID = "09451b74-8500-4b3d-9090-bdf3187a98dd"
ACCEL_SERVICE_UUID = "cc55d02b-0890-43ff-9c6b-c078d26a7d3f"

db_queue = Queue()


def setup_db():
    conn = sqlite3.connect("test_data.db")
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS gyro_data (
            id INTEGER PRIMARY KEY,
            timestamp INTEGER,
            x REAL,
            y REAL,
            z REAL
        )
    """)
    cur.execute("DELETE FROM gyro_data")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS accel_data (
            id INTEGER PRIMARY KEY,
            timestamp INTEGER,
            x REAL,
            y REAL,
            z REAL
        )
    """)
    cur.execute("DELETE FROM accel_data")
    conn.commit()
    conn.close()


def db_worker():
    conn = sqlite3.connect("test_data.db")
    cur = conn.cursor()
    while True:
        item = db_queue.get()
        if item is None:
            break
        table, timestamp, x, y, z = item
        cur.execute(
            f"INSERT INTO {table} (timestamp, x, y, z) VALUES (?, ?, ?, ?)",
            (timestamp, x, y, z),
        )
        conn.commit()
    conn.close()


def make_gyro_handler():
    def handler(sender, data):
        value = struct.unpack("<fff", data)
        db_queue.put(("gyro_data", time.time_ns(), *value))

    return handler


def make_accel_handler():
    def handler(sender, data):
        value = struct.unpack("<fff", data)
        db_queue.put(("accel_data", time.time_ns(), *value))

    return handler


async def read_data():
    setup_db()
    device = await BleakScanner.find_device_by_address("B0:6D:06:5A:59:D1")
    if not device:
        print("Device not found")
        return

    async with BleakClient(device) as client:
        await client.start_notify(GYRO_SERVICE_UUID, make_gyro_handler())
        await client.start_notify(ACCEL_SERVICE_UUID, make_accel_handler())
        print("Notifications started. Listening...")

        await asyncio.Event().wait()  # block forever


def handle_imu_data():
    # Starte DB-Worker-Thread
    Thread(target=db_worker, daemon=True).start()

    # Starte BLE-Loop
    asyncio.run(read_data())


if __name__ == "__main__":
    handle_imu_data()
