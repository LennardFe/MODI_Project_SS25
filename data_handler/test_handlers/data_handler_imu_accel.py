import asyncio
from bleak import BleakClient, BleakScanner
import struct
import sqlite3
import time
from queue import Queue
from threading import Thread

DEVICE_NAME = "MODI_SW_IMU"
ACCEL_SERVICE_UUID = "cc55d02b-0890-43ff-9c6b-c078d26a7d3f"
ADDRESS = "B0:6D:06:5A:59:D1"

db_queue = Queue()


def db_worker():
    conn = sqlite3.connect("assets/MODI.db", check_same_thread=False)
    cur = conn.cursor()
    while True:
        item = db_queue.get()
        if item is None:
            break
        table, timestamp, x, y, z = item
        try:
            cur.execute(
                f"INSERT INTO {table} (timestamp, x, y, z) VALUES (?, ?, ?, ?)",
                (timestamp, x, y, z),
            )
            conn.commit()
        except sqlite3.Error as se:
            print(f"Sqlite error in IMU data handler for table '{table}'.")
            print(se)
        except Exception as e:
            print(e)

    conn.close()


def make_accel_handler():
    def handler(_, data):
        value = struct.unpack("<fff", data)
        db_queue.put(("accel_data", time.time_ns(), *value))

    return handler


async def read_accel_data():
    async with BleakClient(ADDRESS) as client:
        await client.start_notify(ACCEL_SERVICE_UUID, make_accel_handler())
        print("Accelerometer listener running...")
        await asyncio.Event().wait()  # Block forever


def handle_accel():
    Thread(target=db_worker, daemon=True).start()
    asyncio.run(read_accel_data())


if __name__ == "__main__":
    handle_accel()
