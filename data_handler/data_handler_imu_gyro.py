import asyncio
from bleak import BleakClient, BleakScanner
import struct
import sqlite3
import time
from queue import Queue
from threading import Thread

DEVICE_NAME = "MODI_SW_IMU"
GYRO_SERVICE_UUID = "09451b74-8500-4b3d-9090-bdf3187a98dd"
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


def make_gyro_handler():
    def handler(_, data):
        value = struct.unpack("<fff", data)
        db_queue.put(("gyro_data", time.time_ns(), *value))

    return handler


async def read_gyro_data():
    device = await BleakScanner.find_device_by_address(ADDRESS, timeout=20)
    if not device:
        print("IMU not found")
        return

    async with BleakClient(device) as client:
        await client.start_notify(GYRO_SERVICE_UUID, make_gyro_handler())
        print("Gyroscope listener running...")
        await asyncio.Event().wait()  # Block forever


def handle_gyro():
    Thread(target=db_worker, daemon=True).start()
    asyncio.run(read_gyro_data())


if __name__ == "__main__":
    handle_gyro()
