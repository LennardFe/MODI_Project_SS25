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
ADDRESS = "B0:6D:06:5A:59:D1"

db_queue = Queue()

# Shared timestamps for watchdog
last_gyro_time = time.time()
last_accel_time = time.time()
WATCHDOG_TIMEOUT = 1  # seconds


def db_worker():
    conn = sqlite3.connect("assets/MODI.db", check_same_thread=False)
    cur = conn.cursor()
    while True:
        item = db_queue.get()
        if item is None:
            print("DB Worker exiting")
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
        global last_gyro_time
        value = struct.unpack("<fff", data)
        last_gyro_time = time.time()
        db_queue.put(("gyro_data", time.time_ns(), *value))
    return handler


def make_accel_handler():
    def handler(_, data):
        global last_accel_time
        value = struct.unpack("<fff", data)
        last_accel_time = time.time()
        db_queue.put(("accel_data", time.time_ns(), *value))
    return handler


async def watchdog():
    while True:
        await asyncio.sleep(1)
        now = time.time()
        if now - last_gyro_time > WATCHDOG_TIMEOUT:
            print(f"Warning: No GYRO data received in the last {WATCHDOG_TIMEOUT} seconds.")
        if now - last_accel_time > WATCHDOG_TIMEOUT:
            print(f"Warning: No ACCEL data received in the last {WATCHDOG_TIMEOUT} seconds.")


async def read_data():
    device = await BleakScanner.find_device_by_address(ADDRESS, timeout=20)
    if not device:
        print("IMU not found")
        return

    async with BleakClient(device) as client:
        try:
            await asyncio.sleep(0.5)
            await client.start_notify(GYRO_SERVICE_UUID, make_gyro_handler())
            await client.start_notify(ACCEL_SERVICE_UUID, make_accel_handler())
            print("IMU found. Listening...")

            # Run watchdog in background
            await asyncio.gather(
                watchdog(),
                asyncio.Event().wait()  # keep running forever
            )
        except Exception as e:
            print("Could not start: " + str(e))

def handle_imu_data():
    Thread(target=db_worker, daemon=False).start()
    asyncio.run(read_data())


if __name__ == "__main__":
    handle_imu_data()