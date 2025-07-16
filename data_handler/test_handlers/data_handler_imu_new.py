import asyncio
from bleak import BleakClient, BleakScanner
import struct
import sqlite3
import time
from queue import Queue
from threading import Thread
from collections import deque

DEVICE_NAME = "MODI_SW_IMU"
GYRO_SERVICE_UUID = "09451b74-8500-4b3d-9090-bdf3187a98dd"
ACCEL_SERVICE_UUID = "cc55d02b-0890-43ff-9c6b-c078d26a7d3f"
ADDRESS = "B0:6D:06:5A:59:D1"

gyro_queue = deque()
accel_queue = deque()


async def gyro_db_worker():
    conn = sqlite3.connect("assets/MODI.db", check_same_thread=False)
    cur = conn.cursor()
    while True:
        if gyro_queue:
            item = gyro_queue.popleft()
            timestamp, x, y, z = item
            try:
                cur.execute(
                    f"INSERT INTO gyro_data (timestamp, x, y, z) VALUES (?, ?, ?, ?)",
                    (timestamp, x, y, z),
                )
                conn.commit()
            except sqlite3.Error as se:
                print(f"Sqlite error in IMU data handler for table gyro_data.")
                print(se)
            except Exception as e:
                print(e)
        await asyncio.sleep(0.001)


async def accel_db_worker():
    conn = sqlite3.connect("assets/MODI.db", check_same_thread=False)
    cur = conn.cursor()
    while True:
        if accel_queue:
            item = accel_queue.popleft()
            timestamp, x, y, z = item
            try:
                cur.execute(
                    f"INSERT INTO accel_data (timestamp, x, y, z) VALUES (?, ?, ?, ?)",
                    (timestamp, x, y, z),
                )
                conn.commit()
            except sqlite3.Error as se:
                print(f"Sqlite error in IMU data handler for table accel_data.")
                print(se)
            except Exception as e:
                print(e)
        await asyncio.sleep(0.001)


def make_gyro_handler():
    def handler(_, data):
        value = struct.unpack("<fff", data)
        gyro_queue.append((time.time_ns(), *value))
        #print("Gyro handler still running")

    return handler


def make_accel_handler():
    def handler(_, data):
        value = struct.unpack("<fff", data)
        accel_queue.append((time.time_ns(), *value))
        #print("Accel handler still running")

    return handler


async def read_data():
    #device = await BleakScanner.find_device_by_address(ADDRESS, timeout=20)
    #if not device:
        #print("IMU not found")
        #return

    async with BleakClient(ADDRESS, timeout=20) as client:
        await client.start_notify(GYRO_SERVICE_UUID, make_gyro_handler())
        await client.start_notify(ACCEL_SERVICE_UUID, make_accel_handler())
        print("IMU found. Listening...")
        asyncio.create_task(gyro_db_worker())
        asyncio.create_task(accel_db_worker())
        await asyncio.Event().wait()  # block forever


def handle_imu_data():
    # Starte BLE-Loop
    asyncio.run(read_data())


if __name__ == "__main__":
    handle_imu_data()
