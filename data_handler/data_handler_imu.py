from bleak import BleakClient, BleakScanner
import asyncio, struct, sqlite3, time
from threading import Thread
from kivy.clock import Clock
from queue import Queue

# Constants for the MODI IMU device
DEVICE_NAME = "MODI_SW_IMU"
GYRO_SERVICE_UUID = "09451b74-8500-4b3d-9090-bdf3187a98dd"
ACCEL_SERVICE_UUID = "cc55d02b-0890-43ff-9c6b-c078d26a7d3f"
ADDRESS = "B0:6D:06:5A:59:D1"

# Queue for database operations
db_queue = Queue()

# Shared timestamps for watchdog
last_gyro_time = time.time()
last_accel_time = time.time()
WATCHDOG_TIMEOUT = 1  # seconds

# Function to handle incoming IMU data and store it in the database
def db_worker(db_name):
    conn = sqlite3.connect(f"assets/{db_name}.db", check_same_thread=False)
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

# Function to create a handler for GYRO data notifications
def make_gyro_handler():
    def handler(_, data):
        global last_gyro_time
        value = struct.unpack("<fff", data)
        last_gyro_time = time.time()
        db_queue.put(("gyro_data", time.time_ns(), *value))
    return handler

# Function to create a handler for ACCEL data notifications
def make_accel_handler():
    def handler(_, data):
        global last_accel_time
        value = struct.unpack("<fff", data)
        last_accel_time = time.time()
        db_queue.put(("accel_data", time.time_ns(), *value))
    return handler

# Function to monitor the IMU data and print warnings if no data is received within the timeout period
async def watchdog(kivy_instance):
    global last_gyro_time, last_accel_time
    while True:
        await asyncio.sleep(1)
        now = time.time()
        if now - last_gyro_time > WATCHDOG_TIMEOUT:
            print(f"Warning: No GYRO data received in the last {WATCHDOG_TIMEOUT} seconds.")
            Clock.schedule_once(lambda _: kivy_instance.set_error("No Accel and/or Gyro data!"), 0)

            last_gyro_time = now  # Reset last_gyro_time to avoid repeated warnings

        if now - last_accel_time > WATCHDOG_TIMEOUT:
            print(f"Warning: No ACCEL data received in the last {WATCHDOG_TIMEOUT} seconds.")
            Clock.schedule_once(lambda _: kivy_instance.set_error("No Accel and/or Gyro data!"), 0)

            last_accel_time = now  # Reset last_accel_time to avoid repeated warnings

# TODO: Add reconnect logic to handle disconnections gracefully
# Function to start notifiers for the different IMU services
async def read_data(kivy_instance):
    device = await BleakScanner.find_device_by_address(ADDRESS, timeout=20)
    if not device:
        print("IMU not found")
        return

    async with BleakClient(device) as client:
        try:
            await asyncio.sleep(0.5) # Wait x seconds for the device to be ready
            await client.start_notify(GYRO_SERVICE_UUID, make_gyro_handler())
            await client.start_notify(ACCEL_SERVICE_UUID, make_accel_handler())

            # Print and set label that IMU is found
            print("IMU found. Listening...")
            Clock.schedule_once(lambda _: kivy_instance.set_imu_found(), 0)

            # Run watchdog in background
            await asyncio.gather(
                watchdog(kivy_instance),
                asyncio.Event().wait()  # keep running forever
            )

        except Exception as e:
            print("Could not start: " + str(e))

# TODO: Replace Thread with asyncio task 
# Main function to handle IMU data
def handle_imu_data(kivy_instance, db_name):
    Thread(target=db_worker, args=(db_name,),daemon=True).start()
    asyncio.run(read_data(kivy_instance))

# Test function
if __name__ == "__main__":
    handle_imu_data()