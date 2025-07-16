from bleak import BleakClient, BleakScanner
import asyncio, struct, sqlite3, time
from threading import Thread
from kivy.clock import Clock
from queue import Queue

# BLE Address for the DWM Tag
ADDRESS = "CC:49:99:EE:1A:F4"

# UUID for the Location service
LOCATION_UUID = "003bbdf2-c634-4b3d-ab56-7ec889b89a37"

# Queue for database operations
db_queue = Queue()

# Read newest data from the queue and write into the database
def db_worker():
    conn = sqlite3.connect("assets/MODI.db", check_same_thread=False)
    cur = conn.cursor()

    while True:
        item = db_queue.get()
        if item is None:
            break

        (
            timestamp,
            anchor_id,
            distance,
            distance_qf,
            est_position_x,
            est_position_y,
            est_position_z,
            est_position_qf,
        ) = item

        try:
            cur.execute(
                """INSERT INTO location_data (timestamp, anchor_id, 
                                            distance, distance_qf, est_position_x, 
                                            est_position_y, est_position_z, est_position_qf) 
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    timestamp,
                    anchor_id,
                    distance,
                    distance_qf,
                    est_position_x,
                    est_position_y,
                    est_position_z,
                    est_position_qf,
                ),
            )
            conn.commit()

        except sqlite3.Error as se:
            print("Sqlite error in DWM data handler.")
            print(se)
        except Exception as e:
            print(f"Error in DWM data handler: {e}")

    conn.close()

# Parse the location data from the DWM Tag and add to queue
def parse_location_data(data: bytearray):
    # Get current timestamp in nanoseconds
    timestamp = time.time_ns()

    # Check if data is empty
    if len(data) == 0:
        print("No data received")
        return

    # First byte indicates the data type
    data_type = data[0]

    # Position only
    if data_type == 0:
        x, y, z, est_pos_qf = struct.unpack_from("<iiiB", data, offset=1)
        db_queue.put((timestamp, None, None, None, x, y, z, est_pos_qf))

        return

    # Distances only
    elif data_type == 1:
        count = data[1]  # Count = Amount of anchors
        offset = 2

        # Iterate over all anchors
        for _ in range(count):
            anchor_id, distance, distance_qf = struct.unpack_from("<HIB", data, offset)

            # Convert anchor_id to hex string
            anchor_id = f"{anchor_id:04x}".upper()

            db_queue.put(
                (timestamp, anchor_id, distance, distance_qf, None, None, None, None)
            )

            # Offset for the next anchor
            offset += 7

        return

    # Position + Distances
    elif data_type == 2:
        x, y, z, est_pos_qf = struct.unpack_from("<iiiB", data, offset=1)
        count = data[
            14
        ]  # Count = Amount of anchors (Another byte position in this case)
        offset = 15

        for _ in range(count):
            anchor_id, distance, distance_qf = struct.unpack_from("<HIB", data, offset)

            # Convert anchor_id to hex string
            anchor_id = f"{anchor_id:04x}".upper()

            db_queue.put(
                (timestamp, anchor_id, distance, distance_qf, x, y, z, est_pos_qf)
            )

            # Offset for the next anchor
            offset += 7

        return

    else:
        print("Unknown data type")

# Create a notification handler for the location data
def make_location_handler():
    def handler(_, data):
        parse_location_data(data)

    return handler

# Read data from the DWM Tag and handle notifications
async def read_data(kivy_instance):
    device = await BleakScanner.find_device_by_address(ADDRESS, timeout=20)
    if not device:
        print("DWM Tag not found")
        return
    
    async with BleakClient(device) as client:
        try:
            await asyncio.sleep(0.5) # Wait x seconds for the device to be ready
            await client.start_notify(LOCATION_UUID, make_location_handler())

            # Print and set label that the tag is found
            print("Tag found. Listening...")
            if kivy_instance is not None:
                Clock.schedule_once(lambda _: kivy_instance.set_dwm_found(), 0)

            await asyncio.Event().wait()  # block forever until the program is terminated

        except Exception as e:
            print("Could not start: " + str(e))

# Handle UWB data processing
def handle_uwb_data(kivy_instance):
    Thread(target=db_worker, daemon=True).start()
    asyncio.run(read_data(kivy_instance))

# Test function
if __name__ == "__main__":
    handle_uwb_data()