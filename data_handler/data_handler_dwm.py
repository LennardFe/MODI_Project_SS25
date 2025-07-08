import asyncio
from bleak import BleakClient
import struct
from queue import Queue
import sqlite3
from threading import Thread
import time

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

        timestamp, anchor_id,\
        distance, distance_qf, est_position_x,\
        est_position_y, est_position_z, est_position_qf = item

        cur.execute(
            """INSERT INTO location_data (timestamp, anchor_id, 
                                        distance, distance_qf, est_position_x, 
                                        est_position_y, est_position_z, est_position_qf) 
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (timestamp, anchor_id, 
             distance, distance_qf, est_position_x, 
             est_position_y, est_position_z, est_position_qf),
        )
        conn.commit()

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
    print(f"Datatype: {data_type}")

    # Position only
    if data_type == 0:  
        x, y, z, est_pos_qf = struct.unpack_from("<iiiB", data, offset=1)
        db_queue.put((timestamp, None, None, None, x, y, z, est_pos_qf))

        return

    # Distances only
    elif data_type == 1: 
        count = data[1]   # Count = Amount of anchors
        offset = 2

        # Iterate over all anchors
        for _ in range(count):

            anchor_id, distance, distance_qf = struct.unpack_from("<HIB", data, offset)

            # Convert anchor_id to hex string
            anchor_id = f"{anchor_id:04x}".upper()

            db_queue.put((timestamp, anchor_id, distance, distance_qf, None, None, None, None))

            # Offset for the next anchor
            offset += 7

        return

    # Position + Distances
    elif data_type == 2:  
        x, y, z, est_pos_qf = struct.unpack_from("<iiiB", data, offset=1)
        count = data[14] # Count = Amount of anchors (Another byte position in this case)         
        offset = 15
        
        for _ in range(count):
            anchor_id, distance, distance_qf = struct.unpack_from("<HIB", data, offset)
            
            # Convert anchor_id to hex string
            anchor_id = f"{anchor_id:04x}".upper()

            db_queue.put((timestamp, anchor_id, distance, distance_qf, x, y, z, est_pos_qf))

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
async def read_data():
    async with BleakClient(ADDRESS, timeout=20) as client:
        print("Tag found. Listening...")
        await client.start_notify(LOCATION_UUID, make_location_handler())

        await asyncio.Event().wait()  # block forever until the program is terminated

# Handle UWB data processing
def handle_uwb_data():
    # Read the data from the queue and write into database
    Thread(target=db_worker, daemon=True).start()

    # Read data from the DWM Tag and write into the queue
    asyncio.run(read_data())


if __name__ == "__main__":
    handle_uwb_data()