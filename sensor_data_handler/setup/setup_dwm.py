from bleak import BleakClient
import json, struct, asyncio
import time

# UUID of where we write the anchor position
ANCHOR_POSITION_CHARACTERISTIC_UUID = "f0f26c9b-2c8c-49ac-ab60-fe03def1b40c"

# Read from the anchor configuration file
def read_anchor_config():
    with open("assets/anchor_config.json", "r") as f:
        config = json.load(f)

    return config

# Write the anchor positions to the anchors
async def write_anchor_positions(anchor):
    async with BleakClient(anchor["ble_address"]) as client:
            await client.write_gatt_char(
                ANCHOR_POSITION_CHARACTERISTIC_UUID,
                struct.pack("<fffB", anchor["x"], anchor["y"], anchor["z"], 100)
            )

# Wrapper function to write positions to all anchors
def setup_dwm():
    anchors = read_anchor_config()

    async def run_all():
        for anchor in anchors:
            await write_anchor_positions(anchor)

    asyncio.run(run_all())