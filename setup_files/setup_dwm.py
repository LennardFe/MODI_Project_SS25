from bleak import BleakClient
import json, struct, asyncio
import time

# UUID of where we write the anchor position
ANCHOR_POSITION_CHARACTERISTIC_UUID = "f0f26c9b-2c8c-49ac-ab60-fe03def1b40c"

# UUID of the characteristic to write the tag location mode
TAG_LOCATION_MODE_CHARACTERISTIC_UUID = "a02b947e-df97-4516-996a-1882521e0ead"

# BLE address of the DWM Tag
TAG_BLE_ADDRESS = "CC:49:99:EE:1A:F4"


# Read from the anchor configuration file
def read_anchor_config():
    with open("assets/anchor_config.json", "r") as f:
        config = json.load(f)

    return config


# Write the anchor positions to the anchors
async def write_anchor_positions(anchor):
    async with BleakClient(anchor["ble_address"]) as client:
        print(f"Writing position to anchor {anchor['id']} at {anchor['ble_address']}")
        await client.write_gatt_char(
            ANCHOR_POSITION_CHARACTERISTIC_UUID,
            struct.pack("<iiiB", anchor["x"], anchor["y"], anchor["z"], 100),
            response=True,
        )


async def write_tag_location_mode():
    async with BleakClient(TAG_BLE_ADDRESS) as client:
        print("Setting tag location mode to position + distances")
        await client.write_gatt_char(
            TAG_LOCATION_MODE_CHARACTERISTIC_UUID, struct.pack("<B", 2)
        )  # 2 for position + distances


# Wrapper function to write positions to all anchors
def setup_dwm():
    anchors = read_anchor_config()

    async def run_all():
        for anchor in anchors:
            await write_anchor_positions(anchor)

        await write_tag_location_mode()

    asyncio.run(run_all())
