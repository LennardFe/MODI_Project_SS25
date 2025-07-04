import asyncio
from bleak import BleakClient
import struct

# 🔧 Hier deine bekannte BLE-MAC-Adresse eintragen:
ADDRESS = "CC:49:99:EE:1A:F4"  # Beispiel → MAC deines DWM1001

UART_SERVICE_UUID = "680c21d9-c946-4c1f-9c11-baa1c21329e7"
LOCATION_UUID = "003bbdf2-c634-4b3d-ab56-7ec889b89a37"


def parse_location_data(data: bytearray):
    data_type = data[0]
    print(f"📄 Datentyp: {data_type}")

    if data_type == 0:  # Position only
        x, y, z, qf = struct.unpack_from("<iiiB", data, offset=1)
        print(f"📍 Position: x={x}, y={y}, z={z}, Qualität={qf}")

    elif data_type == 1:  # Distances only
        count = data[1]
        print(f"📡 Anzahl Distanzen: {count}")
        offset = 2
        for i in range(count):
            node_id, dist_mm, qf = struct.unpack_from("<HIB", data, offset)
            print(f"  🔹 Anchor ID: {node_id}, Distanz: {dist_mm} mm, Qualität: {qf}")
            offset += 7

    elif data_type == 2:  # Position + Distances
        x, y, z, qf = struct.unpack_from("<iiiB", data, offset=1)
        print(f"📍 Position: x={x}, y={y}, z={z}, Qualität={qf}")
        count = data[14]
        print(f"📡 Anzahl Distanzen: {count}")
        offset = 15
        for i in range(count):
            node_id, dist_mm, qf = struct.unpack_from("<HIB", data, offset)
            print(f"  🔹 Anchor ID: {node_id}, Distanz: {dist_mm} mm, Qualität: {qf}")
            offset += 7

    else:
        print("❌ Unbekannter Datentyp!")


async def read_data():
    async with BleakClient(ADDRESS) as client:
        print("🔗 Verbunden. Lese Location-Daten...")
        data = await client.read_gatt_char(LOCATION_UUID)
        parse_location_data(data)

def handle_uwb_data():
    asyncio.run(read_data())
