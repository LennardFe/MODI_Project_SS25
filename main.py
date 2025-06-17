import time

import pandas as pd
import serial
import re

# Konfiguration
port = "COM3"  # Passe an deinen tatsächlichen COM-Port an
baudrate = 115200  # DWM1001-Standard
timeout = 1  # Sekunde

try:
    with serial.Serial(port, baudrate, timeout=timeout) as ser:
        print(f"✅ Verbindung zu {port} erfolgreich hergestellt.")
        start_time = time.perf_counter_ns()
        # Kleines Delay, damit DWM1001 bereit ist
        time.sleep(1)

        # Sende einmal Enter (wie bei PuTTY)
        ser.write(b'\r')
        time.sleep(1)
        ser.write(b'\r\r')
        time.sleep(3)
        print(ser.read_all().decode())
        ser.write(b'les\r')
        time.sleep(1)
        df = pd.DataFrame(columns=['timestamp', 'id', 'position', 'distance', 'time_taken', 'est_pos'])
        while True:

            response = ser.readline().decode('utf-8')
            if response != '' and '[' in response:
                timestamp = time.perf_counter_ns()
                strings = response.split(' ')
                data = []
                time_taken = 0
                est_position = 0
                strings.reverse()
                for string in strings:

                    time_taken_re = re.search(r'le_us=(?P<time>\d+)', string)
                    if time_taken_re is not None:
                        time_taken_re = time_taken_re.groupdict()
                        time_taken = time_taken_re['time']

                    est_position_re = re.search(r'est(?P<position>\[[\-0-9.,]+])', string)
                    if est_position_re is not None:
                        est_position_re = est_position_re.groupdict()
                        est_position = est_position_re['position']

                    anchor_data = re.search(r'(?P<id>[0-9A-F]{4})(?P<position>\[[\-0-9.,]+])=(?P<distance>\d+\.\d+)', string)
                    if anchor_data is not None:
                        anchor_data = anchor_data.groupdict()
                        id = anchor_data['id']
                        position = anchor_data['position']
                        distance = anchor_data['distance']
                        data.append([round((timestamp-start_time)/1e6), id, position, distance, time_taken, est_position])

                data_df = pd.DataFrame(data, columns=['timestamp', 'id', 'position', 'distance', 'time_taken', 'est_pos'])
                df = pd.concat([df, data_df], ignore_index=True)
                print(df)
                print('----------------------')




except serial.SerialException as e:
    print(f"Fehler beim Öffnen von {port}: {e}")