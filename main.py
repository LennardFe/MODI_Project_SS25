#from target_selection.gesture_recognition import monitor_gesture
from sensor_data_handler.setup.setup_db import setup_db
from sensor_data_handler.setup.setup_dwm import setup_dwm
from sensor_data_handler.data_handler_imu import handle_imu_data
from sensor_data_handler.data_handler_dwm import handle_uwb_data
from threading import Thread

# Drop and recreate the tables
setup_db()

# Push location of the anchors to them (from anchor_config.json)
setup_dwm()

#Thread(target=handle_imu_data).start()
Thread(target=handle_uwb_data).start()