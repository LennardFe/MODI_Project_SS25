#from target_selection.gesture_recognition import monitor_gesture
from data_handler.data_handler_imu import handle_imu_data
from data_handler.data_handler_dwm import handle_uwb_data
from setup_files.setup_dwm import setup_dwm
from setup_files.setup_db import setup_db
from threading import Thread

# Drop and recreate the SQLite tables
setup_db()

# Push location to anchors and location mode for tag
setup_dwm()

#Thread(target=handle_imu_data).start()
Thread(target=handle_uwb_data).start()