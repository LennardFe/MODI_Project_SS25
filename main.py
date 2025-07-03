from target_selection.gesture_recognition import monitor_gesture
from sensor_data_handler.data_handler_imu import main
from threading import Thread
import time

Thread(target=main).start()