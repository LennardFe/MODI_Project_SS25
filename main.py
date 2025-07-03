from threading import Thread

from xiao.read_data import main
from gesture_recognition import monitor_gesture
import time

Thread(target=main).start()


