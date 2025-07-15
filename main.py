# from visualization.lamp_visualization import LampVisualization
from target_selection.gesture_recognition import monitor_gesture
#from data_handler.data_handler_imu_gyro import handle_gyro
#from data_handler.data_handler_imu_accel import handle_accel
from data_handler.data_handler_dwm import handle_uwb_data
#from data_handler.data_handler_imu_new import handle_imu_data
from data_handler.data_handler_imu import handle_imu_data

from setup_files.setup_dwm import setup_dwm
from setup_files.setup_db import setup_db
from threading import Thread

from visualization.lamp_visualization import LampVisualization
from visualization.live_animation import LiveThetaAnimation
import matplotlib
from experiments.simulate_run import RealTimeSimulator
# For macOS users, uncomment the following line to use the macOS backend
#matplotlib.use('macosx')

# For Windows / Linux users, uncomment the following line to use the TkAgg backend
# matplotlib.use('TkAgg')

# Global variables
CALIBRATION_ANCHOR = "5C19"


def main():
    # Drop and recreate the SQLite tables
    setup_db()

    # Push location to anchors and location mode for tag
    setup_dwm_thread = Thread(target=setup_dwm)
    setup_dwm_thread.start()
    setup_dwm_thread.join()

    # Start the threads for the handlers and gesture monitoring
    #Thread(target=handle_accel).start()
    #Thread(target=handle_gyro).start()
    Thread(target=handle_imu_data).start()
    Thread(target=handle_uwb_data).start()

    kivy_instance = LampVisualization()
    Thread(target=monitor_gesture, args=(CALIBRATION_ANCHOR, kivy_instance)).start()
    kivy_instance.run()

    # live_theta_anim = LiveThetaAnimation()
    # live_theta_anim.start()
    # Start simulation
    #simulator = RealTimeSimulator()
    #simulator.run_simulation()

    # Start the lamp visualization
    # LampVisualization().run()


if __name__ == "__main__":
    main()
