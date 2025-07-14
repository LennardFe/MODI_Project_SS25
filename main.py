from experiments.simulate_run import initialize_and_run_simulation
from visualization.lamp_visualization import LampVisualization
from target_selection.gesture_recognition import monitor_gesture
from data_handler.data_handler_imu import handle_imu_data
from data_handler.data_handler_dwm import handle_uwb_data
from setup_files.setup_dwm import setup_dwm
from setup_files.setup_db import setup_db
from threading import Thread
import matplotlib

# For macOS users, uncomment the following line to use the macOS backend
#matplotlib.use('macosx')

# For Windows / Linux users, uncomment the following line to use the TkAgg backend
matplotlib.use('TkAgg')

# Global variables
CALIBRATION_ANCHOR = "5C19"

def main():
    # Drop and recreate the SQLite tables
    # setup_db()

    # Push location to anchors and location mode for tag
    # setup_dwm()

    # Start the threads for the handlers and gesture monitoring
    # Thread(target=handle_imu_data).start()
    # Thread(target=handle_uwb_data).start()
    # Thread(target=monitor_gesture, args=(CALIBRATION_ANCHOR,)).start()

    # Start simulation
    # initialize_and_run_simulation()

    # Start the lamp visualization
    LampVisualization().run()


if __name__ == "__main__":
    main()