from target_selection.gesture_recognition import monitor_gesture
#from data_handler.data_handler_imu_new import handle_imu_data
from visualization.lamp_visualization import LampVisualization
from visualization.live_animation import LiveThetaAnimation
from data_handler.data_handler_dwm import handle_uwb_data
from data_handler.data_handler_imu import handle_imu_data
from experiments.simulate_run import RealTimeSimulator
from setup_files.setup_dwm import setup_dwm
from setup_files.setup_db import setup_db
from threading import Thread
import matplotlib

# For macOS users, uncomment the following line to use the macOS backend
#matplotlib.use('macosx')

# For Windows / Linux users, uncomment the following line to use the TkAgg backend
# matplotlib.use('TkAgg')

# Global variables
CALIBRATION_ANCHOR = "5C19" # Anchor to face in the beginning to calibrate
start_visualization = True # Set to False if you want to skip the visualization
start_setup_dwm = False  # Set to False if you want to skip the DWM setup


def run_simulation(with_animation=True, with_lamp_visualization=False):
    simulator = RealTimeSimulator()
    simulator.run_simulation(with_animation, with_lamp_visualization)

def main(start_setup_dwm):
    # Drop and recreate the SQLite tables
    setup_db()

    # Push new configs to tag and dwms if needed
    if start_setup_dwm:
        setup_dwm_thread = Thread(target=setup_dwm)
        setup_dwm_thread.start()
        setup_dwm_thread.join()

    kivy_instance = LampVisualization()
    Thread(target=handle_imu_data, args=(kivy_instance,), daemon=True).start()
    Thread(target=handle_uwb_data, args=(kivy_instance,), daemon=True).start()
    Thread(target=monitor_gesture, args=(CALIBRATION_ANCHOR, kivy_instance), daemon=True).start()
    kivy_instance.run()

if __name__ == "__main__":
    main(start_setup_dwm)
    #run_simulation(False,False)