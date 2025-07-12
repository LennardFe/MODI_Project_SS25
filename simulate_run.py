import sqlite3
import time
import threading
import os
from target_selection.gesture_recognition import  monitor_gesture
from setup_files.setup_db import setup_db
from live_theta_animation import LiveThetaAnimation
from tqdm import tqdm
# Configuration
SOURCE_DB = "assets/MODI.db"  # Database with recorded data
CALIBRATION_ANCHOR = "5C19"

class RealTimeSimulator:
    def __init__(self, source_db):
        self.source_db = source_db
        self.simulation_db = "assets/MODI_simulation.db"  # Temporary simulation database
        self.simulation_running = False
        self.gesture_monitoring = False
        
    def delete_simulation_database(self):
        """Delete simulation database if it exists"""
        if os.path.exists(self.simulation_db):
            try:
                os.remove(self.simulation_db)
                print(f"🗑️  Deleted existing simulation database: {self.simulation_db}")
            except Exception as e:
                print(f"Warning: Could not delete simulation database: {e}")
        
    
    def setup_simulation_database(self):
        """Create a clean simulation database"""
        print("Setting up simulation database...")
        
        # Delete existing simulation database
        self.delete_simulation_database()
        
        # Create new simulation database
        conn = sqlite3.connect(self.simulation_db)
        cur = conn.cursor()
        
        # Create tables
        cur.execute("""
            CREATE TABLE IF NOT EXISTS gyro_data (
                id INTEGER PRIMARY KEY,
                timestamp INTEGER,
                x REAL,
                y REAL,
                z REAL
            )
        """)
        
        cur.execute("""
            CREATE TABLE IF NOT EXISTS accel_data (
                id INTEGER PRIMARY KEY,
                timestamp INTEGER,
                x REAL,
                y REAL,
                z REAL
            )
        """)
        
        cur.execute("""
            CREATE TABLE IF NOT EXISTS location_data (
                id INTEGER PRIMARY KEY,
                timestamp INTEGER,
                anchor_id TEXT,
                distance INTEGER,
                distance_qf INTEGER,
                est_position_x INTEGER,
                est_position_y INTEGER,
                est_position_z INTEGER,
                est_position_qf INTEGER
            )
        """)
        
        conn.commit()
        conn.close()
        print("✅ Simulation database ready")
    
    def load_historical_data(self):
        """Load all historical data sorted by timestamp"""
        print("Loading historical data...")
        conn = sqlite3.connect(self.source_db)
        cur = conn.cursor()
        
        # Load all data with timestamps
        all_data = []
        
        # Gyro data
        gyro_data = cur.execute("SELECT timestamp, x, y, z FROM gyro_data ORDER BY timestamp").fetchall()
        for row in gyro_data:
            all_data.append((row[0], 'gyro', row[1:]))
            
        # Accel data
        accel_data = cur.execute("SELECT timestamp, x, y, z FROM accel_data ORDER BY timestamp").fetchall()
        for row in accel_data:
            all_data.append((row[0], 'accel', row[1:]))
            
        # Location data
        location_data = cur.execute("""SELECT timestamp, anchor_id, distance, distance_qf, 
                                       est_position_x, est_position_y, est_position_z, est_position_qf 
                                       FROM location_data ORDER BY timestamp""").fetchall()
        for row in location_data:
            all_data.append((row[0], 'location', row[1:]))
        
        conn.close()
        
        # Sort by timestamp
        all_data.sort(key=lambda x: x[0])
        
        if all_data:
            duration_ns = all_data[-1][0] - all_data[0][0]
            duration_seconds = duration_ns / 1e9
            print(f"Loaded {len(all_data)} data points")
            print(f"Original recording duration: {duration_seconds:.1f} seconds ({duration_seconds/60:.1f} minutes)")
        
        return all_data
    

    
    def real_time_data_feed(self, all_data):
        """Feed data into simulation database at real-time intervals"""
        if not all_data:
            print("No data to simulate")
            return
        
        conn = sqlite3.connect(self.simulation_db, check_same_thread=False)
        cur = conn.cursor()
        
        # Get timing info
        start_time = time.time()
        original_start_timestamp = all_data[0][0]
        
        for i, (original_timestamp, data_type, data) in tqdm(enumerate(all_data), total=len(all_data), desc="📊 Feeding data"):
            if not self.simulation_running:
                break
            # Calculate when this data point should be fed
            original_offset_ns = original_timestamp - original_start_timestamp
            original_offset_seconds = original_offset_ns / 1e9
            target_time = start_time + original_offset_seconds
            
            # Wait until it's time to feed this data point
            current_time = time.time()
            if current_time < target_time:
                sleep_duration = target_time - current_time
                time.sleep(sleep_duration)
            
            # Insert data with current timestamp (for gesture detection)
            current_timestamp = time.time_ns()
            
            try:
                if data_type == 'gyro':
                    cur.execute(
                        "INSERT INTO gyro_data (timestamp, x, y, z) VALUES (?, ?, ?, ?)",
                        (current_timestamp, data[0], data[1], data[2])
                    )
                elif data_type == 'accel':
                    cur.execute(
                        "INSERT INTO accel_data (timestamp, x, y, z) VALUES (?, ?, ?, ?)",
                        (current_timestamp, data[0], data[1], data[2])
                    )
                elif data_type == 'location':
                    cur.execute(
                        """INSERT INTO location_data (timestamp, anchor_id, distance, distance_qf, 
                           est_position_x, est_position_y, est_position_z, est_position_qf) 
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                        (current_timestamp, data[0], data[1], data[2], data[3], data[4], data[5], data[6])
                    )
                
                conn.commit()
                    
            except sqlite3.Error as e:
                print(f"Database error: {e}")
        
        conn.close()
        print(f"✅ Data feed completed at {time.strftime('%H:%M:%S')}")

    
    def run_simulation(self):
        """Run the real-time simulation"""
        all_data = self.load_historical_data()
        
        if not all_data:
            print("No historical data found!")
            return
        
        try:
            # Setup simulation environment
            self.setup_simulation_database()
            
            self.simulation_running = True
            
            # Start gesture monitoring thread (like main.py)
            gesture_thread = threading.Thread(
                target=monitor_gesture,
                args=(CALIBRATION_ANCHOR, "MODI_simulation"),
                daemon=True
            )
            gesture_thread.start()
            
            # Start real-time data feed thread (like handle_imu_data and handle_uwb_data in main.py)
            data_feed_thread = threading.Thread(
                target=self.real_time_data_feed,
                args=(all_data,),
                daemon=True
            )
            data_feed_thread.start()
            
            print("🚀 Simulation threads started - running concurrently like main.py")
            print("   📊 Data feed thread: feeding sensor data in real-time")
            print("   🎯 Gesture monitoring thread: watching for gestures")
            print("   🖥️  Live animation will show theta direction and target selection")
            print("⏰ This will take the same amount of time as the original recording")
            
            # Give threads time to start
            time.sleep(2)
            
            # Start live animation for simulation
            animation = LiveThetaAnimation(self.simulation_db)
            
            animation.start()  # This blocks until window closed
            
            
        except KeyboardInterrupt:
            print("\n⏹️  Simulation interrupted by user")
            
        except Exception as e:
            print(f"❌ Simulation error: {e}")
            
        finally:
            self.simulation_running = False
            animation.stop()
            self.simulation_running = False
            print("🏁 Simulation finished!")

def main():
    """Main simulation function"""
    print("=== MODI Project Real-Time Simulation ===")
    print("🕐 Replaying recorded data at original timing intervals")
    print(f"📂 Source database: {SOURCE_DB}")
    print(f"🎯 Calibration anchor: {CALIBRATION_ANCHOR}")
    print("-" * 50)
    
    # Check database
    try:
        conn = sqlite3.connect(SOURCE_DB)
        cur = conn.cursor()
        
        tables = ['gyro_data', 'accel_data', 'location_data']
        total_records = 0
        for table in tables:
            try:
                count = cur.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                print(f"📋 {table}: {count} records")
                total_records += count
            except sqlite3.Error:
                print(f"❌ {table}: table not found")
                
        conn.close()
        
        if total_records == 0:
            print("❌ No data found. Please record some data first.")
            return
            
    except sqlite3.Error as e:
        print(f"❌ Database error: {e}")
        return
    

    simulator = RealTimeSimulator(SOURCE_DB)
    simulator.run_simulation()

if __name__ == "__main__":
    main()
