import matplotlib.pyplot as plt
import sqlite3
import numpy as np
from collections import deque
import time
import json

class LiveVisualizer:
    def __init__(self, db_path="assets/MODI.db"):
        self.db_path = db_path
        self.anchor_positions = self.read_anchor_config()
        self.positions = deque(maxlen=100)
        self.last_timestamp = 0
        self.running = False
        
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.ax.set_xlim(-2, 4)
        self.ax.set_ylim(-4, 2)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title('Live UWB Position')
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        
        self.current_pos, = self.ax.plot([], [], 'bo', markersize=15, markeredgecolor='white', markeredgewidth=2)
        self.trail, = self.ax.plot([], [], 'b-', alpha=0.6, linewidth=2)
        
        for anchor_id, pos in self.anchor_positions.items():
            x, y = pos[:2]
            self.ax.scatter(x, y, color='red', marker='^', s=150)
            self.ax.text(x + 0.1, y + 0.1, anchor_id, fontsize=9)
        
        plt.tight_layout()
    
    def read_anchor_config(self):
        with open("assets/anchor_config.json", "r") as f:
            config = json.load(f)
        anchors = {}
        for anchor in config:
            # Convert from mm to meters like other functions do
            x_m = anchor["x"] / 1000.0
            y_m = anchor["y"] / 1000.0
            z_m = anchor["z"] / 1000.0
            anchors[anchor["id"]] = np.array([x_m, y_m, z_m])
        return anchors
    
    def get_latest_position(self):
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            
            if self.last_timestamp == 0:
                print("Getting initial position...")
                cur.execute("""
                    SELECT timestamp, est_position_x, est_position_y, est_position_z
                    FROM location_data 
                    WHERE est_position_x IS NOT NULL 
                    AND est_position_y IS NOT NULL 
                    AND est_position_z IS NOT NULL
                    ORDER BY timestamp DESC LIMIT 1
                """)
            else:
                cur.execute("""
                    SELECT timestamp, est_position_x, est_position_y, est_position_z
                    FROM location_data 
                    WHERE est_position_x IS NOT NULL 
                    AND est_position_y IS NOT NULL 
                    AND est_position_z IS NOT NULL
                    AND timestamp > ?
                    ORDER BY timestamp DESC LIMIT 1
                """, (self.last_timestamp,))
            
            result = cur.fetchone()
            conn.close()
            
            if result:
                timestamp, x, y, z = result
                x_m, y_m, z_m = x / 1000.0, y / 1000.0, z / 1000.0
                position = np.array([x_m, -y_m, z_m])
                self.last_timestamp = timestamp
                return position
            else:
                if self.last_timestamp == 0:
                    print("No data found in database!")
                return None
        except Exception as e:
            print(f"Database error: {e}")
        return None
    

    
    def start(self):
        self.running = True
        plt.show(block=False)
        
        # Manual update loop - works in threads
        while self.running:
            try:
                position = self.get_latest_position()
                if position is not None:
                    self.positions.append(position)
                    self.current_pos.set_data([position[0]], [position[1]])
                    
                    if len(self.positions) > 1:
                        pos_array = np.array(self.positions)
                        self.trail.set_data(pos_array[:, 0], pos_array[:, 1])
                    
                    # Force update
                    self.fig.canvas.draw_idle()
                    self.fig.canvas.flush_events()
                
                # Keep matplotlib alive
                plt.pause(0.05)
                
            except Exception as e:
                print(f"Visualization error: {e}")
                break
    
    def stop(self):
        self.running = False
        plt.close(self.fig) 