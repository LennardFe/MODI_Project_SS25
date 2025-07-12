"""
Live Theta Animation

Simple real-time animation showing:
- Current position
- Theta direction (where you're looking)
- Anchors with last selected target highlighted
- Bearings to all anchors (calculated using existing functions)
- No threading issues (runs in main thread)

INTEGRATION:
- Uses existing theta_calc.py for theta calculation
- Uses existing selection_manager.py for position functions
- Uses existing bearing_calc.py for bearing calculations
- Eliminates code duplication and ensures consistency
"""

import matplotlib
# Try to set an interactive backend
try:
    matplotlib.use('macosx')  # Native macOS backend
except ImportError:
    try:
        matplotlib.use('TkAgg')  # Second choice
    except ImportError:
        try:
            matplotlib.use('Qt5Agg')  # Third choice
        except ImportError:
            matplotlib.use('Agg')  # Final fallback (non-interactive)
            print("Warning: Using non-interactive backend. Animation may not display properly.")

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sqlite3
import numpy as np
import json
import time
import threading
import math

# Import existing calculation functions to avoid code duplication
from target_selection.calculations.theta_calc import get_theta
from target_selection.selection_manager import get_initial_position, read_anchor_config
from target_selection.calculations.bearing_calc import get_bearings

class LiveThetaAnimation:
    def __init__(self, db_path="assets/MODI.db"):
        self.db_path = db_path
        self.database_name = db_path.split('/')[-1].replace('.db', '')  # Extract database name from path
        self.anchor_positions = self.read_anchor_config()
        self.last_timestamp = 0
        self.last_target_timestamp = 0
        self.current_position = None
        self.current_theta = 0.0
        self.last_selected_target = None
        self.running = False
        
        print(f"🔧 Live animation using existing calculation functions")
        print(f"📊 Database: {self.database_name}")
        print(f"🎯 Reusing theta_calc.py, bearing_calc.py, selection_manager.py functions")
        
        # Initialize plot
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.setup_plot()
        
    def read_anchor_config(self):
        """Load anchor configuration using existing function"""
        try:
            # Use existing function but adapt to include z-coordinate
            anchors_2d = read_anchor_config()
            
            # Load full config to get z-coordinates
            with open("assets/anchor_config.json", "r") as f:
                config = json.load(f)
            
            anchors = {}
            for anchor in config:
                anchor_id = anchor["id"]
                # Convert from mm to meters
                x_m = anchor["x"] / 1000.0
                y_m = anchor["y"] / 1000.0
                z_m = anchor["z"] / 1000.0
                anchors[anchor_id] = np.array([x_m, y_m, z_m])
            
            return anchors
        except Exception as e:
            print(f"Error loading anchor config: {e}")
            return {}
    
    def setup_plot(self):
        """Initialize the plot"""
        self.ax.clear()
        self.ax.set_xlim(-3, 5)
        self.ax.set_ylim(-4, 3)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title('Live UWB Tracking - Theta Direction & Target Selection', 
                         fontsize=14, fontweight='bold')
        self.ax.set_xlabel('X (meters)', fontsize=12)
        self.ax.set_ylabel('Y (meters)', fontsize=12)
        
        # Draw anchors
        selected_anchor_plotted = False
        normal_anchor_plotted = False
        
        for anchor_id, pos in self.anchor_positions.items():
            x, y = pos[0], pos[1]
            
            # Check if this is the last selected target
            if anchor_id == self.last_selected_target:
                # Highlight selected target
                label = 'Last Selected Target' if not selected_anchor_plotted else None
                self.ax.scatter(x, y, color='red', marker='^', s=250, 
                              edgecolor='black', linewidth=3, zorder=10,
                              label=label)
                selected_anchor_plotted = True
                # Add special label
                self.ax.text(x + 0.15, y + 0.15, f'{anchor_id}\nLAST SELECTED', 
                           fontsize=10, fontweight='bold', ha='center',
                           bbox=dict(boxstyle="round,pad=0.4", facecolor="red", 
                                   alpha=0.8, edgecolor='black'))
            else:
                # Normal anchor
                label = 'Anchors' if not normal_anchor_plotted else None
                self.ax.scatter(x, y, color='orange', marker='^', s=150, 
                              edgecolor='black', linewidth=1, zorder=8,
                              label=label)
                normal_anchor_plotted = True
                # Normal label
                self.ax.text(x + 0.1, y + 0.1, anchor_id, fontsize=10, fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # Draw current position if available
        if self.current_position is not None:
            pos_m = self.current_position / 1000.0  # Convert mm to meters
            
            # Current position
            self.ax.scatter(pos_m[0], -pos_m[1], color='blue', marker='o', s=200,
                          edgecolor='white', linewidth=3, zorder=15,
                          label='Current Position')
            
            # Theta direction vector
            if self.current_theta != 0:
                theta_rad = np.radians(self.current_theta)
                arrow_length = 1.0  # 1 meter
                
                # Calculate arrow end point
                arrow_end_x = pos_m[0] + arrow_length * np.cos(theta_rad)
                arrow_end_y = -pos_m[1] + arrow_length * np.sin(theta_rad)
                
                # Draw theta direction arrow using annotate (better for legends)
                self.ax.annotate('', xy=(arrow_end_x, arrow_end_y), 
                               xytext=(pos_m[0], -pos_m[1]),
                               arrowprops=dict(arrowstyle='->', lw=4, color='green'))
                
                # Add invisible point for legend
                self.ax.plot([], [], color='green', linewidth=4, 
                           label=f'Looking Direction (θ={self.current_theta:.1f}°)')
        else:
            # Add placeholder elements for consistent legend
            self.ax.plot([], [], color='blue', marker='o', markersize=10, 
                       label='Current Position', linestyle='None')
            self.ax.plot([], [], color='green', linewidth=4, 
                       label='Looking Direction (θ=0.0°)')
        
        # Add legend only if we have labeled elements
        handles, labels = self.ax.get_legend_handles_labels()
        if handles:
            self.ax.legend(loc='upper right', fontsize=10)
        
        # Add info text
        info_text = []
        if self.current_position is not None:
            pos_m = self.current_position / 1000.0
            info_text.extend([
                f"Position: ({pos_m[0]:.2f}, {pos_m[1]:.2f}) m",
                f"Theta: {self.current_theta:.1f}°"
            ])
            
            # Show bearings to anchors using existing calculation functions
            bearings = self.calculate_bearings()
            if bearings:
                info_text.append("Bearings to anchors:")
                for anchor_id, bearing in bearings.items():
                    info_text.append(f"  {anchor_id}: {bearing:.1f}°")
        else:
            info_text.append("Waiting for position data...")
            
        if self.last_selected_target:
            info_text.append(f"Last Selected: {self.last_selected_target}")
        else:
            info_text.append("No target selected yet")
            
        info_str = "\n".join(info_text)
        self.ax.text(0.02, 0.98, info_str, transform=self.ax.transAxes,
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    def get_latest_position(self):
        """Get latest position from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            
            # Get latest position
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
                self.last_timestamp = timestamp
                self.current_position = np.array([x, y, z])
                return True
            
            return False
            
        except Exception as e:
            print(f"Database error: {e}")
            return False
    
    def get_latest_theta(self):
        """Calculate latest theta using existing function"""
        try:
            # Use existing theta calculation function
            theta_rad = get_theta(self.database_name)
            self.current_theta = np.degrees(theta_rad)
            return True
            
        except Exception as e:
            print(f"Theta calculation error: {e}")
            return False
    
    def get_initial_position_using_existing(self):
        """Get initial position using existing function"""
        try:
            # Use existing function from selection_manager
            initial_pos_2d = get_initial_position(self.database_name)
            # Convert to 3D with z=0 for consistency
            return np.array([initial_pos_2d[0], initial_pos_2d[1], 0])
        except Exception as e:
            print(f"Initial position error: {e}")
            return None
    
    def calculate_bearings(self, calibration_anchor="5C19"):
        """Calculate bearings to all anchors using existing function"""
        try:
            if self.current_position is None or self.current_theta == 0:
                return {}
            
            # Get anchors in 2D format (mm) for bearing calculation
            anchors_2d = read_anchor_config()
            
            # Get initial position in 2D format (mm)
            initial_pos_2d = get_initial_position(self.database_name)
            
            # Current position in 2D format (mm)
            current_pos_2d = np.array([self.current_position[0], self.current_position[1]])
            
            # Calculate bearings using existing function
            bearings = get_bearings(
                anchors_2d,
                calibration_anchor,
                initial_pos_2d,
                math.radians(self.current_theta),
                current_pos_2d
            )
            
            return bearings
            
        except Exception as e:
            print(f"Bearing calculation error: {e}")
            return {}
    
    def check_for_new_target_selection(self):
        """Check for new target selections in database"""
        try:
            # This is a simple approach - in practice you might want to store 
            # target selections in a separate table or file
            # For now, we'll check if there's recent gesture activity
            
            # You could implement a simple target selection storage mechanism here
            # For demonstration, we'll just simulate checking
            pass
            
        except Exception as e:
            print(f"Target check error: {e}")
    
    def update_from_file(self):
        """Check for target selection updates from file"""
        try:
            # Simple file-based communication for target selection
            target_file = "plots/last_selected_target.txt"
            try:
                with open(target_file, 'r') as f:
                    line = f.read().strip()
                    if line:
                        parts = line.split(',')
                        if len(parts) >= 2:
                            timestamp = float(parts[0])
                            target = parts[1]
                            
                            # Only update if this is newer than what we have
                            if timestamp > self.last_target_timestamp:
                                self.last_selected_target = target
                                self.last_target_timestamp = timestamp
                                print(f"📍 Target updated: {target}")
                                return True
            except FileNotFoundError:
                pass  # File doesn't exist yet
                
        except Exception as e:
            print(f"File check error: {e}")
            
        return False
    
    def update_animation(self, frame):
        """Update animation frame"""
        if not self.running:
            return []
        
        try:
            # Get latest data
            position_updated = self.get_latest_position()
            theta_updated = self.get_latest_theta()
            target_updated = self.update_from_file()
            
            # Redraw if anything updated
            if position_updated or theta_updated or target_updated:
                self.setup_plot()
                
        except Exception as e:
            print(f"Animation update error: {e}")
        
        return []
    
    def start(self):
        """Start the animation"""
        print("Starting live theta animation...")
        print("📍 Target selections will be highlighted automatically")
        print("🔄 Animation updates in real-time")
        
        self.running = True
        
        # Create animation and store it as instance variable
        self.ani = animation.FuncAnimation(
            self.fig, self.update_animation, 
            interval=200,  # Update every 200ms
            blit=False,
            cache_frame_data=False
        )
        
        plt.tight_layout()
        
        # Show the plot - this will block until window is closed
        try:
            plt.show(block=True)
        except KeyboardInterrupt:
            print("\nAnimation interrupted by user")
        except Exception as e:
            print(f"Animation display error: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the animation"""
        self.running = False
        if hasattr(self, 'ani'):
            self.ani.event_source.stop()
        plt.close(self.fig)
        print("Animation stopped") 