from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.app import App
import json

class LampVisualization(App):
    # Is called when app is started
    def build(self):
        
        self.title = "Lamp Visualization"

        # Paths to images
        self.img_sources = {
            "on": "assets/images/bulb_on.jpg",
            "off": "assets/images/bulb_off.jpg",
        }

        # Save labels from anchor config
        with open("assets/anchor_config.json") as anchor_config:
            anchors = json.load(anchor_config)
        self.labels = [anchor["id"] for anchor in anchors]

        # Grid for labels aka anchor ids
        label_grid = GridLayout(rows=1, size_hint_y=None, height=40)
        for anchor_id in self.labels:
            label_grid.add_widget(Label(text=anchor_id, font_size="40sp", bold=True))

        # Grid for images
        self.image_grid = GridLayout(rows=1)
        self.images = []
        for _ in self.labels:
            img = Image(source=self.img_sources["off"])
            self.images.append(img)
            self.image_grid.add_widget(img)

        # Put grids together in a layout
        layout = BoxLayout(orientation="vertical")
        layout.add_widget(label_grid)
        layout.add_widget(self.image_grid)

        # Add alert label
        self.alert_label = Label(
            text="", font_size="30sp", bold=True, color=(0.2, 0.6, 1, 1), size_hint_y=None, height=40
        )
        layout.add_widget(self.alert_label)

        # Add IMU found label
        self.imu_label = Label(
            text="IMU NOT FOUND", font_size="20sp", bold=False, color=(1, 0, 0, 1), size_hint_y=None, height=25
        )
        layout.add_widget(self.imu_label)

        # Add DWM found label
        self.dwm_label = Label(
            text="DWM NOT FOUND", font_size="20sp", bold=False, color=(1, 0, 0, 1), size_hint_y=None, height=25
        )
        layout.add_widget(self.dwm_label)

        # Add potential error label
        self.error_label = Label(
            text="", font_size="20sp", bold=False, color=(1, 0, 0, 1), size_hint_y=None, height=25
        )
        layout.add_widget(self.error_label)

        return layout

    def set_on(self, anchor_id):
        if anchor_id in self.labels:
            self.alert_label.text = "SELECTION EVENT"
            index = self.labels.index(anchor_id)
            self.images[index].source = self.img_sources["on"]
            self.images[index].reload()

    def set_all_off(self):
        self.alert_label.text = ""
        for img in self.images:
            img.source = self.img_sources["off"]
            img.reload()

    def set_imu_found(self):
        self.imu_label.text = "IMU FOUND"
        self.imu_label.color = (0, 1, 0, 1)

    def set_dwm_found(self):
        self.dwm_label.text = "DWM FOUND"
        self.dwm_label.color = (0, 1, 0, 1)

    def set_error(self, error_message):
        self.error_label.text = f"ERROR: {error_message}"
