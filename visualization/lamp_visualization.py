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
            text="", font_size="40sp", bold=True, color=(1, 0, 0, 1)
        )
        layout.add_widget(self.alert_label)

        # Test
        #Clock.schedule_once(lambda dt: self.set_on("DC0F"), 0)  # nach 3 Sekunden
        #Clock.schedule_once(lambda dt: self.set_all_off(), 5)  # nach 5 Sekunden
        #Clock.schedule_once(lambda dt: self.set_on("96BB"), 7)  # nach 3 Sekunden

        return layout

    def set_on(self, anchor_id):
        """Show turned-on light bulb for one anchor_id."""
        if anchor_id in self.labels:
            self.alert_label.text = "SELECTION EVENT"
            index = self.labels.index(anchor_id)
            self.images[index].source = self.img_sources["on"]
            self.images[index].reload()

    def set_all_off(self):
        """Turn all light bulbs off."""
        self.alert_label.text = ""
        for img in self.images:
            img.source = self.img_sources["off"]
            img.reload()
