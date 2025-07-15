import math
import numpy as np
from playsound3 import playsound


def get_rotation_matrix(theta):
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    return np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])


def calc_angle(x, y):
    dot_product = np.dot(x, y)
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)
    return math.acos(dot_product / (norm_x * norm_y))


def calc_bearing(current_heading_v, anchor, current_position):
    return math.degrees(calc_angle(current_heading_v, anchor - current_position))


def play_sounds(bearings):
    min_bearing = round(min(bearings.values()))
    min_bearing_string = f"{min_bearing:03}"
    playsound("assets/sound_files/ESMContactBearing.wav")
    for char in min_bearing_string:
        playsound(f"assets/sound_files/numbers/{char}.wav")


def get_bearings(
    anchors, calibration_anchor, initial_position, theta_rad, current_position
):
    ini_vec = anchors[calibration_anchor] - initial_position
    rotation_matrix = get_rotation_matrix(theta_rad)
    current_heading_v = np.dot(rotation_matrix, ini_vec)
    bearings = {}
    for anchor_name, anchor in anchors.items():
        bearings[anchor_name] = calc_bearing(
            current_heading_v, anchor, current_position
        )
    return bearings
