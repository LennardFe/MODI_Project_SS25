import math
import numpy as np
from playsound3 import playsound

A1 = (0, 2)
A2 = (1, 0)
A3 = (2, 1)

theta = math.radians(30)
anchors = {
    "A1": np.array(A1),
    "A2": np.array(A2),
    "A3": np.array(A3),
}

T = np.array((1, 1))


def get_rotation_matrix(alpha):
    alpha = -alpha
    cos_alpha = math.cos(alpha)
    sin_alpha = math.sin(alpha)
    return np.array([[cos_alpha, -sin_alpha], [sin_alpha, cos_alpha]])


def calc_angle(x, y):
    dot_product = np.dot(x, y)
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)
    return math.acos(dot_product / (norm_x * norm_y))


def calc_bearing(current_heading_v, a, t):
    return math.degrees(calc_angle(current_heading_v, a - t))


def play_sounds(bearings):
    min_bearing = round(min(bearings.values()))
    min_bearing_string = f"{min_bearing:03}"
    playsound("assets/sound_files/ESMContactBearing.wav")
    for char in min_bearing_string:
        playsound(f"assets/sound_files/numbers/{char}.wav")


def get_bearings(anchors, calibration_anchor, initial_position, theta, t):
    ini_vec = anchors[calibration_anchor] - initial_position
    rotation_matrix = get_rotation_matrix(theta)
    current_heading_v = np.dot(rotation_matrix, ini_vec)
    bearings = {}
    for anchor_name, anchor in anchors.items():
        bearings[anchor_name] = calc_bearing(current_heading_v, anchor, t)
    return bearings


if __name__ == "__main__":
    INITIAL_POSITION = np.array([1, 1])
    bearings = get_bearings(anchors, "A1", INITIAL_POSITION, theta, T)
