import math
import numpy as np

A1 = (0, 2)
A2 = (1, 0)
A3 = (2, 1)

theta = math.radians(210)
anchors = {
    "A1": np.array(A1),
    "A2": np.array(A2),
    "A3": np.array(A3),
}

T = np.array((1, 1))
ini_vec = np.array((-1, 1))

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

def get_bearings(anchors, ini_vec, theta):
    initial_heading = -calc_angle(np.array([0, 1]), ini_vec)
    print("Initial heading: {}".format(math.degrees(initial_heading)))

    rotation_matrix = get_rotation_matrix(theta)
    current_heading_v = np.dot(rotation_matrix, ini_vec)
    print("Current heading v: {}".format(current_heading_v))
    bearings = {}
    for anchor_name, anchor in anchors.items():
        bearings[anchor_name] = calc_bearing(current_heading_v, anchor, T)
    print("Bearings: {}".format(bearings))
    return bearings


bearings = get_bearings(anchors, ini_vec, theta)
