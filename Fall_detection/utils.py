import cv2
import math
import numpy as np

def draw_corner_border(img, pt1, pt2, color, thickness=2, corner_length=20):
    x1, y1 = pt1
    x2, y2 = pt2
    # Top left
    cv2.line(img, (x1, y1), (x1 + corner_length, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + corner_length), color, thickness)
    # Top right
    cv2.line(img, (x2, y1), (x2 - corner_length, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + corner_length), color, thickness)
    # Bottom left
    cv2.line(img, (x1, y2), (x1 + corner_length, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - corner_length), color, thickness)
    # Bottom right
    cv2.line(img, (x2, y2), (x2 - corner_length, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - corner_length), color, thickness)

def calculate_angle(shoulder_center, hip_center):
    dy = shoulder_center[1] - hip_center[1]
    dx = shoulder_center[0] - hip_center[0]
    angle = math.atan2(dy, dx)
    return abs(90 - np.degrees(angle))

def classify_posture(torso_angle, falling_threshold=30):
    if torso_angle < falling_threshold:
        return "Standing"
    else:
        return "Falling"