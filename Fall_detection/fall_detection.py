import cv2
import math
import numpy as np
from utils import draw_corner_border, calculate_angle
from ultralytics import YOLO
import mediapipe as mp

prev_keypoints = None

icon = cv2.imread("icon3.jpg")
if icon is not None:
    icon = cv2.resize(icon, (20, 20))
else:
    icon = np.zeros((20, 20, 3), dtype=np.uint8)
def process_and_display(frame, yolo_model, mp_pose, movement_threshold=15):
    global prev_keypoints
    status = "Waiting..."
    display_frame = frame.copy()
    results = yolo_model(frame)
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        for result in results:
            for bbox, cls in zip(result.boxes.xyxy, result.boxes.cls):
                if int(cls) == 0:
                    x1, y1, x2, y2 = map(int, bbox)
                    person_bbox = frame[y1:y2, x1:x2]
                    if person_bbox.size == 0:
                        continue
                    person_bbox_rgb = cv2.cvtColor(person_bbox, cv2.COLOR_BGR2RGB)
                    person_results = pose.process(person_bbox_rgb)
                    if person_results.pose_landmarks:
                        landmarks = person_results.pose_landmarks.landmark
                        keypoints = np.array([[lm.x * person_bbox.shape[1], lm.y * person_bbox.shape[0]] for lm in landmarks])
                        shoulders = [
                            (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * person_bbox.shape[1], landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * person_bbox.shape[0]),
                            (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * person_bbox.shape[1], landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * person_bbox.shape[0])
                        ]
                        hips = [
                            (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * person_bbox.shape[1], landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * person_bbox.shape[0]),
                            (landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * person_bbox.shape[1], landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * person_bbox.shape[0])
                        ]
                        shoulder_center = ((shoulders[0][0] + shoulders[1][0]) / 2, (shoulders[0][1] + shoulders[1][1]) / 2)
                        hip_center = ((hips[0][0] + hips[1][0]) / 2, (hips[0][1] + hips[1][1]) / 2)
                        torso_angle = calculate_angle(hip_center, shoulder_center)
                        posture = "Standing" if torso_angle < 30 else "Falling"
                        movement = 0
                        if prev_keypoints is not None and keypoints.shape == prev_keypoints.shape:
                            movement = np.linalg.norm(keypoints - prev_keypoints)
                        prev_keypoints = keypoints
                        if posture == "Standing" and movement > movement_threshold:
                            status = "Moving"
                        else:
                            status = posture
                        if status == "Falling":
                            draw_corner_border(display_frame, (x1, y1), (x2, y2), (0, 0, 255), thickness=3, corner_length=20)
                            # Overlay icon at top-left
                            try:
                                icon_h, icon_w = icon.shape[:2]
                                x1_icon, y1_icon = x1, y1
                                x2_icon, y2_icon = x1_icon + icon_w, y1_icon + icon_h

                                if x2_icon < display_frame.shape[1] and y2_icon < display_frame.shape[0]:
                        
                                    display_frame[y1_icon:y2_icon, x1_icon:x2_icon] = icon
                            except Exception as e:
                                print(f"Error overlaying icon: {e}")

    return display_frame, status