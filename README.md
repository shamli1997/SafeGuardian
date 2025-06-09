# 🛡️ SafeGuardian – Real-Time Fall Detection for Elderly Care

**SafeGuardian** is a smart, responsive, and real-time fall detection system built with YOLOv8, MediaPipe, and Streamlit. It is designed to detect falls in elderly individuals and instantly alert guardians via push notifications using free services like ntfy.sh.

---

## 🚀 Features

- 🎯 **Accurate Fall Detection** using YOLOv8 + MediaPipe Pose
- 🖥️ **Supports Webcam & Video Upload**
- 🔔 **Push Notification Alerts** via ntfy.sh (Free)

---

## 📸 Demo Preview

> The app shows live detection with badge updates for:
>
> - 🟢 Moving
> - 🟡 Standing
> - 🚨 Falling

---

## 🧠 How Fall Detection Works

SafeGuardian uses a hybrid computer vision pipeline that combines **YOLOv8** for object detection and **MediaPipe Pose** for body posture analysis.

### ✅ 1. Person Detection – YOLOv8

- YOLOv8 detects people in each frame using bounding boxes.
- Only the `"person"` class is passed to the next stage, filtering out noise from other objects.

### ✅ 2. Pose Estimation – MediaPipe

- For each detected person, MediaPipe Pose extracts 33 key body landmarks (e.g., nose, shoulders, hips, knees).
- These landmarks form a skeleton model representing the person's posture.

### ✅ 3. Posture Analysis Logic

The system analyzes:

- **Body Orientation**

  - Standing → body keypoints form a vertical alignment.
  - Fallen → keypoints shift to a horizontal alignment.

- **Bounding Box Ratio**

  - Tall (height > width) → Standing
  - Wide (width > height) → Falling

- **Head Position**

  - If the head drops below hips/knees → potential fall

- **Sudden Posture Change**
  - Sharp change in orientation between frames can trigger fall detection.

### ✅ 4. Fall Confirmation

- To avoid false positives, the system checks that the “Falling” posture is detected consistently across a few frames.
- Once confirmed:
  - The status badge changes to `🚨 Fall Detected`
  - A push notification is sent to the guardian using [ntfy.sh](https://ntfy.sh)
  - A short toast alert is shown in the app as visual confirmation

---

## 🧰 Installation

```bash
pip install -r requirements.txt
```

Required packages:

- streamlit
- opencv-python
- ultralytics
- mediapipe
- requests

---

## ▶️ Running the App

```bash
python3 -m streamlit run app.py
```

---

## 🔔 Notification Setup

1. Open a browser or the ntfy app
2. Visit or subscribe to: `https://ntfy.sh/safeguardian`
3. When a fall is detected, you’ll receive an instant push notification

---

## 📁 File Structure

```
.
├── app.py        # Main Streamlit app UI
├── fall_detection.py      # Core YOLO + pose logic
├── notify.py           # Push notification logic using ntfy.sh
```

---

## ✨ Customization

- Change topic name in `notify.py` to personalize notification channels

---

## 👩‍⚕️ Use Cases

- Home elderly safety monitoring
- Hospital patient care
- Independent living support

---

## 🙌 Credits

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [MediaPipe](https://google.github.io/mediapipe/)
- [ntfy.sh](https://ntfy.sh)

---
