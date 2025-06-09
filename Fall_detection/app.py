import streamlit as st
import cv2
import math
import numpy as np
import tempfile
from ultralytics import YOLO
import mediapipe as mp
from fall_detection import process_and_display
from notify import send_notification_to_guardian

# --- Status badge ---
def status_badge(status):
    color_map = {
        "Falling": ("#d7263d", "🚨 Fall Detected"),
        "Moving": ("#1b998b", "🟢 Moving"),
        "Standing": ("#ffe066", "🟡 Standing"),
        "Waiting...": ("#888", "⏳ Waiting...")
    }
    color, text = color_map.get(status, ("#888", "⏳ Waiting..."))
    st.markdown(f'<div style="color:white;background-color:{color};padding:8px 20px;border-radius:10px;font-weight:bold;text-align:left;font-size:16px;max-width:300px;margin:10px 0;">{text}</div>', unsafe_allow_html=True)


# --- Main App ---
def main():
    st.set_page_config(page_title="SafeGuardian - Elderly Fall Detection", layout="centered")

    with st.sidebar:
        st.markdown("""
            <div style='text-align: center;'>
                <img src="https://cdn-icons-png.flaticon.com/512/1048/1048949.png" width="90">
                <h1 style='color: #1b998b; margin: 10px 0;'>🛡️ SafeGuardian</h1>
                <p style='font-size: 15px; font-weight: bold; color: #222;'>Real-time Fall Detection<br>for Elderly Safety</p>
            </div>
            <div style='background-color: #f5f5f5; border-left: 4px solid #1b998b; padding: 15px; border-radius: 10px; margin-top: 15px;'>
                <ul style='padding-left: 20px; color: #1b998b; font-size: 15px; font-weight: bold; line-height: 1.8;'>
                    <li><b>YOLOv8</b> + <b>MediaPipe</b> Precision</li>
                    <li><b>Live Monitoring</b> Interface</li>
                    <li><b>Push Notifications</b> Guardian Alerts</li>
                </ul>
            </div>
            <hr style='border:0.5px solid #ccc; margin-top: 20px;'>
            <p style='font-size: 12px; color: #888; text-align:center;'>© 2025 <strong>Shamli Ingole</strong></p>
        """, unsafe_allow_html=True)

    @st.cache_resource
    def load_models():
        return YOLO('yolov8n.pt'), mp.solutions.pose

    yolo_model, mp_pose = load_models()
    status_placeholder = st.empty()
    frame_placeholder = st.empty()

    if "last_status" not in st.session_state:
        st.session_state.last_status = "Waiting..."
        st.session_state.sms_sent = False
        st.session_state.input_mode = "Upload"

    # --- Modern tab-style layout ---
    st.markdown(f"""
        <div style='max-width: 700px; margin: auto; background-color: #f1f8f6; padding: 15px; border-radius: 10px; border-left: 5px solid #1b998b;'>
            <h3 style='color:#1b998b; margin-bottom: 10px;'>🎥 Choose your Input Source</h3>
        
        </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📷 Webcam Mode", "📁 Upload Video"])

    def handle_status(status):
        st.session_state.last_status = status

        if status == "Falling" and not st.session_state.sms_sent:
            send_notification_to_guardian("A fall has been detected by 🛡️ SafeGuardian!")
            st.toast("Push notification sent to the guardian ✅")
            st.session_state.sms_sent = True
        elif status != "Falling":
            st.session_state.sms_sent = False

        status_placeholder.markdown("")
        with status_placeholder.container():
            status_badge(status)

    def run_detection(cap, skip=3):
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame_idx % skip != 0:
                continue
            frame = cv2.resize(frame, (320, 240))
            processed_frame, status = process_and_display(frame, yolo_model, mp_pose)
            handle_status(status)
            frame_placeholder.image(processed_frame, channels="BGR", use_container_width=True, caption="Live Feed" if st.session_state.input_mode == "Webcam" else "Video Frame")
        cap.release()

    with tab1:
        st.session_state.input_mode = "Webcam"
        if st.button("Start Webcam", use_container_width=True):
            run_detection(cv2.VideoCapture(0), skip=3)

    with tab2:
        st.session_state.input_mode = "Upload"
        with st.container():
            uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
            st.write("")
        if uploaded_file:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            run_detection(cv2.VideoCapture(tfile.name), skip=5)

    st.markdown("""
        <hr>
        <div style='text-align: center; color: #aaa;'>© 2025 SafeGuardian. All rights reserved.</div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
