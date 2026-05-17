import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from ultralytics import YOLO

st.set_page_config(
    page_title="Intelligent Traffic Surveillance System",
    layout="wide"
)

st.markdown("""
<style>
.main-title {
    font-size: 52px;
    font-weight: 900;
    color: #38bdf8;
    margin-bottom: 25px;
    line-height: 1.2;
}

.section-title {
    font-size: 24px;
    font-weight: 700;
    color: #e2e8f0;
    margin-top: 20px;
    margin-bottom: 15px;
}

.metric-card {
    background-color: #020617;
    padding: 20px;
    border-radius: 16px;
    text-align: center;
    border: 1px solid #1e40af;
}

.metric-number {
    font-size: 38px;
    font-weight: bold;
    color: #22c55e;
}

.metric-label {
    color: #cbd5e1;
    font-size: 16px;
    margin-top: 5px;
}

.info-box {
    background-color: #111827;
    padding: 18px;
    border-radius: 14px;
    border: 1px solid #334155;
    color: #cbd5e1;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

if "running" not in st.session_state:
    st.session_state.running = False

if "stop" not in st.session_state:
    st.session_state.stop = False


def start_analysis():
    st.session_state.running = True
    st.session_state.stop = False


def stop_analysis():
    st.session_state.stop = True
    st.session_state.running = False


st.markdown(
    """
    <div class="main-title">
        Intelligent Traffic Surveillance & Object Analysis System
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="info-box">
        This system analyzes traffic and surveillance videos in real time by detecting objects,
        analyzing motion, extracting shapes, and displaying live counters for vehicles and people.
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.title("Control Panel")

canny_low = st.sidebar.slider("Canny Low Threshold", 0, 255, 50)
canny_high = st.sidebar.slider("Canny High Threshold", 0, 255, 150)
hough_threshold = st.sidebar.slider("Hough Line Threshold", 20, 200, 80)
motion_threshold = st.sidebar.slider("Motion Threshold", 0, 255, 25)
min_area = st.sidebar.slider("Minimum Motion Area", 100, 10000, 900)

st.sidebar.markdown("---")
st.sidebar.info("Upload a video, then press Start Analysis. Press Stop Analysis to stop processing.")

model = YOLO("yolov8n.pt")

uploaded_video = st.file_uploader(
    "Upload Video",
    type=["mp4", "avi", "mov", "mkv"]
)

allowed_classes = ["car", "bicycle", "person", "bus", "truck", "motorcycle"]


def process_frame(frame, bg_subtractor):
    frame = cv2.resize(frame, (960, 540))
    display = frame.copy()

    car_count = 0
    bicycle_count = 0
    person_count = 0

    fg_mask = bg_subtractor.apply(frame)

    _, thresh = cv2.threshold(
        fg_mask,
        motion_threshold,
        255,
        cv2.THRESH_BINARY
    )

    kernel = np.ones((5, 5), np.uint8)

    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.dilate(thresh, kernel, iterations=2)

    results = model(frame, verbose=False)

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = model.names[cls_id]

            if class_name not in allowed_classes or conf < 0.35:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            roi = frame[y1:y2, x1:x2]

            if roi.size == 0:
                continue

            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blur, canny_low, canny_high)

            lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi / 180,
                threshold=hough_threshold,
                minLineLength=25,
                maxLineGap=10
            )

            lines_count = 0

            if lines is not None:
                lines_count = len(lines)

                for line in lines[:8]:
                    lx1, ly1, lx2, ly2 = line[0]

                    cv2.line(
                        display,
                        (x1 + lx1, y1 + ly1),
                        (x1 + lx2, y1 + ly2),
                        (255, 0, 255),
                        2
                    )

            circles = cv2.HoughCircles(
                blur,
                cv2.HOUGH_GRADIENT,
                dp=1.2,
                minDist=25,
                param1=max(50, canny_high),
                param2=25,
                minRadius=8,
                maxRadius=60
            )

            circles_count = 0

            if circles is not None:
                circles = np.uint16(np.around(circles))
                circles_count = len(circles[0])

                for circle in circles[0, :5]:
                    cx, cy, r = circle

                    cv2.circle(
                        display,
                        (x1 + cx, y1 + cy),
                        r,
                        (0, 255, 255),
                        2
                    )

                    cv2.circle(
                        display,
                        (x1 + cx, y1 + cy),
                        2,
                        (0, 0, 255),
                        3
                    )

            if class_name in ["car", "bus", "truck"]:
                label = "car"
                car_count += 1
                color = (0, 255, 0)

            elif class_name == "bicycle":
                label = "bicycle"
                bicycle_count += 1
                color = (255, 255, 0)

            elif class_name == "person":
                label = "person"
                person_count += 1
                color = (255, 0, 0)

            else:
                label = class_name
                color = (0, 165, 255)

            cv2.rectangle(
                display,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            cv2.putText(
                display,
                f"{label} {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

            cv2.putText(
                display,
                f"Lines:{lines_count} Circles:{circles_count}",
                (x1, y2 + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

    return display, thresh, car_count, bicycle_count, person_count


if uploaded_video is not None:
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_file.write(uploaded_video.read())
    video_path = temp_file.name

    st.success("Video uploaded successfully.")

    col_start, col_stop = st.columns([1, 1])

    with col_start:
        st.button("Start Analysis", on_click=start_analysis)

    with col_stop:
        st.button("Stop Analysis", on_click=stop_analysis)

    metric_col1, metric_col2, metric_col3 = st.columns(3)

    cars_box = metric_col1.empty()
    bikes_box = metric_col2.empty()
    persons_box = metric_col3.empty()

    st.markdown(
        '<div class="section-title">Live Analysis Output</div>',
        unsafe_allow_html=True
    )

    video_col, mask_col = st.columns(2)

    frame_window = video_col.empty()
    mask_window = mask_col.empty()

    status_box = st.empty()

    if st.session_state.running:
        cap = cv2.VideoCapture(video_path)

        bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=50,
            detectShadows=True
        )

        while cap.isOpened():
            if st.session_state.stop:
                status_box.warning("Analysis stopped by user.")
                break

            ret, frame = cap.read()

            if not ret:
                status_box.success("Analysis completed.")
                break

            processed_frame, motion_mask, cars, bicycles, persons = process_frame(
                frame,
                bg_subtractor
            )

            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

            cars_box.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-number">{cars}</div>
                    <div class="metric-label">Cars</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            bikes_box.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-number">{bicycles}</div>
                    <div class="metric-label">Bicycles</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            persons_box.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-number">{persons}</div>
                    <div class="metric-label">Persons</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            frame_window.image(
                processed_frame,
                caption="Processed Video Output",
                channels="RGB",
                use_container_width=True
            )

            mask_window.image(
                motion_mask,
                caption="Motion Detection Mask",
                use_container_width=True
            )

        cap.release()

else:
    st.warning("Upload a video to start analysis.")