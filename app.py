import streamlit as st
import cv2
import torch
from ultralytics import YOLO
import os

st.set_page_config(
    page_title="VisionX · Object Detection",
    page_icon="🎯",
    layout="centered",
)

st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap');

/* ── Root variables ── */
:root {
    --bg:        #080c10;
    --surface:   #0d1117;
    --border:    #1a2332;
    --accent:    #00ffa3;
    --accent2:   #00c8ff;
    --danger:    #ff4f6b;
    --text:      #c8d8e8;
    --muted:     #4a6070;
    --font-mono: 'Share Tech Mono', monospace;
    --font-ui:   'Rajdhani', sans-serif;
}

/* ── Global reset ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--font-ui) !important;
}

[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 80% 50% at 50% -10%, rgba(0,255,163,0.07) 0%, transparent 60%),
        repeating-linear-gradient(0deg, transparent, transparent 39px, rgba(0,255,163,0.02) 40px),
        repeating-linear-gradient(90deg, transparent, transparent 39px, rgba(0,255,163,0.02) 40px);
    pointer-events: none;
    z-index: 0;
}

[data-testid="stMain"] {
    background: transparent !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--accent); border-radius: 2px; }

/* ── Top logo bar ── */
.logo-bar {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 28px 0 8px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 6px;
}
.logo-icon {
    width: 44px;
    height: 44px;
    border: 2px solid var(--accent);
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 22px;
    box-shadow: 0 0 18px rgba(0,255,163,0.25);
}
.logo-title {
    font-family: var(--font-mono);
    font-size: 24px;
    color: var(--accent);
    letter-spacing: 3px;
    text-shadow: 0 0 20px rgba(0,255,163,0.4);
}
.logo-sub {
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--muted);
    letter-spacing: 2px;
    margin-top: 2px;
}

/* ── Status badge ── */
.status-row {
    display: flex;
    align-items: center;
    gap: 8px;
    margin: 16px 0 28px;
    font-family: var(--font-mono);
    font-size: 12px;
    color: var(--muted);
    letter-spacing: 1.5px;
}
.status-dot {
    width: 8px; height: 8px;
    background: var(--accent);
    border-radius: 50%;
    box-shadow: 0 0 8px var(--accent);
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.35; }
}

/* ── Section label ── */
.section-label {
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--accent);
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 10px;
    opacity: 0.8;
}

/* ── Mode selector ── */
[data-testid="stSelectbox"] label {
    font-family: var(--font-mono) !important;
    font-size: 11px !important;
    color: var(--accent) !important;
    letter-spacing: 2.5px !important;
    text-transform: uppercase !important;
}
[data-testid="stSelectbox"] > div > div {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    color: var(--text) !important;
    font-family: var(--font-ui) !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    transition: border-color 0.2s;
}
[data-testid="stSelectbox"] > div > div:hover,
[data-testid="stSelectbox"] > div > div:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(0,255,163,0.12) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    border: 1px dashed var(--border) !important;
    border-radius: 10px !important;
    background: var(--surface) !important;
    padding: 8px !important;
    transition: border-color 0.2s, box-shadow 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent) !important;
    box-shadow: 0 0 20px rgba(0,255,163,0.08) !important;
}
[data-testid="stFileUploader"] label {
    color: var(--text) !important;
    font-family: var(--font-ui) !important;
    font-size: 14px !important;
    font-weight: 600 !important;
}
[data-testid="stFileUploaderDropzone"] {
    background: transparent !important;
    border: none !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] {
    color: var(--muted) !important;
    font-family: var(--font-mono) !important;
    font-size: 12px !important;
}

/* ── Camera input ── */
[data-testid="stCameraInput"] {
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    overflow: hidden !important;
    background: var(--surface) !important;
}
[data-testid="stCameraInput"] label {
    display: none !important;
}

/* ── Primary button ── */
[data-testid="stButton"] > button {
    width: 100%;
    background: transparent !important;
    color: var(--accent) !important;
    border: 1px solid var(--accent) !important;
    border-radius: 6px !important;
    padding: 12px 24px !important;
    font-family: var(--font-mono) !important;
    font-size: 13px !important;
    letter-spacing: 3px !important;
    text-transform: uppercase !important;
    cursor: pointer;
    position: relative;
    overflow: hidden;
    transition: background 0.2s, box-shadow 0.2s, color 0.2s;
    margin-top: 8px !important;
}
[data-testid="stButton"] > button::before {
    content: '';
    position: absolute;
    inset: 0;
    background: var(--accent);
    transform: scaleX(0);
    transform-origin: left;
    transition: transform 0.25s ease;
    z-index: 0;
}
[data-testid="stButton"] > button:hover::before { transform: scaleX(1); }
[data-testid="stButton"] > button:hover {
    color: var(--bg) !important;
    box-shadow: 0 0 24px rgba(0,255,163,0.35) !important;
}
[data-testid="stButton"] > button > * { position: relative; z-index: 1; }

/* ── Info / spinner text ── */
[data-testid="stText"], .stSpinner {
    font-family: var(--font-mono) !important;
    color: var(--muted) !important;
    font-size: 12px !important;
    letter-spacing: 1px;
}

/* ── Result image / video ── */
[data-testid="stImage"] img {
    border-radius: 10px;
    border: 1px solid var(--border);
    box-shadow: 0 0 30px rgba(0,255,163,0.1);
    width: 100%;
    margin-top: 12px;
}
[data-testid="stVideo"] video {
    border-radius: 10px;
    border: 1px solid var(--border);
    width: 100%;
    margin-top: 12px;
}

/* ── Error box ── */
[data-testid="stAlert"] {
    background: rgba(255,79,107,0.08) !important;
    border: 1px solid var(--danger) !important;
    border-radius: 6px !important;
    color: var(--danger) !important;
    font-family: var(--font-mono) !important;
    font-size: 12px !important;
}

/* ── Feedback stars ── */
[data-testid="stFeedback"] {
    margin-top: 20px;
    opacity: 0.6;
}

/* ── Divider ── */
.custom-divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 28px 0 22px;
}

/* ── Result label ── */
.result-label {
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--accent2);
    letter-spacing: 2px;
    margin-top: 18px;
    margin-bottom: 4px;
}
</style>
""", unsafe_allow_html=True)

model = YOLO("yolov8s.pt")


def predict_image(input_image, output_image):
    results = model.predict(input_image, device="cpu")
    image_rgb = cv2.imread(input_image)
    image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]
            cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (0, 255, 163), 2)
            cls = int(box.cls[0])
            label = model.names[cls]
            cv2.putText(
                image_rgb, f"{label}  {confidence*100:.1f}%",
                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 255), 2,
            )
    image = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)
    cv2.imwrite(output_image, image)
    return output_image


def predict_and_plot_video(input_path, output_path):
    try:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            st.error(f"Could not open video: {input_path}")
            return None
        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        out = cv2.VideoWriter(output_path, fourcc, fps, (fw, fh))
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model.predict(rgb, device="cpu")
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0]) * 100.0
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 163), 2)
                    cv2.putText(
                        frame, f"{conf:.1f}%", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 255), 2,
                    )
            out.write(frame)
        cap.release()
        out.release()
        return output_path
    except Exception as e:
        st.error(f"Error processing video: {e}")
        return None


def process_media(input_path, output_path):
    ext = os.path.splitext(input_path)[1].lower()
    if ext in [".jpg", ".jpeg", ".png"]:
        return predict_image(input_path, output_path)
    if ext in [".mp4", ".avi", ".mov", ".mkv"]:
        return predict_and_plot_video(input_path, output_path)


st.markdown("""
<div class="logo-bar">
    <div class="logo-icon">🎯</div>
    <div>
        <div class="logo-title">VISION·X</div>
        <div class="logo-sub">REAL-TIME OBJECT DETECTION SYSTEM</div>
    </div>
</div>
<div class="status-row">
    <div class="status-dot"></div>
    MODEL ONLINE &nbsp;·&nbsp; YOLOv8x &nbsp;·&nbsp; CPU INFERENCE
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-label">// Select Input Mode</div>', unsafe_allow_html=True)
choice = st.selectbox(
    "Input Mode",
    ["Image", "Video", "Camera"],
    label_visibility="collapsed",
)

st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

os.makedirs("temp", exist_ok=True)
os.makedirs("output", exist_ok=True)
if choice == "Image":
    st.markdown('<div class="section-label">// Upload Image</div>', unsafe_allow_html=True)
    image_data = st.file_uploader(
        "Upload",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )
    clicked = st.button("⚡  RUN DETECTION")
    if clicked:
        if image_data is not None:
            input_path  = f"temp/{image_data.name}"
            output_path = f"output/{image_data.name}"
            try:
                with open(input_path, "wb") as f:
                    f.write(image_data.getbuffer())
                with st.spinner("Analyzing frame..."):
                    result_path = process_media(input_path, output_path)
                st.markdown('<div class="result-label">// Detection Output</div>', unsafe_allow_html=True)
                st.image(result_path)
            except Exception as e:
                st.error(f"Processing error: {e}")
            finally:
                st.feedback("stars")
        else:
            st.warning("Please upload an image first.")

elif choice == "Video":
    st.markdown('<div class="section-label">// Upload Video <= 10mb</div>', unsafe_allow_html=True)
    video_data = st.file_uploader(
        "Upload",
        type=["mp4", "avi", "mov", "mkv"],
        label_visibility="collapsed",
    )
    clicked = st.button("⚡  RUN DETECTION")
    if clicked:
        if video_data is not None:
            input_path  = f"temp/{video_data.name}"
            output_path = f"output/{video_data.name}"
            try:
                with open(input_path, "wb") as f:
                    f.write(video_data.getbuffer())
                with st.spinner("Processing video frames..."):
                    result_path = process_media(input_path, output_path)
                st.markdown('<div class="result-label">// Detection Output</div>', unsafe_allow_html=True)
                with open(result_path, "rb") as vf:
                    st.video(vf.read())
            except Exception as e:
                st.error(f"Processing error: {e}")
            finally:
                st.feedback("stars")
        else:
            st.warning("Please upload a video first.")

elif choice == "Camera":
    st.markdown('<div class="section-label">// Capture Frame</div>', unsafe_allow_html=True)
    camera_data = st.camera_input("Camera", label_visibility="collapsed")
    clicked = st.button("⚡  RUN DETECTION")
    if clicked:
        if camera_data is not None:
            input_path  = f"temp/camera_capture.jpg"
            output_path = f"output/camera_capture.jpg"
            try:
                with open(input_path, "wb") as f:
                    f.write(camera_data.getbuffer())
                with st.spinner("Analyzing frame..."):
                    result_path = process_media(input_path, output_path)
                st.markdown('<div class="result-label">// Detection Output</div>', unsafe_allow_html=True)
                st.image(result_path)
            except Exception as e:
                st.error(f"Processing error: {e}")
            finally:
                st.feedback("stars")
        else:
            st.warning("Please capture a photo first.")

            