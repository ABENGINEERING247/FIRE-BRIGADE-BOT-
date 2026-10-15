import os
import time
import threading
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

# FIRE BRIGADE BOT — AI fire detection and alarm dashboard
# Place a fire-trained Ultralytics YOLO model at models/fire.pt, or set FIRE_MODEL_PATH.

st.set_page_config(
    page_title="FIRE BRIGADE BOT | AI Fire Watch",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# LIGHT THEME & SPARKLE ALERT MODAL STYLES
# ---------------------------------------------------------
st.markdown("""
<style>
/* Base Light Theme Background */
.stApp {
    background: #f8fafc;
    color: #0f172a;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}

/* Sidebar Light Styling */
section[data-testid="stSidebar"] {
    background-color: #ffffff !important;
    border-right: 1px solid #e2e8f0;
}

/* Header Hero Container */
.hero {
    padding: 1.5rem 2rem;
    border: 1px solid #fecaca;
    border-radius: 16px;
    background: linear-gradient(135deg, #fff5f5 0%, #fee2e2 100%);
    box-shadow: 0 4px 12px rgba(239, 68, 68, 0.08);
    margin-bottom: 1.5rem;
}
.hero h1 {
    margin: 0;
    color: #dc2626;
    font-size: 2.5rem;
    font-weight: 800;
    letter-spacing: -0.025em;
}
.hero p {
    margin: 0.4rem 0 0;
    color: #475569;
    font-size: 1.05rem;
    font-weight: 500;
}

/* Status Cards */
.metric-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.25rem;
    box-shadow: 0 2px 4px rgba(0,0,0,0.02);
}
.safe-box {
    background: #f0fdf4;
    border: 1.5px solid #22c55e;
    padding: 1.25rem;
    border-radius: 12px;
    color: #15803d;
    font-weight: 600;
    font-size: 1.1rem;
}

/* ---------------------------------------------------------
   ULTRA BIG FIRE BRIGADE SPARKLE POP-UP ALERT
   --------------------------------------------------------- */
@keyframes sparkAnim {
    0% { transform: translate(0, 0) scale(1) rotate(0deg); opacity: 1; }
    50% { transform: translate(var(--dx), var(--dy)) scale(1.4) rotate(180deg); opacity: 0.8; }
    100% { transform: translate(0, 0) scale(1) rotate(360deg); opacity: 1; }
}

@keyframes pulseGlow {
    0% { box-shadow: 0 0 25px rgba(220, 38, 38, 0.6), inset 0 0 15px rgba(239, 68, 68, 0.3); }
    50% { box-shadow: 0 0 55px rgba(220, 38, 38, 0.9), inset 0 0 30px rgba(255, 100, 0, 0.5); }
    100% { box-shadow: 0 0 25px rgba(220, 38, 38, 0.6), inset 0 0 15px rgba(239, 68, 68, 0.3); }
}

.fire-alert-overlay {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    background: rgba(15, 23, 42, 0.75);
    backdrop-filter: blur(8px);
    z-index: 999999;
    display: flex;
    justify-content: center;
    align-items: center;
    animation: fadeIn 0.3s ease-out;
}

.fire-alert-modal {
    position: relative;
    background: #ffffff;
    border: 4px solid #dc2626;
    border-radius: 24px;
    padding: 2.5rem 3rem;
    text-align: center;
    max-width: 650px;
    width: 90%;
    animation: pulseGlow 1.5s infinite alternate;
}

.fire-alert-title {
    font-size: 2.8rem;
    font-weight: 900;
    color: #dc2626;
    margin: 0 0 0.5rem 0;
    text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.fire-alert-subtitle {
    font-size: 1.25rem;
    color: #475569;
    font-weight: 600;
    margin-bottom: 1.5rem;
}

/* Dynamic Sparkle Elements */
.sparkle {
    position: absolute;
    width: 14px;
    height: 14px;
    background: #f59e0b;
    border-radius: 50%;
    box-shadow: 0 0 10px #f59e0b, 0 0 20px #ef4444;
}
.spark1 { top: -15px; left: 10%; --dx: -20px; --dy: -30px; animation: sparkAnim 1.2s infinite ease-in-out; }
.spark2 { top: -15px; right: 10%; --dx: 20px; --dy: -25px; animation: sparkAnim 1.5s infinite ease-in-out 0.2s; }
.spark3 { bottom: -15px; left: 15%; --dx: -15px; --dy: 30px; animation: sparkAnim 1.1s infinite ease-in-out 0.4s; }
.spark4 { bottom: -15px; right: 15%; --dx: 25px; --dy: 20px; animation: sparkAnim 1.3s infinite ease-in-out 0.1s; }
.spark5 { top: 50%; left: -20px; --dx: -35px; --dy: -10px; animation: sparkAnim 1.4s infinite ease-in-out 0.3s; }
.spark6 { top: 50%; right: -20px; --dx: 35px; --dy: 10px; animation: sparkAnim 1.6s infinite ease-in-out 0.5s; }

.small-note {
    color: #64748b;
    font-size: 0.88rem;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# HERO BANNER
# ---------------------------------------------------------
st.markdown("""
<div class="hero">
  <h1>🔥 FIRE BRIGADE BOT</h1>
  <p>AI-powered fire watch • Live camera monitoring • Visual warning • Audible alarm</p>
</div>
""", unsafe_allow_html=True)

MODEL_PATH = os.getenv("FIRE_MODEL_PATH", "models/fire.pt")

@st.cache_resource(show_spinner=False)
def load_model(model_path):
    from ultralytics import YOLO
    return YOLO(model_path)

def get_model():
    if not Path(MODEL_PATH).exists():
        return None, f"Model file not found: {MODEL_PATH}"
    try:
        return load_model(MODEL_PATH), None
    except Exception as exc:
        return None, f"Could not load model: {exc}"

def detect_fire(model, frame, confidence=0.35, fire_labels=None):
    """Run YOLO and return annotated frame, fire flag, and matching detections."""
    result = model.predict(frame, conf=confidence, verbose=False)[0]
    names = result.names if hasattr(result, "names") else {}
    fire_labels = {x.strip().lower() for x in (fire_labels or ["fire", "flame", "smoke"])}
    found = []
    annotated = frame.copy()
    if result.boxes is not None:
        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            score = float(box.conf[0].item())
            label = str(names.get(cls_id, cls_id)).lower()
            if any(term in label for term in fire_labels):
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                found.append((label, score, (x1, y1, x2, y2)))
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(annotated, f"FIRE RISK: {label} {score:.0%}",
                            (x1, max(25, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 255), 2, cv2.LINE_AA)
    return annotated, bool(found), found

def make_alarm_html():
    """Generates the Ultra Big Fire Brigade Pop-Up and Audio Alert."""
    return """
    <div class="fire-alert-overlay">
      <div class="fire-alert-modal">
        <div class="sparkle spark1"></div>
        <div class="sparkle spark2"></div>
        <div class="sparkle spark3"></div>
        <div class="sparkle spark4"></div>
        <div class="sparkle spark5"></div>
        <div class="sparkle spark6"></div>
        <div class="fire-alert-title">🚨 FIRE DETECTED! 🚨</div>
        <div class="fire-alert-subtitle">
            FIRE BRIGADE BOT ALERT: Fire/Smoke signature detected in the video stream!
        </div>
        <div style="background:#fef2f2; border:1px solid #fca5a5; padding:12px; border-radius:10px; color:#991b1b; font-weight:600;">
            🔊 AUDIBLE ALARM ACTIVE — Follow immediate safety procedures and verify visually.
        </div>
      </div>
    </div>
    <audio autoplay loop>
      <source src="data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA=" type="audio/wav">
    </audio>
    <script>
      try {
        const ctx = new (window.AudioContext || window.webkitAudioContext)();
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.type = 'sawtooth';
        osc.frequency.value = 880;
        gain.gain.value = 0.035;
        osc.connect(gain);
        gain.connect(ctx.destination);
        osc.start();
        setTimeout(() => { osc.stop(); ctx.close(); }, 900);
      } catch(e) {}
    </script>
    """

# ---------------------------------------------------------
# SIDEBAR SETTINGS
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Monitoring Settings")
    mode = st.radio("Video source", ["Live camera", "Upload image", "Upload video"], index=0)
    confidence = st.slider("Detection confidence", 0.10, 0.90, 0.35, 0.05)
    label_text = st.text_input("Fire-related class names (comma-separated)", "fire, flame, smoke")
    fire_labels = [s.strip().lower() for s in label_text.split(",") if s.strip()]
    st.divider()
    st.markdown("**Model status**")
    st.caption(f"Expected model: `{MODEL_PATH}`")
    st.caption("Use a fire-trained model. COCO/general-purpose YOLO weights usually do not detect fire reliably.")
    st.divider()
    st.markdown("**Emergency checklist**")
    st.caption("1. Do not approach the fire.")
    st.caption("2. Evacuate and alert people nearby.")
    st.caption("3. Call your local fire/emergency service.")
    st.caption("4. Never rely on this prototype as the only safety system.")

model, model_error = get_model()
if model is None:
    st.error("AI model is not ready. " + str(model_error))
    st.info("Add your fire-trained YOLO weights as `models/fire.pt`, or set the `FIRE_MODEL_PATH` environment variable. This app intentionally does not pretend a generic model can reliably identify fire.")
    st.markdown("### Quick setup")
    st.code("mkdir models\n# Copy your fire-trained YOLO weights to models/fire.pt\npip install -r requirements.txt", language="bash")
    st.stop()

# ---------------------------------------------------------
# MAIN DASHBOARD LAYOUT
# ---------------------------------------------------------
left, right = st.columns([1.6, 1], gap="large")
with left:
    st.subheader("📹 Monitoring Console")
    preview = st.empty()
with right:
    st.subheader("🚨 System Status")
    status_area = st.empty()
    count_area = st.empty()
    alarm_area = st.empty()

if mode == "Live camera":
    st.warning("Allow camera access if prompted. For remote deployment, WebRTC requires a secure HTTPS connection.")
    
    class FireVideoProcessor(VideoProcessorBase):
        def __init__(self):
            self.model = model
            self.confidence = confidence
            self.fire_labels = fire_labels
            self.fire_detected = False
            self.detections = []
            self.last_frame = None
            self.lock = threading.Lock()

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            try:
                annotated, detected, found = detect_fire(self.model, img, self.confidence, self.fire_labels)
            except Exception:
                annotated, detected, found = img, False, []
            with self.lock:
                self.fire_detected = detected
                self.detections = found
                self.last_frame = annotated
            return frame.from_ndarray(annotated, format="bgr24")

    ctx = webrtc_streamer(
        key="fire-brigade-live",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=FireVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    if ctx.video_processor:
        processor = ctx.video_processor
        with processor.lock:
            detected = processor.fire_detected
            detections = list(processor.detections)
            last = processor.last_frame.copy() if processor.last_frame is not None else None
        if last is not None:
            preview.image(cv2.cvtColor(last, cv2.COLOR_BGR2RGB), caption="Live AI annotated camera feed", use_container_width=True)
        if detected:
            status_area.markdown('<div class="fire-alert-subtitle" style="color:#dc2626; font-weight:700;">🔥 FIRE DETECTED</div>', unsafe_allow_html=True)
            count_area.metric("Matching detections", len(detections))
            alarm_area.markdown(make_alarm_html(), unsafe_allow_html=True)
            st.error("Take immediate precautions. Confirm visually, evacuate if needed, and contact emergency services.")
        else:
            status_area.markdown('<div class="safe-box">🟢 No matching fire class detected in the latest frame</div>', unsafe_allow_html=True)
            count_area.metric("Matching detections", 0)
            alarm_area.info("Alarm standby")
    else:
        status_area.info("Camera is waiting to start. Press START above and grant camera permission.")
        count_area.metric("Matching detections", "—")
        alarm_area.info("Alarm standby")

elif mode == "Upload image":
    upload = st.file_uploader("Upload an image to inspect", type=["jpg", "jpeg", "png", "webp"])
    if upload:
        rgb = np.array(Image.open(upload).convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        annotated, detected, detections = detect_fire(model, bgr, confidence, fire_labels)
        preview.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="AI inspection result", use_container_width=True)
        if detected:
            status_area.markdown('<div class="fire-alert-subtitle" style="color:#dc2626; font-weight:700;">🔥 FIRE DETECTED</div>', unsafe_allow_html=True)
            alarm_area.markdown(make_alarm_html(), unsafe_allow_html=True)
            st.error("Possible fire-related object found. Verify immediately and follow your site emergency plan.")
        else:
            status_area.markdown('<div class="safe-box">🟢 No matching fire class detected</div>', unsafe_allow_html=True)
            alarm_area.info("Alarm standby")
        count_area.metric("Matching detections", len(detections))
        if detections:
            st.dataframe([{"class": label, "confidence": round(score, 3)} for label, score, _ in detections], use_container_width=True)

else:
    upload = st.file_uploader("Upload a video to inspect", type=["mp4", "mov", "avi", "mkv"])
    if upload:
        temp_path = Path("uploaded_fire_check.mp4")
        temp_path.write_bytes(upload.getbuffer())
        cap = cv2.VideoCapture(str(temp_path))
        total, fire_frames, checked = 0, 0, 0
        max_frames = st.slider("Maximum frames to analyze", 30, 600, 180, 30)
        progress = st.progress(0)
        run = st.button("▶ Analyze video", type="primary")
        if run:
            while checked < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                total += 1
                if total % 2 == 0:
                    annotated, detected, detections = detect_fire(model, frame, confidence, fire_labels)
                    checked += 1
                    if detected:
                        fire_frames += 1
                        preview.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption=f"Analyzed frame {total}", use_container_width=True)
                        status_area.markdown('<div class="fire-alert-subtitle" style="color:#dc2626; font-weight:700;">🔥 FIRE DETECTED IN VIDEO</div>', unsafe_allow_html=True)
                        alarm_area.markdown(make_alarm_html(), unsafe_allow_html=True)
                    else:
                        preview.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption=f"Analyzed frame {total}", use_container_width=True)
                        status_area.markdown('<div class="safe-box">🟢 No matching class in this frame</div>', unsafe_allow_html=True)
                    count_area.metric("Frames with fire-like detections", fire_frames)
                    progress.progress(min(checked / max_frames, 1.0))
            cap.release()
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass
            if fire_frames:
                st.error(f"Review required: matching detections appeared in {fire_frames} analyzed frames.")
            else:
                st.success("No matching fire-related class was detected in the analyzed frames.")

st.divider()
st.markdown("""
### ℹ️ Important safety notice
This is a prototype decision-support tool, not a certified fire alarm or fire-suppression system. Accuracy depends on the model, camera angle, lighting, smoke conditions, and training data.
""")
st.markdown('<p class="small-note">FIRE BRIGADE BOT • AI safety prototype • Built with Streamlit, OpenCV, WebRTC and a fire-trained YOLO model</p>', unsafe_allow_html=True)
