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

st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #10151f 0%, #1d2028 55%, #35130f 100%); color: #f8fafc; }
.hero { padding: 1.3rem 1.5rem; border: 1px solid #6b2c22; border-radius: 18px;
        background: linear-gradient(100deg, rgba(130,25,15,.55), rgba(30,35,45,.85)); }
.hero h1 { margin: 0; color: #ff6b35; font-size: 2.4rem; }
.hero p { margin: .4rem 0 0; color: #e5e7eb; }
.metric-card { background: rgba(255,255,255,.06); border: 1px solid rgba(255,255,255,.13);
               border-radius: 14px; padding: 1rem; }
.alert-box { background: #5b160f; border: 2px solid #ff3b30; padding: 1rem; border-radius: 12px;
             color: white; font-size: 1.15rem; font-weight: 700; }
.safe-box { background: #123b2b; border: 1px solid #1d9a62; padding: 1rem; border-radius: 12px; color: white; }
.small-note { color: #cbd5e1; font-size: .88rem; }
</style>
""", unsafe_allow_html=True)

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
            # Match common class names; customize the label list in the sidebar.
            if any(term in label for term in fire_labels):
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                found.append((label, score, (x1, y1, x2, y2)))
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(annotated, f"FIRE RISK: {label} {score:.0%}",
                            (x1, max(25, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 255), 2, cv2.LINE_AA)
    return annotated, bool(found), found

def make_alarm_html():
    # Browser policies may block automatic sound until the user interacts with the page.
    return """
    <div style="padding:10px;border-radius:10px;background:#40110d;color:white;border:1px solid #ff3b30">
      <b>🔊 ALARM ACTIVE</b> — Fire-like object detected. Enable browser sound if prompted.
    </div>
    <audio autoplay loop>
      <source src="data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA=" type="audio/wav">
    </audio>
    <script>
      // Short repeating tone using Web Audio API; requires browser audio permission/user gesture.
      try {
        const ctx = new (window.AudioContext || window.webkitAudioContext)();
        const osc = ctx.createOscillator(); const gain = ctx.createGain();
        osc.type = 'sawtooth'; osc.frequency.value = 880; gain.gain.value = 0.035;
        osc.connect(gain); gain.connect(ctx.destination); osc.start();
        setTimeout(()=>{osc.stop();ctx.close();}, 900);
      } catch(e) {}
    </script>
    """

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
    st.warning("Allow camera and microphone access if your browser requests it. For remote deployment, WebRTC requires a secure HTTPS connection and compatible network/firewall settings.")
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
            status_area.markdown('<div class="alert-box">🔥 FIRE / SMOKE-LIKE OBJECT DETECTED</div>', unsafe_allow_html=True)
            count_area.metric("Matching detections", len(detections))
            alarm_area.markdown(make_alarm_html(), unsafe_allow_html=True)
            st.error("Take immediate precautions. Confirm visually, evacuate if needed, and contact emergency services.")
        else:
            status_area.markdown('<div class="safe-box">🟢 No matching fire class detected in the latest processed frame</div>', unsafe_allow_html=True)
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
            status_area.markdown('<div class="alert-box">🔥 FIRE / SMOKE-LIKE OBJECT DETECTED</div>', unsafe_allow_html=True)
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
                # Analyze every second frame for a faster first-pass review.
                if total % 2 == 0:
                    annotated, detected, detections = detect_fire(model, frame, confidence, fire_labels)
                    checked += 1
                    if detected:
                        fire_frames += 1
                        preview.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption=f"Analyzed frame {total}", use_container_width=True)
                        status_area.markdown('<div class="alert-box">🔥 POSSIBLE FIRE DETECTED IN VIDEO</div>', unsafe_allow_html=True)
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
                st.success("No matching fire-related class was detected in the analyzed frames. This does not prove the video is fire-free.")

st.divider()
st.markdown("""
### ℹ️ Important safety notice
This is a prototype decision-support tool, not a certified fire alarm or fire-suppression system. Accuracy depends on the model, camera angle, lighting, smoke conditions, and training data. False positives and missed fires are possible. For real buildings, use certified smoke/heat detectors and approved alarm panels; keep human monitoring and emergency procedures in place.
""")
st.markdown('<p class="small-note">FIRE BRIGADE BOT • AI safety prototype • Built with Streamlit, OpenCV, WebRTC and a fire-trained YOLO model</p>', unsafe_allow_html=True)
