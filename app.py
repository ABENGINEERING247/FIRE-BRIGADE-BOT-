import os
import time
import threading
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

# PDF & Word generation imports
from fpdf import FPDF
from docx import Document

# FIRE BRIGADE BOT — AI fire detection and alarm dashboard
# Designed and engineered by Certified Generative and Agentic AI Developer Engr Bilal Mehmood
# Place a fire-trained Ultralytics YOLO model at models/fire.pt, or set FIRE_MODEL_PATH.

st.set_page_config(
    page_title="FIRE BRIGADE BOT | AI Fire Watch",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# LIGHT THEME & ULTRA BIG SPARKLE ALERT MODAL STYLES
# ---------------------------------------------------------
st.markdown("""
<style>
.stApp {
    background: #f8fafc;
    color: #0f172a;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
section[data-testid="stSidebar"] {
    background-color: #ffffff !important;
    border-right: 1px solid #e2e8f0;
}
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
.dev-badge {
    display: inline-block;
    margin-top: 0.6rem;
    background: #dc2626;
    color: #ffffff;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.025em;
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

@keyframes sparkAnim {
    0% { transform: translate(0, 0) scale(1) rotate(0deg); opacity: 1; }
    50% { transform: translate(var(--dx), var(--dy)) scale(1.6) rotate(180deg); opacity: 0.9; }
    100% { transform: translate(0, 0) scale(1) rotate(360deg); opacity: 1; }
}

@keyframes pulseGlow {
    0% { box-shadow: 0 0 30px rgba(220, 38, 38, 0.6), inset 0 0 20px rgba(239, 68, 68, 0.3); }
    50% { box-shadow: 0 0 70px rgba(220, 38, 38, 0.95), inset 0 0 40px rgba(255, 100, 0, 0.6); }
    100% { box-shadow: 0 0 30px rgba(220, 38, 38, 0.6), inset 0 0 20px rgba(239, 68, 68, 0.3); }
}

.fire-alert-overlay {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    background: rgba(15, 23, 42, 0.8);
    backdrop-filter: blur(10px);
    z-index: 999999;
    display: flex;
    justify-content: center;
    align-items: center;
}

.fire-alert-modal {
    position: relative;
    background: #ffffff;
    border: 5px solid #dc2626;
    border-radius: 28px;
    padding: 3rem 3.5rem;
    text-align: center;
    max-width: 700px;
    width: 92%;
    animation: pulseGlow 1.4s infinite alternate;
}

.fire-alert-title {
    font-size: 3.2rem;
    font-weight: 900;
    color: #dc2626;
    margin: 0 0 0.75rem 0;
    text-shadow: 0 3px 6px rgba(0, 0, 0, 0.12);
}

.fire-alert-subtitle {
    font-size: 1.35rem;
    color: #334155;
    font-weight: 700;
    margin-bottom: 1.75rem;
}

.sparkle {
    position: absolute;
    width: 18px;
    height: 18px;
    background: #f59e0b;
    border-radius: 50%;
    box-shadow: 0 0 12px #f59e0b, 0 0 24px #ef4444, 0 0 36px #ffffff;
}
.spark1 { top: -20px; left: 8%; --dx: -25px; --dy: -35px; animation: sparkAnim 1.1s infinite ease-in-out; }
.spark2 { top: -20px; right: 8%; --dx: 25px; --dy: -30px; animation: sparkAnim 1.4s infinite ease-in-out 0.2s; }
.spark3 { bottom: -20px; left: 12%; --dx: -20px; --dy: 35px; animation: sparkAnim 1.0s infinite ease-in-out 0.4s; }
.spark4 { bottom: -20px; right: 12%; --dx: 30px; --dy: 25px; animation: sparkAnim 1.3s infinite ease-in-out 0.1s; }
.spark5 { top: 45%; left: -25px; --dx: -40px; --dy: -15px; animation: sparkAnim 1.5s infinite ease-in-out 0.3s; }
.spark6 { top: 45%; right: -25px; --dx: 40px; --dy: 15px; animation: sparkAnim 1.2s infinite ease-in-out 0.5s; }

.small-note {
    color: #64748b;
    font-size: 0.88rem;
}
</style>
""", unsafe_allow_html=True)

# Session State Initialization for Incident Log History
if "incident_logs" not in st.session_state:
    st.session_state.incident_logs = []

# ---------------------------------------------------------
# WELCOME DIALOG POPUP
# ---------------------------------------------------------
if "welcome_shown" not in st.session_state:
    st.session_state.welcome_shown = False

@st.dialog("🔥 Welcome to AI-Powered Fire Brigade Bot", width="large")
def show_welcome_dialog():
    st.markdown("""
    ### 🛡️ Intelligent Early Fire Detection & Emergency Watch System
    
    Welcome to the **AI-Powered Fire Brigade Bot**! This application features live camera monitoring, acoustic/visual popup alarms across mobile and PC platforms, automated incident tracking, and report generation in Tabular, PDF, and Word formats.
    
    #### 🌟 Key Capabilities:
    * **Live AI Monitoring:** Stream live video feeds with YOLO-based fire and smoke detection.
    * **Cross-Device Sound & Popup Alarms:** Automated siren triggers on any PC, laptop, or mobile screen.
    * **Exportable Audit Reports:** Log incidents and download structured reports in PDF and Word (.docx).
    
    ---
    *Designed and Engineered by **Engr Bilal Mehmood** (Certified Generative and Agentic AI Developer).*
    """)
    if st.button("🚀 Enter Dashboard", type="primary", use_container_width=True):
        st.session_state.welcome_shown = True
        st.rerun()

if not st.session_state.welcome_shown:
    show_welcome_dialog()

# ---------------------------------------------------------
# HERO BANNER WITH DEVELOPER CREDITS
# ---------------------------------------------------------
st.markdown("""
<div class="hero">
  <h1>🔥 FIRE BRIGADE BOT</h1>
  <p>AI-powered fire watch • Live camera monitoring • Cross-device sound/visual alarms • Report Generator</p>
  <div class="dev-badge">👨‍💻 Designed & Developed by Engr Bilal Mehmood (Certified GenAI & Agentic AI Developer)</div>
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
    """Generates the Ultra Big Fire Brigade Sparkle Pop-Up and Cross-Platform Sound Alarm."""
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
            FIRE BRIGADE BOT ALERT: Active Fire or Smoke signature detected on your device!
        </div>
        <div style="background:#fef2f2; border:1px solid #fca5a5; padding:14px; border-radius:12px; color:#991b1b; font-weight:700; font-size:1.05rem;">
            🔊 EMERGENCY SOUND & VISUAL ALARM ACTIVE — Take immediate safety precautions!
        </div>
        <div style="margin-top:18px; font-size:0.9rem; color:#475569; font-weight:600;">
            Architecture & Design by Engr Bilal Mehmood
        </div>
      </div>
    </div>
    <script>
      try {
        const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        if (audioCtx.state === 'suspended') { audioCtx.resume(); }
        const osc = audioCtx.createOscillator();
        const gainNode = audioCtx.createGain();
        osc.type = 'sawtooth';
        osc.frequency.setValueAtTime(880, audioCtx.currentTime);
        // Siren frequency modulation for mobile & PC hardware speakers
        osc.frequency.exponentialRampToValueAtTime(1200, audioCtx.currentTime + 0.5);
        osc.frequency.exponentialRampToValueAtTime(880, audioCtx.currentTime + 1.0);
        gainNode.gain.setValueAtTime(0.15, audioCtx.currentTime);
        osc.connect(gainNode);
        gainNode.connect(audioCtx.destination);
        osc.start();
        setTimeout(() => { osc.stop(); audioCtx.close(); }, 1200);
      } catch(e) {}
    </script>
    """

# Report Exporters
def generate_pdf_report(logs):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, txt="FIRE BRIGADE BOT - INCIDENT REPORT", ln=True, align="C")
    pdf.set_font("Arial", "", 10)
    pdf.cell(200, 8, txt=f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Engr Bilal Mehmood", ln=True, align="C")
    pdf.ln(10)
    
    pdf.set_font("Arial", "B", 11)
    pdf.cell(50, 10, "Timestamp", 1)
    pdf.cell(50, 10, "Detection Class", 1)
    pdf.cell(40, 10, "Confidence", 1)
    pdf.cell(50, 10, "Status", 1)
    pdf.ln()
    
    pdf.set_font("Arial", "", 10)
    for log in logs:
        pdf.cell(50, 10, str(log.get("Timestamp", "")), 1)
        pdf.cell(50, 10, str(log.get("Class", "")), 1)
        pdf.cell(40, 10, str(log.get("Confidence", "")), 1)
        pdf.cell(50, 10, str(log.get("Status", "")), 1)
        pdf.ln()
    return pdf.output(dest='S').encode('latin1')

def generate_docx_report(logs):
    doc = Document()
    doc.add_heading("FIRE BRIGADE BOT - INCIDENT REPORT", 0)
    doc.add_paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    doc.add_paragraph("Designed and Engineered by Engr Bilal Mehmood (Certified GenAI & Agentic AI Developer)")
    doc.add_heading("Detected Fire Events", level=1)
    
    table = doc.add_table(rows=1, cols=4)
    hdr_cells = table.rows[0].cells
    hdr_cells[0].text = 'Timestamp'
    hdr_cells[1].text = 'Detection Class'
    hdr_cells[2].text = 'Confidence'
    hdr_cells[3].text = 'Status'
    
    for log in logs:
        row_cells = table.add_row().cells
        row_cells[0].text = str(log.get("Timestamp", ""))
        row_cells[1].text = str(log.get("Class", ""))
        row_cells[2].text = str(log.get("Confidence", ""))
        row_cells[3].text = str(log.get("Status", ""))
        
    from io import BytesIO
    bio = BytesIO()
    doc.save(bio)
    bio.seek(0)
    return bio

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
    st.caption("Use a fire-trained model for accurate flame and smoke detection.")
    st.divider()
    st.markdown("**Emergency checklist**")
    st.caption("1. Do not approach the fire.")
    st.caption("2. Evacuate and alert people nearby.")
    st.caption("3. Call your local fire/emergency service.")

model, model_error = get_model()
if model is None:
    st.error("AI model is not ready. " + str(model_error))
    st.info("Add your fire-trained YOLO weights as `models/fire.pt`, or set the `FIRE_MODEL_PATH` environment variable.")
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
    st.warning("Allow camera access if prompted. Audio and visual alarms will sound across your PC/laptop or mobile browser.")
    
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
            for label, score, _ in detections:
                st.session_state.incident_logs.append({
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Class": label,
                    "Confidence": f"{score:.2%}",
                    "Status": "ACTIVE FIRE DETECTED"
                })
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
            for label, score, _ in detections:
                st.session_state.incident_logs.append({
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Class": label,
                    "Confidence": f"{score:.2%}",
                    "Status": "IMAGE FIRE DETECTED"
                })
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
                        for label, score, _ in detections:
                            st.session_state.incident_logs.append({
                                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "Class": label,
                                "Confidence": f"{score:.2%}",
                                "Status": f"VIDEO FRAME {total}"
                            })
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

# ---------------------------------------------------------
# TABULAR INCIDENT REPORT & DOWNLOAD CENTER
# ---------------------------------------------------------
st.divider()
st.subheader("📋 Incident Report & Export Center")
if st.session_state.incident_logs:
    st.markdown("Below is the live tabular log of detected fire events during your current session:")
    st.dataframe(st.session_state.incident_logs, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        pdf_data = generate_pdf_report(st.session_state.incident_logs)
        st.download_button(
            label="📄 Download PDF Report",
            data=pdf_data,
            file_name=f"Fire_Incident_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
            use_container_width=True
        )
    with col2:
        docx_data = generate_docx_report(st.session_state.incident_logs)
        st.download_button(
            label="📝 Download Word (.docx) Report",
            data=docx_data,
            file_name=f"Fire_Incident_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True
        )
else:
    st.info("No fire incidents recorded in the current session yet. Tabular logs and report downloads will appear here once events are triggered.")

st.markdown("""
### ℹ️ Important safety notice
This is a prototype decision-support tool, not a certified fire alarm or fire-suppression system. Accuracy depends on the model, camera angle, lighting, smoke conditions, and training data.
""")
st.markdown('<p class="small-note">FIRE BRIGADE BOT • AI safety prototype • Built with Streamlit, OpenCV, WebRTC and a fire-trained YOLO model<br><strong>Engineered and Designed by Engr Bilal Mehmood</strong></p>', unsafe_allow_html=True)
