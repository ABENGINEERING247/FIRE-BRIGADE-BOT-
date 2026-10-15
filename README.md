# 🔥 FIRE BRIGADE BOT — Streamlit AI Fire Watch

A prototype Streamlit dashboard for checking live camera frames, images, or uploaded videos for fire/smoke-like classes using a **fire-trained Ultralytics YOLO model**.

## Features
- Live camera monitoring through WebRTC
- Fire/smoke bounding-box overlay
- Visual alarm state and browser tone attempt
- Upload-image and upload-video inspection modes
- Confidence slider and configurable class labels
- Streamlit Cloud / local deployment guidance

## Important: provide a fire-trained model
The app expects model weights at `models/fire.pt` by default. Generic YOLO/COCO weights do **not** reliably identify fire or smoke. Supply a model trained on suitable fire/smoke data, with class names such as `fire`, `flame`, and/or `smoke`.

Alternatively set `FIRE_MODEL_PATH` to your model path. Do not commit private or licensed model weights to a public repository unless you have permission.

## Run locally
Use Python 3.10 or 3.11 for the best package compatibility.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
mkdir models
# Copy your fire-trained model to models/fire.pt
streamlit run app.py
```

## Deploy
1. Push `app.py`, `requirements.txt`, `README.md`, and your permitted model file to GitHub.
2. Create a Streamlit Community Cloud app from the repository.
3. Configure the app's main file as `app.py`.
4. Large model files may exceed repository/deployment limits. In that case, store the model in an authorized external model host and download it securely during startup, or use an inference endpoint.
5. Live camera requires browser camera permission, HTTPS, and WebRTC-compatible networking. Local webcam access may not work from all cloud/mobile browsers.

## Alarm behavior
The app displays an in-page alarm state and attempts a short browser tone. Browsers may block automatic sound until the user interacts with the page; test sound on the target device. Streamlit reruns and browser policies mean this is not a guaranteed unattended alarm channel.

## Safety limitations
- Not a certified life-safety product.
- Detection can miss real fires or flag unrelated bright/orange objects.
- Do not use as the only fire detection system.
- Use certified smoke/heat detectors and approved alarm panels for real installations.
- On suspected fire: evacuate, follow site procedures, and contact local emergency services.
