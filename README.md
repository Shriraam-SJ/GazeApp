# GazeApp Attention Dashboard

Realtime attention-span monitoring from a webcam. The app captures a face, derives gaze vectors, blink activity, and facial action-unit-like descriptors, then displays an attention dashboard with interpretation, confidence, calibration progress, and system health telemetry.

## What Changed

- Multi-Modal Transformer in `models/temporal_model.py`
  - Cross-attention fusion for gaze and facial AU streams.
  - Reliability gating so missing or low-quality streams receive less weight.
  - Rotary positional embeddings for the 150-frame temporal window.
  - Softmax confidence thresholding. Confidence below `0.6` becomes `Data Uncertain`.
  - Factory pattern for future streams such as EEG or heart rate.
  - Subject baseline calibration for the first 10 seconds.
  - ONNX export hook for TensorRT/OpenVINO-style deployment.

- Vectorized attention scoring in `src/attention_score.py`
  - PoR Persistence 2.0.
  - Saccadic suppression during rapid eye movement.
  - Leaky integrator for smooth transitions.
  - Bounded realtime history buffers.

- Deployment dashboard in `main.py`
  - Opens the camera using OpenCV.
  - Extracts face features using MediaPipe FaceMesh.
  - Runs capture and inference on separate threads.
  - Shows a live dashboard with Focused, Mind-Wandering, Drowsy, and Data Uncertain states.
  - Logs frame-by-frame system health to `attention_monitor.log`.

## Install

```bash
pip install -r requirements.txt
```

Python 3.10 or newer is recommended. A webcam is required for the live dashboard.

## Run The Dashboard

```bash
python main.py
```

The app now opens a small Tkinter control panel first. Detection remains idle until you click **Start**. Click **Stop** to halt the camera feed and generate the end-of-session report.

Use a different camera index if needed:

```bash
python main.py --camera 1
```

Optional trained model weights:

```bash
python main.py --model path\to\weights.pt
```

For automated runs or environments where the control panel is not needed:

```bash
python main.py --no-control
```

Press `q` or `Esc` in the dashboard window to stop the active session.

## Temporal Attention Window And Session Reporting

### Tech Stack

- Python for orchestration, buffering, telemetry, and UI control.
- OpenCV for webcam capture, frame rendering, dashboard drawing, and Haar fallback detection.
- MediaPipe FaceMesh when available for face landmarks, iris location, blink cues, and gaze vector extraction.
- PyTorch `MultiModalTransformer` backbone in `models/temporal_model.py`.
- Transformer/CNN backbone details:
  - Gaze and AU streams are projected into a 128-dimensional token space.
  - `CrossAttentionFusion` performs bidirectional gaze-to-AU and AU-to-gaze fusion with reliability gating.
  - `BlinkFeatureExtractor` is a 1D-CNN blink encoder.
  - `TemporalTransformerEncoder` uses rotary positional embeddings over the temporal token sequence.
- ONNX export is supported through `MultiModalTransformer.export_to_onnx(...)`; the exported graph can be compiled for TensorRT, ONNX Runtime, OpenVINO, or another accelerator backend.

### Spatiotemporal Fusion Logic

Each camera frame is converted into structured signals: a normalized 3D gaze vector, blink intensity, facial AU-like descriptors, and stream-validity flags. The model path uses these signals as temporal tokens for multimodal Transformer fusion, while the realtime interpretation layer also maintains a rolling 2-second PoR buffer.

The 2-second buffer performs state inference every 2 seconds. Gaze vectors are normalized and smoothed with a trailing weighted moving average, which suppresses high-frequency head shake, micro-jitter, and single-frame eye noise. After smoothing, the system evaluates whether the point of regard is inside the target zone. A distraction is only emitted when the smoothed PoR remains outside that zone for more than 60% of the rolling 2-second window.

This means brief shakes and saccades do not immediately flip the state. The temporal state is fused back into the dashboard interpretation so the UI can distinguish stable focus from sustained attention drift.

### Operational Flow

1. Launch `python main.py`.
2. The Tkinter control panel appears and the detection pipeline stays idle.
3. Click **Start** to open the camera feed, begin calibration, start threaded feature extraction, and activate telemetry.
4. During the active session, GazeApp tracks `Total_Active_Time`, `Focused_Duration`, and `Distracted_Duration`.
5. Every 2 seconds, the temporal buffer emits a stable state using the 60% outside-zone rule.
6. Click **Stop**, or press `q`/`Esc` in the OpenCV dashboard, to halt the camera feed.
7. GazeApp generates an end-of-session report with total session length, attention ratio (`Focused_Duration / Total_Active_Time`), and a timestamped distraction-spike log.

## Calibration

For the first 10 seconds, the app learns a subject-specific baseline:

- Gaze origin
- Blink mean and standard deviation
- Facial descriptor mean

During calibration, the dashboard still reports a heuristic attention interpretation. After calibration, the MMT backend is also used when the 150-frame window is full.

## Tests

```bash
python test_system.py
python test_system_production.py
```

The production smoke test does not require a physical camera.

## Deployment Notes

The default runtime uses eager PyTorch because it is the most reliable path on a standard Windows CPU/GPU setup. To prepare for TensorRT/OpenVINO deployment, export the model to ONNX:

```python
from models.temporal_model import MultiModalTransformer

model = MultiModalTransformer()
model.export_to_onnx("gazeapp_mmt.onnx")
```

That ONNX artifact can be compiled with your target accelerator stack. The realtime pipeline is already structured around fixed-size NumPy buffers and backend isolation, so an ONNX Runtime, TensorRT, or OpenVINO backend can replace `TorchInferenceBackend` without changing capture or dashboard code.

## Current Practical Limit

The app is fully runnable and performs live analysis, but classification quality depends on trained weights. Without a trained checkpoint, the dashboard uses the calibrated PoR/blink heuristic as the dependable interpretation layer and treats low-confidence model output as `Data Uncertain`.

If your installed MediaPipe package does not expose the older `mp.solutions.face_mesh` API, the app automatically falls back to OpenCV Haar face/eye detection. FaceMesh gives richer gaze features, but the fallback keeps the dashboard working on Python/package combinations where MediaPipe only ships the newer `tasks` API.
