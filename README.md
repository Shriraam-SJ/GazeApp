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

Use a different camera index if needed:

```bash
python main.py --camera 1
```

Optional trained model weights:

```bash
python main.py --model path\to\weights.pt
```

Press `q` or `Esc` in the dashboard window to exit.

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
