#!/usr/bin/env python3
"""Realtime dashboard for multi-modal attention monitoring."""

import argparse
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from queue import Empty, Queue
from typing import Any, Deque, Dict, Optional, Tuple

import cv2
import numpy as np
import psutil

try:
    import mediapipe as mp
except Exception:
    mp = None

from models.temporal_model import CLASS_NAMES, SubjectBaselineCalibration, TorchInferenceBackend, UNCERTAIN_CLASS
from src.attention_score import AttentionScorer


WINDOW_SIZE = 150
AU_DIM = 17
LOG_PATH = "attention_monitor.log"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()],
)
logger = logging.getLogger("GazeApp")


@dataclass
class FrameFeatures:
    frame_id: int
    timestamp: float
    frame: np.ndarray
    gaze: np.ndarray
    blink: np.ndarray
    au: np.ndarray
    gaze_valid: bool
    au_valid: bool
    face_detected: bool
    stream_sync_offset_ms: float


class Telemetry:
    def __init__(self) -> None:
        self.started_at = time.perf_counter()
        self.frame_count = 0
        self.inference_ms: Deque[float] = deque(maxlen=120)
        self.sync_offsets_ms: Deque[float] = deque(maxlen=120)
        self.capture_ms: Deque[float] = deque(maxlen=120)
        self.process = psutil.Process()

    def update(self, inference_ms: float, sync_offset_ms: float, capture_ms: float = 0.0) -> None:
        self.frame_count += 1
        self.inference_ms.append(float(inference_ms))
        self.sync_offsets_ms.append(float(sync_offset_ms))
        self.capture_ms.append(float(capture_ms))

    def report(self) -> Dict[str, float]:
        elapsed = max(time.perf_counter() - self.started_at, 1e-3)
        return {
            "fps": self.frame_count / elapsed,
            "avg_inference_ms": float(np.mean(self.inference_ms)) if self.inference_ms else 0.0,
            "max_inference_ms": float(np.max(self.inference_ms)) if self.inference_ms else 0.0,
            "avg_capture_ms": float(np.mean(self.capture_ms)) if self.capture_ms else 0.0,
            "cpu_percent": float(psutil.cpu_percent(interval=None)),
            "memory_percent": float(psutil.virtual_memory().percent),
            "process_mb": float(self.process.memory_info().rss / (1024 * 1024)),
            "avg_sync_offset_ms": float(np.mean(self.sync_offsets_ms)) if self.sync_offsets_ms else 0.0,
        }


class FaceFeatureExtractor:
    """Extracts gaze vector, blink signal, and AU-like face descriptors."""

    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]
    LEFT_IRIS = [468, 469, 470, 471]
    RIGHT_IRIS = [473, 474, 475, 476]

    def __init__(self) -> None:
        self.backend = "haar"
        self.face_mesh = None
        if mp is not None and hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
            self.backend = "mediapipe"
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            logger.info("Using MediaPipe FaceMesh feature extractor.")
        else:
            cascade_dir = cv2.data.haarcascades
            self.face_cascade = cv2.CascadeClassifier(cascade_dir + "haarcascade_frontalface_default.xml")
            self.eye_cascade = cv2.CascadeClassifier(cascade_dir + "haarcascade_eye.xml")
            if self.face_cascade.empty() or self.eye_cascade.empty():
                raise RuntimeError("OpenCV Haar cascades are unavailable; cannot detect face/eyes.")
            logger.warning(
                "MediaPipe FaceMesh solutions API is unavailable in this environment; "
                "using OpenCV Haar face/eye fallback."
            )

    @staticmethod
    def _points(landmarks: Any, indices: list, width: int, height: int) -> np.ndarray:
        return np.asarray([(landmarks[i].x * width, landmarks[i].y * height) for i in indices], dtype=np.float32)

    @staticmethod
    def _distance(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))

    def _eye_aspect_ratio(self, points: np.ndarray) -> float:
        horizontal = self._distance(points[0], points[3])
        vertical = self._distance(points[1], points[5]) + self._distance(points[2], points[4])
        return float(vertical / max(2.0 * horizontal, 1e-6))

    def extract(self, frame: np.ndarray, frame_id: int) -> FrameFeatures:
        if self.backend == "haar":
            return self._extract_haar(frame, frame_id)
        return self._extract_mediapipe(frame, frame_id)

    def _missing(self, frame: np.ndarray, frame_id: int, started: float) -> FrameFeatures:
        return FrameFeatures(
            frame_id=frame_id,
            timestamp=time.time(),
            frame=frame,
            gaze=np.zeros(3, dtype=np.float32),
            blink=np.zeros(1, dtype=np.float32),
            au=np.zeros(AU_DIM, dtype=np.float32),
            gaze_valid=False,
            au_valid=False,
            face_detected=False,
            stream_sync_offset_ms=(time.perf_counter() - started) * 1000.0,
        )

    def _extract_haar(self, frame: np.ndarray, frame_id: int) -> FrameFeatures:
        started = time.perf_counter()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(90, 90))
        if len(faces) == 0:
            return self._missing(frame, frame_id, started)

        x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
        cv2.rectangle(frame, (x, y), (x + w, y + h), (105, 220, 135), 2)
        upper_face = gray[y : y + int(h * 0.62), x : x + w]
        eyes = self.eye_cascade.detectMultiScale(upper_face, scaleFactor=1.08, minNeighbors=6, minSize=(20, 20))
        eyes = sorted(eyes, key=lambda box: box[2] * box[3], reverse=True)[:2]

        face_center = np.array([x + w * 0.5, y + h * 0.45], dtype=np.float32)
        eye_centers = []
        eye_ratios = []
        for ex, ey, ew, eh in eyes:
            cx = x + ex + ew * 0.5
            cy = y + ey + eh * 0.5
            eye_centers.append([cx, cy])
            eye_ratios.append(eh / max(ew, 1))
            cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (80, 190, 255), 1)

        if eye_centers:
            eye_center = np.mean(np.asarray(eye_centers, dtype=np.float32), axis=0)
            eye_span = max(float(w), 1.0)
            gaze_xy = (eye_center - face_center) / eye_span
            gaze_valid = True
        else:
            gaze_xy = np.zeros(2, dtype=np.float32)
            eye_span = max(float(w), 1.0)
            gaze_valid = False

        gaze = np.array([gaze_xy[0], gaze_xy[1], 1.0], dtype=np.float32)
        gaze /= max(np.linalg.norm(gaze), 1e-6)
        eye_ratio = float(np.mean(eye_ratios)) if eye_ratios else 0.24
        blink = np.array([np.clip(0.24 - eye_ratio, 0.0, 0.24) / 0.24], dtype=np.float32)

        face_area = (w * h) / max(frame.shape[0] * frame.shape[1], 1)
        aspect = w / max(h, 1)
        au = np.array(
            [
                eye_ratio,
                eye_ratio,
                blink[0],
                aspect,
                face_area,
                (face_center[1] / max(frame.shape[0], 1)),
                (face_center[0] / max(frame.shape[1], 1)),
                w / max(frame.shape[1], 1),
                h / max(frame.shape[0], 1),
                gaze_xy[0],
                gaze_xy[1],
                abs(gaze_xy[0]),
                abs(gaze_xy[1]),
                len(eyes) / 2.0,
                float(gaze_valid),
                0.0,
                1.0,
            ],
            dtype=np.float32,
        )
        return FrameFeatures(
            frame_id=frame_id,
            timestamp=time.time(),
            frame=frame,
            gaze=gaze,
            blink=blink,
            au=au,
            gaze_valid=gaze_valid,
            au_valid=True,
            face_detected=True,
            stream_sync_offset_ms=(time.perf_counter() - started) * 1000.0,
        )

    def _extract_mediapipe(self, frame: np.ndarray, frame_id: int) -> FrameFeatures:
        started = time.perf_counter()
        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb)
        if not result.multi_face_landmarks:
            return self._missing(frame, frame_id, started)

        landmarks = result.multi_face_landmarks[0].landmark
        left_eye = self._points(landmarks, self.LEFT_EYE, width, height)
        right_eye = self._points(landmarks, self.RIGHT_EYE, width, height)
        left_iris = self._points(landmarks, self.LEFT_IRIS, width, height).mean(axis=0)
        right_iris = self._points(landmarks, self.RIGHT_IRIS, width, height).mean(axis=0)
        left_center = left_eye[[0, 3]].mean(axis=0)
        right_center = right_eye[[0, 3]].mean(axis=0)
        face_center = (left_center + right_center) * 0.5
        iris_center = (left_iris + right_iris) * 0.5
        eye_span = max(self._distance(left_eye[0], right_eye[3]), 1.0)
        gaze_xy = (iris_center - face_center) / eye_span
        gaze = np.array([gaze_xy[0], gaze_xy[1], 1.0], dtype=np.float32)
        gaze /= max(np.linalg.norm(gaze), 1e-6)

        left_ear = self._eye_aspect_ratio(left_eye)
        right_ear = self._eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) * 0.5
        blink = np.array([np.clip(0.28 - ear, 0.0, 0.28) / 0.28], dtype=np.float32)

        def lm(idx: int) -> np.ndarray:
            point = landmarks[idx]
            return np.array([point.x * width, point.y * height], dtype=np.float32)

        mouth_w = self._distance(lm(61), lm(291)) / eye_span
        mouth_open = self._distance(lm(13), lm(14)) / eye_span
        brow_lift = ((lm(159)[1] - lm(70)[1]) + (lm(386)[1] - lm(300)[1])) / (2.0 * eye_span)
        face_w = self._distance(lm(234), lm(454)) / max(width, 1)
        face_h = self._distance(lm(10), lm(152)) / max(height, 1)
        nose_x = (lm(1)[0] - face_center[0]) / eye_span
        nose_y = (lm(1)[1] - face_center[1]) / eye_span
        au = np.array(
            [
                left_ear,
                right_ear,
                blink[0],
                mouth_w,
                mouth_open,
                brow_lift,
                face_w,
                face_h,
                nose_x,
                nose_y,
                abs(gaze_xy[0]),
                abs(gaze_xy[1]),
                self._distance(lm(0), lm(17)) / eye_span,
                self._distance(lm(78), lm(308)) / eye_span,
                self._distance(lm(105), lm(334)) / eye_span,
                self._distance(lm(50), lm(280)) / eye_span,
                1.0,
            ],
            dtype=np.float32,
        )
        return FrameFeatures(
            frame_id=frame_id,
            timestamp=time.time(),
            frame=frame,
            gaze=gaze,
            blink=blink,
            au=np.nan_to_num(au),
            gaze_valid=True,
            au_valid=True,
            face_detected=True,
            stream_sync_offset_ms=(time.perf_counter() - started) * 1000.0,
        )


class CameraCaptureThread(threading.Thread):
    def __init__(self, data_queue: Queue, stop_event: threading.Event, camera_index: int = 0):
        super().__init__(daemon=True)
        self.data_queue = data_queue
        self.stop_event = stop_event
        self.camera_index = camera_index
        self.error: Optional[str] = None

    def run(self) -> None:
        try:
            extractor = FaceFeatureExtractor()
            camera = cv2.VideoCapture(self.camera_index)
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
            camera.set(cv2.CAP_PROP_FPS, 30)
            if not camera.isOpened():
                self.error = f"Could not open camera index {self.camera_index}."
                logger.error(self.error)
                self.stop_event.set()
                return
            frame_id = 0
            while not self.stop_event.is_set():
                capture_start = time.perf_counter()
                ok, frame = camera.read()
                if not ok:
                    logger.warning("Camera frame read failed.")
                    time.sleep(0.02)
                    continue
                features = extractor.extract(frame, frame_id)
                features.stream_sync_offset_ms += (time.perf_counter() - capture_start) * 1000.0
                if self.data_queue.full():
                    try:
                        self.data_queue.get_nowait()
                    except Empty:
                        pass
                self.data_queue.put(features)
                frame_id += 1
        except Exception as exc:
            self.error = str(exc)
            logger.exception("Capture thread failed: %s", exc)
            self.stop_event.set()
        finally:
            try:
                camera.release()
            except Exception:
                pass


class InferenceThread(threading.Thread):
    def __init__(self, data_queue: Queue, result_queue: Queue, stop_event: threading.Event, model_path: Optional[str] = None):
        super().__init__(daemon=True)
        self.data_queue = data_queue
        self.result_queue = result_queue
        self.stop_event = stop_event
        self.backend = TorchInferenceBackend(model_path=model_path)
        self.calibration = SubjectBaselineCalibration(fps=30, duration_seconds=10)
        self.scorer = AttentionScorer(window_size=WINDOW_SIZE)
        self.telemetry = Telemetry()
        self.gaze_buffer: Deque[np.ndarray] = deque(maxlen=WINDOW_SIZE)
        self.blink_buffer: Deque[np.ndarray] = deque(maxlen=WINDOW_SIZE)
        self.au_buffer: Deque[np.ndarray] = deque(maxlen=WINDOW_SIZE)
        self.gaze_valid_buffer: Deque[bool] = deque(maxlen=WINDOW_SIZE)
        self.au_valid_buffer: Deque[bool] = deque(maxlen=WINDOW_SIZE)

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                packet: FrameFeatures = self.data_queue.get(timeout=0.2)
            except Empty:
                continue
            result = self.process(packet)
            if self.result_queue.full():
                try:
                    self.result_queue.get_nowait()
                except Empty:
                    pass
            self.result_queue.put(result)
            self.data_queue.task_done()

    def process(self, packet: FrameFeatures) -> Dict[str, Any]:
        started = time.perf_counter()
        self.calibration.update(packet.gaze, packet.blink, packet.au)
        gaze, blink, au = self.calibration.transform(packet.gaze, packet.blink, packet.au)
        self.gaze_buffer.append(gaze)
        self.blink_buffer.append(blink)
        self.au_buffer.append(au)
        self.gaze_valid_buffer.append(packet.gaze_valid)
        self.au_valid_buffer.append(packet.au_valid)

        attention_score = self.scorer.update(gaze, valid=packet.gaze_valid)
        model_result = {
            "state": UNCERTAIN_CLASS,
            "confidence": 0.0,
            "probabilities": np.zeros(len(CLASS_NAMES), dtype=np.float32),
            "is_uncertain": True,
            "fusion_weights": np.array([0.5, 0.5], dtype=np.float32),
        }
        if len(self.gaze_buffer) >= WINDOW_SIZE and self.calibration.calibrated:
            model_result = self.backend.predict(
                np.asarray(self.gaze_buffer, dtype=np.float32),
                np.asarray(self.blink_buffer, dtype=np.float32),
                np.asarray(self.au_buffer, dtype=np.float32),
                np.asarray(self.gaze_valid_buffer, dtype=bool),
                np.asarray(self.au_valid_buffer, dtype=bool),
            )

        interpreted = self.interpret(attention_score, blink, packet.face_detected, model_result)
        inference_ms = (time.perf_counter() - started) * 1000.0
        self.telemetry.update(inference_ms, packet.stream_sync_offset_ms)
        health = self.telemetry.report()
        logger.info(
            "frame=%s state=%s confidence=%.3f attention=%.3f inference_ms=%.2f cpu=%.1f sync_ms=%.2f",
            packet.frame_id,
            interpreted["state"],
            interpreted["confidence"],
            attention_score,
            inference_ms,
            health["cpu_percent"],
            packet.stream_sync_offset_ms,
        )
        return {
            "frame_id": packet.frame_id,
            "timestamp": packet.timestamp,
            "frame": packet.frame,
            "face_detected": packet.face_detected,
            "attention_score": attention_score,
            "saccade": self.scorer.last_saccade,
            "state": interpreted["state"],
            "interpretation": interpreted["interpretation"],
            "confidence": interpreted["confidence"],
            "probabilities": model_result["probabilities"],
            "fusion_weights": model_result["fusion_weights"],
            "calibration_progress": self.calibration.progress,
            "calibrated": self.calibration.calibrated,
            "health": health,
        }

    def interpret(self, score: float, blink: np.ndarray, face_detected: bool, model_result: Dict[str, Any]) -> Dict[str, Any]:
        if not face_detected:
            return {"state": UNCERTAIN_CLASS, "confidence": 0.0, "interpretation": "Face lost. Improve lighting or camera angle."}
        blink_level = float(np.asarray(blink).reshape(-1)[0])
        if blink_level > 2.0 and score < 0.45:
            return {"state": "Drowsy", "confidence": 0.75, "interpretation": "Long eye closures and unstable gaze suggest fatigue."}
        if not model_result["is_uncertain"] and model_result["confidence"] >= 0.6:
            return {
                "state": model_result["state"],
                "confidence": float(model_result["confidence"]),
                "interpretation": self.describe(model_result["state"], score),
            }
        if score >= 0.68:
            return {"state": "Focused", "confidence": min(0.95, 0.62 + score * 0.3), "interpretation": "Gaze is persistent and stable on the point of regard."}
        if score >= 0.38:
            return {"state": "Mind-Wandering", "confidence": 0.62, "interpretation": "Attention is fluctuating with moderate gaze drift."}
        return {"state": UNCERTAIN_CLASS, "confidence": max(0.0, score), "interpretation": "Signals are weak or inconsistent. Holding classification."}

    @staticmethod
    def describe(state: str, score: float) -> str:
        if state == "Focused":
            return "Stable gaze and facial cues support focused attention."
        if state == "Mind-Wandering":
            return "Gaze persistence dropped and facial cues suggest attention drift."
        if state == "Drowsy":
            return "Blink and facial dynamics suggest reduced alertness."
        return "Model confidence is below threshold."


class Dashboard:
    def __init__(self, result_queue: Queue, stop_event: threading.Event):
        self.result_queue = result_queue
        self.stop_event = stop_event
        self.latest: Optional[Dict[str, Any]] = None

    def run(self) -> None:
        cv2.namedWindow("GazeApp Attention Dashboard", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("GazeApp Attention Dashboard", 1280, 720)
        while not self.stop_event.is_set():
            try:
                while True:
                    self.latest = self.result_queue.get_nowait()
                    self.result_queue.task_done()
            except Empty:
                pass
            canvas = self.render(self.latest)
            cv2.imshow("GazeApp Attention Dashboard", canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                self.stop_event.set()
                break
        cv2.destroyAllWindows()

    def render(self, result: Optional[Dict[str, Any]]) -> np.ndarray:
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        canvas[:] = (24, 26, 30)
        if result is None:
            self.put(canvas, "Starting camera and calibration...", (40, 80), 0.9)
            self.put(canvas, "Press q or Esc to stop.", (40, 125), 0.65, (190, 190, 190))
            return canvas

        frame = cv2.resize(result["frame"], (820, 615))
        canvas[30:645, 30:850] = frame
        panel_x = 890
        state = result["state"]
        color = self.state_color(state)
        self.put(canvas, state, (panel_x, 70), 1.25, color, 2)
        self.put(canvas, f"Confidence {result['confidence']:.2f}", (panel_x, 115), 0.65)
        self.put(canvas, result["interpretation"], (panel_x, 155), 0.52, (215, 215, 215))
        self.bar(canvas, "Attention", result["attention_score"], (panel_x, 210), color)
        self.bar(canvas, "Calibration", result["calibration_progress"], (panel_x, 270), (120, 190, 255))
        gaze_w, au_w = result["fusion_weights"]
        self.bar(canvas, "Gaze weight", float(gaze_w), (panel_x, 330), (120, 220, 170))
        self.bar(canvas, "AU weight", float(au_w), (panel_x, 390), (170, 170, 240))

        health = result["health"]
        y = 475
        lines = [
            f"Face: {'yes' if result['face_detected'] else 'no'}",
            f"Saccadic suppression: {'on' if result['saccade'] else 'off'}",
            f"Inference: {health['avg_inference_ms']:.1f} ms avg",
            f"CPU: {health['cpu_percent']:.1f}%",
            f"FPS: {health['fps']:.1f}",
            f"Sync offset: {health['avg_sync_offset_ms']:.1f} ms",
        ]
        for line in lines:
            self.put(canvas, line, (panel_x, y), 0.55, (220, 220, 220))
            y += 32
        self.put(canvas, "q / Esc exits", (30, 690), 0.55, (180, 180, 180))
        return canvas

    @staticmethod
    def put(img: np.ndarray, text: str, org: Tuple[int, int], scale: float, color: Tuple[int, int, int] = (245, 245, 245), thickness: int = 1) -> None:
        max_chars = 46
        lines = [text[i : i + max_chars] for i in range(0, len(text), max_chars)] or [""]
        for i, line in enumerate(lines[:3]):
            cv2.putText(img, line, (org[0], org[1] + i * int(28 * scale + 10)), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

    def bar(self, img: np.ndarray, label: str, value: float, org: Tuple[int, int], color: Tuple[int, int, int]) -> None:
        value = float(np.clip(value, 0.0, 1.0))
        self.put(img, f"{label} {value:.2f}", org, 0.52, (225, 225, 225))
        x, y = org[0], org[1] + 18
        cv2.rectangle(img, (x, y), (x + 320, y + 20), (65, 68, 74), 1)
        cv2.rectangle(img, (x + 2, y + 2), (x + int(316 * value), y + 18), color, -1)

    @staticmethod
    def state_color(state: str) -> Tuple[int, int, int]:
        if state == "Focused":
            return (105, 220, 135)
        if state == "Mind-Wandering":
            return (80, 190, 255)
        if state == "Drowsy":
            return (80, 120, 245)
        return (180, 180, 180)


class AttentionMonitor:
    def __init__(self, model_path: Optional[str] = None, camera_index: int = 0):
        self.stop_event = threading.Event()
        self.data_queue: Queue = Queue(maxsize=4)
        self.result_queue: Queue = Queue(maxsize=4)
        self.capture_thread = CameraCaptureThread(self.data_queue, self.stop_event, camera_index)
        self.inference_thread = InferenceThread(self.data_queue, self.result_queue, self.stop_event, model_path)
        self.dashboard = Dashboard(self.result_queue, self.stop_event)

    def start(self) -> None:
        logger.info("Starting GazeApp dashboard.")
        self.capture_thread.start()
        self.inference_thread.start()
        self.dashboard.run()

    def stop(self) -> None:
        self.stop_event.set()
        self.capture_thread.join(timeout=2)
        self.inference_thread.join(timeout=2)
        logger.info("GazeApp stopped.")

    def get_status(self) -> Dict[str, Any]:
        return {
            "running": not self.stop_event.is_set(),
            "data_queue_size": self.data_queue.qsize(),
            "result_queue_size": self.result_queue.qsize(),
            "threads_alive": {
                "capture": self.capture_thread.is_alive(),
                "inference": self.inference_thread.is_alive(),
            },
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime GazeApp attention dashboard.")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index.")
    parser.add_argument("--model", type=str, default=None, help="Optional trained PyTorch state_dict path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    monitor = AttentionMonitor(model_path=args.model, camera_index=args.camera)
    try:
        monitor.start()
    finally:
        monitor.stop()


if __name__ == "__main__":
    main()
