from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Optional

import numpy as np


EPS = 1e-8


def _as_gaze_array(gaze_sequence: Iterable[np.ndarray]) -> np.ndarray:
    arr = np.asarray(list(gaze_sequence), dtype=np.float32)
    if arr.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    arr = arr.reshape(-1, arr.shape[-1])
    if arr.shape[1] < 3:
        padded = np.zeros((arr.shape[0], 3), dtype=np.float32)
        padded[:, : arr.shape[1]] = arr
        return padded
    return arr[:, :3]


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(norms, EPS)


def vectorized_angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    v1_norm = normalize_vectors(np.asarray(v1, dtype=np.float32))
    v2_norm = normalize_vectors(np.asarray(v2, dtype=np.float32))
    cos_angles = np.sum(v1_norm * v2_norm, axis=-1)
    return np.arccos(np.clip(cos_angles, -1.0, 1.0))


def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    return float(vectorized_angle_between_vectors(np.asarray(v1)[None, :], np.asarray(v2)[None, :])[0])


def detect_saccades(
    gaze_sequence: Iterable[np.ndarray],
    angular_velocity_threshold: float = 0.12,
    suppression_frames: int = 2,
) -> np.ndarray:
    """Return a mask where rapid gaze movements should be ignored."""

    gaze = _as_gaze_array(gaze_sequence)
    if len(gaze) < 2:
        return np.zeros(len(gaze), dtype=bool)

    angular_step = vectorized_angle_between_vectors(gaze[:-1], gaze[1:])
    mask = np.zeros(len(gaze), dtype=bool)
    rapid = angular_step > angular_velocity_threshold
    rapid_idx = np.flatnonzero(rapid)
    for idx in rapid_idx:
        start = max(0, idx - suppression_frames)
        end = min(len(mask), idx + suppression_frames + 2)
        mask[start:end] = True
    return mask


def leaky_integrator(values: np.ndarray, leak_rate: float = 0.16, initial_value: float = 0.0) -> np.ndarray:
    """Smooth values with a one-pole leaky integrator."""

    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values.astype(np.float32)
    leak_rate = float(np.clip(leak_rate, 0.0, 1.0))
    out = np.empty_like(values, dtype=np.float32)
    state = float(initial_value)
    for i, value in enumerate(values):
        state = ((1.0 - leak_rate) * state) + (leak_rate * float(value))
        out[i] = state
    return out


def compute_attention_score(
    gaze_sequence: Iterable[np.ndarray],
    window_size: int = 150,
    persistence_threshold: float = 0.72,
    max_angular_dispersion: float = 0.45,
    angular_velocity_threshold: float = 0.12,
    leak_rate: float = 0.16,
    previous_score: float = 0.0,
) -> float:
    """PoR Persistence 2.0 with saccadic suppression and smooth transitions."""

    gaze = _as_gaze_array(gaze_sequence)
    if len(gaze) < 3:
        return float(previous_score)

    gaze = gaze[-window_size:]
    saccade_mask = detect_saccades(gaze, angular_velocity_threshold=angular_velocity_threshold)
    valid = ~saccade_mask
    valid_ratio = float(valid.mean()) if valid.size else 0.0
    if valid.sum() < 3:
        return float(previous_score * (1.0 - leak_rate))

    valid_gaze = gaze[valid]
    angles = vectorized_angle_between_vectors(valid_gaze[:-1], valid_gaze[1:])
    dispersion = float(np.nanmean(angles)) if angles.size else 0.0
    stability = float(np.clip(1.0 - (dispersion / max_angular_dispersion), 0.0, 1.0))

    centered = valid_gaze - np.nanmean(valid_gaze, axis=0, keepdims=True)
    por_spread = float(np.nanmean(np.linalg.norm(centered, axis=1)))
    por_persistence = float(np.clip(1.0 - (por_spread / 0.35), 0.0, 1.0))
    blink_quality = valid_ratio
    raw = (0.52 * stability) + (0.36 * por_persistence) + (0.12 * blink_quality)
    if raw < persistence_threshold:
        raw *= 0.72
    score = leaky_integrator(np.array([raw], dtype=np.float32), leak_rate=leak_rate, initial_value=previous_score)[-1]
    return float(np.clip(score, 0.0, 1.0))


class AttentionScorer:
    """Realtime vectorized scorer with bounded history."""

    def __init__(
        self,
        window_size: int = 150,
        persistence_threshold: float = 0.72,
        angular_velocity_threshold: float = 0.12,
        leak_rate: float = 0.16,
        history_size: int = 1000,
    ):
        self.window_size = window_size
        self.persistence_threshold = persistence_threshold
        self.angular_velocity_threshold = angular_velocity_threshold
        self.leak_rate = leak_rate
        self.gaze_history: Deque[np.ndarray] = deque(maxlen=history_size)
        self.score_history: Deque[float] = deque(maxlen=history_size)
        self.last_score = 0.0
        self.last_saccade = False

    def update(self, new_gaze: np.ndarray, valid: bool = True) -> float:
        gaze = np.asarray(new_gaze if valid else [np.nan, np.nan, np.nan], dtype=np.float32)
        if not np.isfinite(gaze).all():
            self.last_score = float(self.last_score * (1.0 - self.leak_rate))
            self.score_history.append(self.last_score)
            return self.last_score

        self.gaze_history.append(gaze)
        recent = list(self.gaze_history)[-self.window_size :]
        saccades = detect_saccades(recent, angular_velocity_threshold=self.angular_velocity_threshold)
        self.last_saccade = bool(saccades[-1]) if len(saccades) else False
        self.last_score = compute_attention_score(
            recent,
            window_size=min(self.window_size, len(recent)),
            persistence_threshold=self.persistence_threshold,
            angular_velocity_threshold=self.angular_velocity_threshold,
            leak_rate=self.leak_rate,
            previous_score=self.last_score,
        )
        self.score_history.append(self.last_score)
        return self.last_score

    def get_trend(self, window: int = 50) -> Dict[str, float]:
        recent = np.asarray(list(self.score_history)[-window:], dtype=np.float32)
        if len(recent) < 3:
            return {"trend": 0.0, "volatility": 0.0, "current": self.last_score, "mean_recent": self.last_score}
        x = np.arange(len(recent), dtype=np.float32)
        slope = float(np.polyfit(x, recent, 1)[0])
        return {
            "trend": slope,
            "volatility": float(np.std(recent)),
            "current": float(self.last_score),
            "mean_recent": float(np.mean(recent)),
        }

    def reset(self) -> None:
        self.gaze_history.clear()
        self.score_history.clear()
        self.last_score = 0.0
        self.last_saccade = False


@dataclass
class TemporalGazeSample:
    timestamp: float
    gaze: np.ndarray
    valid: bool


class TemporalStateInference:
    """Two-second PoR state inference with weighted smoothing and jitter tolerance."""

    def __init__(
        self,
        window_seconds: float = 2.0,
        inference_interval_seconds: float = 2.0,
        outside_threshold: float = 0.60,
        target_radius: float = 0.18,
        max_history_seconds: float = 4.0,
    ) -> None:
        self.window_seconds = float(window_seconds)
        self.inference_interval_seconds = float(inference_interval_seconds)
        self.outside_threshold = float(outside_threshold)
        self.target_radius = float(target_radius)
        self.samples: Deque[TemporalGazeSample] = deque()
        self.max_history_seconds = float(max_history_seconds)
        self.last_inference_at: Optional[float] = None
        self.last_result: Dict[str, float | str | bool] = self._empty_result("Collecting")

    def update(self, timestamp: float, gaze: np.ndarray, valid: bool = True) -> Dict[str, float | str | bool]:
        timestamp = float(timestamp)
        self.samples.append(TemporalGazeSample(timestamp, np.asarray(gaze, dtype=np.float32), bool(valid)))
        self._trim(timestamp, self.max_history_seconds)

        if self.last_inference_at is None:
            self.last_inference_at = timestamp
            return self.last_result

        if timestamp - self.last_inference_at < self.inference_interval_seconds:
            return self.last_result

        self.last_inference_at = timestamp
        self.last_result = self.infer(timestamp)
        return self.last_result

    def infer(self, now: Optional[float] = None) -> Dict[str, float | str | bool]:
        if now is None:
            now = self.samples[-1].timestamp if self.samples else 0.0
        window = self._window(float(now))
        valid = [sample for sample in window if sample.valid and np.isfinite(sample.gaze).all()]
        duration = self._duration(window)
        if len(valid) < 3 or duration < self.window_seconds * 0.5:
            return self._empty_result("Data Uncertain", duration)

        gaze = normalize_vectors(_as_gaze_array(sample.gaze for sample in valid))
        smoothed = weighted_moving_average(gaze)
        por_radius = np.linalg.norm(smoothed[:, :2], axis=1)
        outside_ratio = float(np.mean(por_radius > self.target_radius))
        mean_radius = float(np.mean(por_radius))
        state = "Distracted" if outside_ratio > self.outside_threshold else "Focused"
        confidence = float(np.clip(abs(outside_ratio - self.outside_threshold) / max(self.outside_threshold, EPS), 0.35, 0.98))
        return {
            "state": state,
            "outside_ratio": outside_ratio,
            "mean_por_radius": mean_radius,
            "window_duration": duration,
            "sample_count": float(len(valid)),
            "confidence": confidence,
            "inferred": True,
        }

    def reset(self) -> None:
        self.samples.clear()
        self.last_inference_at = None
        self.last_result = self._empty_result("Collecting")

    def _trim(self, now: float, history_seconds: float) -> None:
        cutoff = now - history_seconds
        while self.samples and self.samples[0].timestamp < cutoff:
            self.samples.popleft()

    def _window(self, now: float) -> List[TemporalGazeSample]:
        cutoff = now - self.window_seconds
        return [sample for sample in self.samples if sample.timestamp >= cutoff]

    @staticmethod
    def _duration(samples: List[TemporalGazeSample]) -> float:
        if len(samples) < 2:
            return 0.0
        return float(max(samples[-1].timestamp - samples[0].timestamp, 0.0))

    @staticmethod
    def _empty_result(state: str, duration: float = 0.0) -> Dict[str, float | str | bool]:
        return {
            "state": state,
            "outside_ratio": 0.0,
            "mean_por_radius": 0.0,
            "window_duration": duration,
            "sample_count": 0.0,
            "confidence": 0.0,
            "inferred": False,
        }


def weighted_moving_average(values: np.ndarray, span: int = 5) -> np.ndarray:
    """Apply a short trailing weighted moving average to suppress high-frequency jitter."""

    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    span = int(max(1, span))
    smoothed = np.empty_like(values, dtype=np.float32)
    for idx in range(len(values)):
        start = max(0, idx - span + 1)
        chunk = values[start : idx + 1]
        weights = np.arange(1, len(chunk) + 1, dtype=np.float32)
        weights /= weights.sum()
        smoothed[idx] = np.sum(chunk * weights[:, None], axis=0)
    return smoothed


if __name__ == "__main__":
    stable = [np.array([0.0, 0.0, 1.0], dtype=np.float32) for _ in range(150)]
    noisy = [np.random.randn(3).astype(np.float32) for _ in range(150)]
    print(f"Stable: {compute_attention_score(stable):.3f}")
    print(f"Noisy: {compute_attention_score(noisy):.3f}")
