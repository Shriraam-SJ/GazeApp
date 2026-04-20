#!/usr/bin/env python3
"""Production-path smoke test that does not require a physical camera."""

import numpy as np

from main import InferenceThread, SessionTelemetry, format_session_report
from models.temporal_model import TorchInferenceBackend


def test_backend_predict():
    backend = TorchInferenceBackend()
    gaze = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), (150, 1))
    blink = np.zeros((150, 1), dtype=np.float32)
    au = np.zeros((150, 17), dtype=np.float32)
    result = backend.predict(gaze, blink, au)
    print(f"Backend state: {result['state']}, confidence: {result['confidence']:.3f}")
    assert "state" in result
    assert 0.0 <= result["confidence"] <= 1.0


def test_interpreter_rules():
    inferred = InferenceThread.__new__(InferenceThread)
    focused = InferenceThread.interpret(
        inferred,
        score=0.8,
        blink=np.array([0.0], dtype=np.float32),
        face_detected=True,
        model_result={"state": "Data Uncertain", "confidence": 0.4, "is_uncertain": True},
    )
    print(f"Interpreter state: {focused['state']}")
    assert focused["state"] == "Focused"


def test_session_report():
    session = SessionTelemetry()
    session.started_at_wall = 100.0
    session.update(100.0, {"state": "Focused", "inferred": True})
    session.update(101.0, {"state": "Distracted", "inferred": True, "outside_ratio": 0.75, "mean_por_radius": 0.22})
    session.update(102.0, {"state": "Focused", "inferred": True})
    session.stopped_at_wall = 103.0
    report = session.summary()
    text = format_session_report(report)
    print(text)
    assert report["Focused_Duration"] == 1.0
    assert report["Distracted_Duration"] == 1.0
    assert report["distraction_spikes"]


if __name__ == "__main__":
    test_backend_predict()
    test_interpreter_rules()
    test_session_report()
    print("Production smoke tests passed")
