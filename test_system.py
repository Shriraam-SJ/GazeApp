#!/usr/bin/env python3
"""Smoke tests for the realtime attention system."""

import numpy as np
import torch

from models.temporal_model import BlinkFeatureExtractor, MultiModalAttentionDetector, MultiModalTransformer
from src.attention_score import AttentionScorer, compute_attention_score


def test_attention_score():
    stable_gaze = [np.array([0.0, 0.0, 1.0], dtype=np.float32) for _ in range(200)]
    noisy_gaze = [np.random.randn(3).astype(np.float32) for _ in range(200)]
    stable_score = compute_attention_score(stable_gaze)
    noisy_score = compute_attention_score(noisy_gaze)
    print(f"Stable score: {stable_score:.3f}")
    print(f"Noisy score: {noisy_score:.3f}")
    assert 0.0 <= stable_score <= 1.0
    assert 0.0 <= noisy_score <= 1.0
    assert stable_score > noisy_score


def test_realtime_scorer():
    scorer = AttentionScorer(window_size=30)
    for _ in range(45):
        score = scorer.update(np.array([0.0, 0.0, 1.0], dtype=np.float32))
    trend = scorer.get_trend(window=20)
    print(f"Realtime score: {score:.3f}, trend: {trend['trend']:.4f}")
    assert score > 0.5


def test_blink_extractor():
    extractor = BlinkFeatureExtractor()
    blink_input = torch.randn(2, 150, 1)
    features = extractor(blink_input)
    print(f"Blink feature shape: {tuple(features.shape)}")
    assert features.shape == (2, 64)


def test_model_forward():
    gaze = torch.randn(2, 150, 3)
    blink = torch.randn(2, 150, 1)
    au = torch.randn(2, 150, 17)
    model = MultiModalTransformer()
    output = model(gaze, blink, au)
    print(f"MMT logits shape: {tuple(output['logits'].shape)}")
    assert output["logits"].shape == (2, 3)
    assert output["probabilities"].shape == (2, 3)
    assert output["predictions"].shape == (2,)

    compat = MultiModalAttentionDetector(au_dim=17)
    logits = compat(gaze, blink, au)
    assert logits.shape == (2, 3)


if __name__ == "__main__":
    print("Running GazeApp smoke tests")
    test_attention_score()
    test_realtime_scorer()
    test_blink_extractor()
    test_model_forward()
    print("All smoke tests passed")
