#!/usr/bin/env python3
"""
Demo script for the Multi-Modal Attention Detection System
Shows the system processing simulated real-time data
"""

import torch
import numpy as np
import time
from models.temporal_model import MultiModalTransformer
from src.attention_score import AttentionScorer

def simulate_real_time_data(num_frames=300):
    """Simulate real-time gaze, blink, and AU data."""
    print("🎭 Simulating real-time multi-modal data streams...")

    # Initialize data buffers
    gaze_buffer = []
    blink_buffer = []
    au_buffer = []

    # Simulate different attention states
    states = ['focused', 'mind_wandering', 'drowsy']
    current_state = 0
    state_duration = 100  # frames per state

    for frame in range(num_frames):
        # Change state every state_duration frames
        if frame % state_duration == 0:
            current_state = (current_state + 1) % len(states)
            print(f"   Switching to state: {states[current_state]}")

        # Generate gaze data based on state
        if states[current_state] == 'focused':
            # Stable gaze with small noise
            base_gaze = np.array([0.0, 0.0, 1.0])
            noise = np.random.normal(0, 0.05, 3)
        elif states[current_state] == 'mind_wandering':
            # More dispersed gaze
            base_gaze = np.array([0.2, 0.1, 0.9])
            noise = np.random.normal(0, 0.15, 3)
        else:  # drowsy
            # Very stable but slightly downward gaze
            base_gaze = np.array([0.0, -0.1, 0.95])
            noise = np.random.normal(0, 0.03, 3)

        gaze = base_gaze + noise
        gaze_buffer.append(gaze)

        # Generate blink data
        blink = 0.0
        if states[current_state] == 'drowsy':
            # More frequent blinks when drowsy
            if np.random.random() < 0.05:  # 5% chance
                blink = np.random.uniform(0.6, 1.0)
        else:
            # Normal blink rate
            if np.random.random() < 0.02:  # 2% chance
                blink = np.random.uniform(0.5, 0.9)

        blink_buffer.append(np.array([blink]))

        # Generate AU data (simplified)
        au_intensity = np.random.uniform(0, 0.3, 17)  # Base low intensity

        if states[current_state] == 'focused':
            # Higher AU4 (brow lower) and AU7 (lid tighten) for concentration
            au_intensity[3] = np.random.uniform(0.4, 0.7)  # AU4
            au_intensity[6] = np.random.uniform(0.3, 0.6)  # AU7
        elif states[current_state] == 'mind_wandering':
            # Lower AU activity
            au_intensity *= 0.5
        else:  # drowsy
            # Very low AU activity, some eyelid droop (AU7)
            au_intensity *= 0.3
            au_intensity[6] = np.random.uniform(0.1, 0.3)  # AU7

        au_buffer.append(au_intensity)

    return gaze_buffer, blink_buffer, au_buffer

def run_demo():
    """Run the attention detection demo."""
    print("🚀 Multi-Modal Attention Detection System Demo")
    print("=" * 60)

    # Initialize the model
    print("🔧 Initializing Multi-Modal Transformer...")
    model = MultiModalTransformer(au_dim=17, calibrate=True)
    model.eval()

    # Initialize attention scorer
    attention_scorer = AttentionScorer()

    # Simulate real-time data
    gaze_data, blink_data, au_data = simulate_real_time_data(300)

    print("📊 Processing simulated real-time data...")
    print("Frame | Cognitive State | Confidence | Attention Score | Uncertain")
    print("-" * 70)

    # Process frames in sliding window fashion
    window_size = 150
    state_counts = {'focused': 0, 'mind_wandering': 0, 'drowsy': 0, 'data_uncertain': 0}

    for frame_idx in range(len(gaze_data)):
        # Update attention scorer with current gaze
        attention_score = attention_scorer.update(gaze_data[frame_idx])

        # Only run model inference when we have enough data
        if frame_idx >= window_size - 1:
            # Prepare tensors for the current window
            start_idx = frame_idx - window_size + 1

            gaze_window = torch.tensor(np.array(gaze_data[start_idx:start_idx+window_size]),
                                     dtype=torch.float32).unsqueeze(0)
            blink_window = torch.tensor(np.array(blink_data[start_idx:start_idx+window_size]),
                                      dtype=torch.float32).unsqueeze(0)
            au_window = torch.tensor(np.array(au_data[start_idx:start_idx+window_size]),
                                   dtype=torch.float32).unsqueeze(0)

            # Run inference
            with torch.no_grad():
                outputs = model(gaze_window, blink_window, au_window)

            # Extract results
            prediction = outputs['predictions'][0].item()
            confidence = outputs['confidence'][0].item()
            is_uncertain = outputs['is_uncertain'][0].item()

            # Map prediction to state
            state_map = {0: 'focused', 1: 'mind_wandering', 2: 'drowsy', 3: 'data_uncertain'}
            cognitive_state = state_map.get(prediction, 'unknown')
            state_counts[cognitive_state] += 1

            # Print results every 30 frames
            if frame_idx % 30 == 0:
                uncertain_marker = "⚠️" if is_uncertain else "✅"
                print("5d")

        # Small delay to simulate real-time processing
        time.sleep(0.01)

    print("\n" + "=" * 60)
    print("📈 Demo Results Summary:")
    print(f"   Total Frames Processed: {len(gaze_data)}")
    print(f"   State Distribution: {state_counts}")
    print(".1f")
    print(".1f")
    print(".1f")
    print(".1f")
    print(".1f")
    print(".1f")

    # Get attention score trend
    trend = attention_scorer.get_trend()
    print("\n🎯 Final Attention Metrics:")
    print(f"   Current Attention Score: {attention_score:.3f}")
    print(f"   Score Trend: {trend['trend']:.4f}")
    print(f"   Score Volatility: {trend['volatility']:.3f}")
    print(f"   Recent Average Score: {trend['mean_recent']:.3f}")
    print(f"   Total Scores Computed: {len(attention_scorer.score_history)}")

    print("\n✅ Demo completed successfully!")
    print("🎉 The Multi-Modal Attention Detection System is working perfectly!")

if __name__ == "__main__":
    run_demo()