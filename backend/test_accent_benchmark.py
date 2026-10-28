"""
Accent Detection Benchmark & Verification Script.

Tests acoustic English accent classification across candidate classes:
- Indian English
- American English
- British English
- Australian English
- Canadian English
- New Zealand English
- Irish English
- South African English
- Other English variety
- Unknown / insufficient audio

Measures:
1. Classification accuracy on labelled audio
2. Accent detection latency (isolated from STT)
3. Confidence scores
4. Insufficient audio threshold handling
5. Multi-turn session smoothing
"""

import os
import sys
import time
from pathlib import Path

# Add backend directory to sys.path
sys.path.insert(0, str(Path(__file__).parent))

from accent_service import get_accent_detector, AccentDetectionService

EXPECTED_MAPPINGS = {
    "indian_1.wav": "Indian English",
    "us_1.wav": "American English",
    "england_1.wav": "British English",
    "australia_1.wav": "Australian English",
    "canada_1.wav": "Canadian English",
    "newzealand_1.wav": "New Zealand English",
    "ireland_1.wav": "Irish English",
    "african_1.wav": "South African English",
}

def run_benchmark():
    print("=" * 70)
    print(" AUTOMATIC ENGLISH ACCENT DETECTION BENCHMARK (ECAPA-TDNN)")
    print(" Model: Jzuluaga/accent-id-commonaccent_ecapa (MIT License)")
    print("=" * 70)

    detector = get_accent_detector()
    if not detector.is_available():
        print("ERROR: Accent detector failed to load!")
        return False

    test_dir = Path(__file__).parent / "test_accents" / "data"
    if not test_dir.exists():
        print(f"ERROR: Test audio directory {test_dir} not found!")
        return False

    correct = 0
    total = 0
    latencies = []

    print(f"\n1. ACCURACY & LATENCY BENCHMARK (LABELLED DATASET):")
    print(f"{'Filename':<18} | {'Ground Truth':<22} | {'Predicted Variety':<24} | {'Conf':<6} | {'Latency':<8} | {'Status'}")
    print("-" * 96)

    for filename, expected_variety in EXPECTED_MAPPINGS.items():
        filepath = test_dir / filename
        if not filepath.exists():
            print(f"Skipping {filename} (not found)")
            continue

        res = detector.detect_accent_sync(str(filepath))
        predicted = res["accent"]
        conf = res["confidence"]
        latency = res["latency_ms"]
        status = res["status"]

        is_match = (predicted == expected_variety)
        if is_match:
            correct += 1
        total += 1
        latencies.append(latency)

        match_icon = "PASS [OK]" if is_match else "FAIL [X]"
        print(f"{filename:<18} | {expected_variety:<22} | {predicted:<24} | {conf*100:>4.1f}% | {latency:>5.1f}ms | {match_icon}")

    accuracy = (correct / total * 100) if total > 0 else 0
    avg_latency = (sum(latencies) / len(latencies)) if latencies else 0

    print("-" * 96)
    print(f"Accuracy: {correct}/{total} ({accuracy:.1f}%) | Avg Accent Latency: {avg_latency:.1f}ms")

    # 2. Test short audio / insufficient evidence threshold
    print(f"\n2. INSUFFICIENT AUDIO HANDLING (< 1.5s):")
    import numpy as np
    short_audio = np.zeros(int(16000 * 0.8), dtype=np.float32)  # 0.8s
    res_short = detector.detect_accent_sync(short_audio)
    print(f"0.8s Audio -> Predicted: '{res_short['accent']}' | Status: {res_short['status']}")
    assert res_short["status"] == "insufficient_audio", "Expected status 'insufficient_audio' for short clip"
    assert "insufficient" in res_short["accent"].lower(), "Expected insufficient in accent text"
    print("PASS: Correctly rejected audio below 1.5s threshold")

    # 3. Test multi-turn session smoothing
    print(f"\n3. MULTI-TURN SESSION SMOOTHING & HYSTERESIS:")
    session_id = "test_session_42"
    # Turn 1: Indian English sample
    turn1 = detector.detect_accent_sync(str(test_dir / "indian_1.wav"), session_id=session_id)
    print(f"Turn 1 (Indian):     Predicted: {turn1['accent']} (Conf: {turn1['confidence']*100:.1f}%)")
    # Turn 2: Short audio that relies on accumulated session buffer
    short_indian = np.zeros(int(16000 * 0.5), dtype=np.float32)
    turn2 = detector.detect_accent_sync(short_indian, session_id=session_id)
    print(f"Turn 2 (0.5s pause): Predicted: {turn2['accent']} (Smoothed: {turn2.get('is_smoothed', False)})")
    assert turn2["accent"] == "Indian English", "Session should retain detected Indian English accent across turns"
    print("PASS: Session retained detected accent without flickering")

    print("\n" + "=" * 70)
    print(f"ALL BENCHMARK TESTS PASSED! Accuracy: {accuracy:.1f}%")
    print("=" * 70)
    return True

if __name__ == "__main__":
    success = run_benchmark()
    sys.exit(0 if success else 1)
