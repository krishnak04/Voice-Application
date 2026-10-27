#!/usr/bin/env python3
"""
Quick TTS verification script for dev server.
Test text-to-voice synthesis and measure VRAM / latency.

Run with:
    source "/media/i3admin/New Volume/voice-to-text/stt-base-en/venv/bin/activate"
    python test_tts.py
"""

import time
import os

print("=" * 60)
print("Testing Text-to-Voice (TTS) Engines")
print("=" * 60)

test_text = "Hello! This is a test of real-time voice calling using low VRAM speech synthesis."

# 1. Test Edge-TTS (0 VRAM)
print("\n[1/2] Testing Edge-TTS (0 MB VRAM, 0 CPU)...")
try:
    import edge_tts
    import asyncio

    async def run_edge():
        comm = edge_tts.Communicate(test_text, voice="en-US-JennyNeural")
        await comm.save("test_edge_output.mp3")

    start = time.perf_counter()
    asyncio.run(run_edge())
    elapsed = time.perf_counter() - start
    size = os.path.getsize("test_edge_output.mp3")
    print(f"  SUCCESS: Generated test_edge_output.mp3 ({size} bytes) in {elapsed:.3f}s")
except ImportError:
    print("  Edge-TTS not installed. Run: pip install edge-tts")
except Exception as e:
    print(f"  Edge-TTS error: {e}")

# 2. Test Kokoro-82M (~300MB VRAM)
print("\n[2/2] Testing Kokoro-82M (~300 MB VRAM / CPU)...")
try:
    from kokoro import KPipeline
    import soundfile as sf
    import numpy as np

    print("  Loading Kokoro-82M pipeline...")
    start_load = time.perf_counter()
    pipeline = KPipeline(lang_code='a')
    print(f"  Model loaded in {time.perf_counter() - start_load:.2f}s")

    print(f"  Synthesizing text: '{test_text}'")
    start = time.perf_counter()
    generator = pipeline(test_text, voice='af_heart', speed=1.0)
    all_audio = []
    for _, _, audio in generator:
        if audio is not None:
            if hasattr(audio, "cpu"):
                audio = audio.cpu().numpy()
            all_audio.append(audio)

    elapsed = time.perf_counter() - start
    if all_audio:
        combined = np.concatenate(all_audio)
        sf.write("test_kokoro_output.wav", combined, 24000)
        size = os.path.getsize("test_kokoro_output.wav")
        audio_dur = len(combined) / 24000
        rtf = elapsed / audio_dur
        print(f"  SUCCESS: Generated test_kokoro_output.wav ({audio_dur:.2f}s audio) in {elapsed:.3f}s (RTF: {rtf:.3f})")
    else:
        print("  Warning: No audio produced.")
except ImportError:
    print("  Kokoro not installed. Run: pip install kokoro soundfile")
except Exception as e:
    print(f"  Kokoro error: {e}")

print("\n" + "=" * 60)
print("TTS test completed!")
print("=" * 60)
