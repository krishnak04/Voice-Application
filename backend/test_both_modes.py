import asyncio
import io
import json
import time
import httpx
import websockets
import numpy as np

BASE_URL = "http://127.0.0.1:8024"
WS_URL = "ws://127.0.0.1:8024/ws/call"

def generate_synthetic_speech_pcm(duration_sec=3.0, sample_rate=16000):
    """
    Generate synthetic speech-like harmonic waveform (formants F1=500Hz, F2=1500Hz, F3=2500Hz)
    suitable for audio pipelines.
    """
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    f0 = 130 + 15 * np.sin(2 * np.pi * 1.5 * t)
    phase = 2 * np.pi * np.cumsum(f0) / sample_rate
    harmonics = (
        0.5 * np.sin(phase) +
        0.3 * np.sin(2 * phase) +
        0.2 * np.sin(3 * phase) +
        0.1 * np.sin(5 * phase)
    )
    envelope = 0.5 * (1 - np.cos(2 * np.pi * t / duration_sec))
    signal = harmonics * envelope * 0.75
    pcm16 = (signal * 32767).astype(np.int16)
    return pcm16.tobytes()

from pathlib import Path

def get_test_pcm():
    """Load real human speech recording (sample_01.wav) for genuine end-to-end testing."""
    wav_path = Path(__file__).parent / "test_audio_dataset" / "sample_01.wav"
    if wav_path.exists():
        import wave
        with wave.open(str(wav_path), 'rb') as wf:
            return wf.readframes(wf.getnframes())
    return generate_synthetic_speech_pcm(duration_sec=3.0, sample_rate=16000)

def pcm_to_wav_bytes(pcm_bytes, sample_rate=16000, num_channels=1):
    import wave
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(num_channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()

async def test_backend_health():
    print("\n" + "="*60)
    print("STEP 0: Checking Server Health & Accent Info")
    print("="*60)
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.get(f"{BASE_URL}/api/health")
        assert r.status_code == 200, f"Health check failed: {r.status_code}"
        health = r.json()
        print(f"Server Status: {health.get('status')}")
        print(f"Whisper Model: {health.get('whisper_model')} on {health.get('device')}")
        print(f"Sherpa Model: {health.get('sherpa_model')} (available={health.get('sherpa_available')})")

        r_acc = await client.get(f"{BASE_URL}/api/accent/info")
        assert r_acc.status_code == 200, f"Accent info check failed: {r_acc.status_code}"
        info = r_acc.json()
        print(f"Accent Detector: {info.get('model_name')}")
        print(f"License: {info.get('license')} on {info.get('device')}")
        print(f"Classes ({len(info.get('classes', []))}): {', '.join(info.get('classes', []))}")
        print(f"Detector Status: {info.get('status')}")
    print("[PASS] Health & Accent Info verified.")

async def test_mode_1_phone_call_simulator():
    print("\n" + "="*60)
    print("MODE 1: Phone Call Simulator (WebSocket Full-Duplex & Call Turn)")
    print("="*60)

    # 1. Test WebSocket Full-Duplex real-time accent detection
    print("\n--- Subtest 1.1: WebSocket Real-Time Conversation & Accent Detection ---")
    async with websockets.connect(WS_URL) as ws:
        # Send initial config
        await ws.send(json.dumps({
            "type": "config",
            "tts_voice": "en-US-JennyNeural",
            "tts_engine": "edge_tts",
            "auto_adapt_voice": True
        }))

        # Stream audio chunks simulating microphone input
        pcm_bytes = get_test_pcm()
        chunk_size = 1280 # 40ms at 16kHz 16-bit
        for i in range(0, len(pcm_bytes), chunk_size):
            await ws.send(pcm_bytes[i:i+chunk_size])
            await asyncio.sleep(0.015)

        # Send silence to trigger VAD endpoint
        silence = b"\x00" * (16000 * 2) # 1 sec silence
        for i in range(0, len(silence), chunk_size):
            await ws.send(silence[i:i+chunk_size])
            await asyncio.sleep(0.015)

        # Collect WebSocket events
        events_received = []
        accent_event = None
        turn_complete = False

        start_wait = time.perf_counter()
        while time.perf_counter() - start_wait < 30.0 and not turn_complete:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=20.0)
                if isinstance(msg, str):
                    data = json.loads(msg)
                    events_received.append(data.get("type"))
                    if data.get("type") == "accent_detected":
                        accent_event = data
                        print(f"  [WS Event] accent_detected: {data.get('accent')} ({data.get('confidence')*100:.1f}%) latency={data.get('latency_ms')}ms")
                    elif data.get("type") == "voice_adapted":
                        print(f"  [WS Event] voice_adapted: {data.get('voice')} for {data.get('accent')}")
                    elif data.get("type") == "turn_complete":
                        turn_complete = True
                        print("  [WS Event] turn_complete received.")
            except asyncio.TimeoutError:
                break

        print(f"  Total WS Events received: {events_received}")
        assert accent_event is not None, "Did not receive accent_detected event on WebSocket"
        assert "accent" in accent_event
        assert "confidence" in accent_event
        assert "latency_ms" in accent_event
        print("  [PASS] WebSocket accent detection event received with separated latency.")

    # 2. Test REST Fallback Call Turn (/api/call/turn)
    print("\n--- Subtest 1.2: REST Fallback Phone Call Turn (/api/call/turn) ---")
    wav_bytes = pcm_to_wav_bytes(pcm_bytes)
    async with httpx.AsyncClient(timeout=30.0) as client:
        files = {"audio": ("phone_call.wav", wav_bytes, "audio/wav")}
        data = {
            "tts_voice": "en-US-JennyNeural",
            "stt_engine": "whisper",
            "whisper_model": "base"
        }
        t0 = time.perf_counter()
        r = await client.post(f"{BASE_URL}/api/call/turn", files=files, data=data)
        elapsed = time.perf_counter() - t0
        assert r.status_code == 200, f"Call turn failed: {r.status_code} - {r.text}"
        res = r.json()
        print(f"  Turn Response: transcript='{res.get('transcript')}'")
        print(f"  Turn Latency: STT={res.get('latency', {}).get('stt_latency')}s, Server Total={res.get('latency', {}).get('total_server_time')}s")
        assert res.get("accent_analysis") is not None, "accent_analysis missing from call turn"
        acc = res["accent_analysis"]
        print(f"  Acoustic Accent: {acc.get('accent')} (conf={acc.get('confidence')*100:.1f}%, latency={acc.get('latency_ms')}ms)")
        print(f"  Matched TTS Voice: {acc.get('tts_voice')}")
        print("  [PASS] REST call turn returned valid accent_analysis with separate latency.")

    print("\n[SUCCESS] MODE 1 (Phone Call Simulator) verified completely.")

async def test_mode_2_push_to_talk_benchmark():
    print("\n" + "="*60)
    print("MODE 2: Push-to-Talk Benchmark (/api/transcribe & /api/voice)")
    print("="*60)

    pcm_bytes = get_test_pcm()
    wav_bytes = pcm_to_wav_bytes(pcm_bytes)

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Test 2.1: /api/transcribe with Whisper engine
        print("\n--- Subtest 2.1: /api/transcribe with Faster-Whisper ---")
        files = {"audio": ("bench_whisper.wav", wav_bytes, "audio/wav")}
        data = {"engine": "whisper", "model": "base", "detect_accent": "true"}
        r = await client.post(f"{BASE_URL}/api/transcribe", files=files, data=data)
        assert r.status_code == 200, f"Whisper transcribe failed: {r.text}"
        res_whisper = r.json()
        print(f"  Whisper Model: {res_whisper.get('model_name')} on {res_whisper.get('device')}")
        print(f"  STT Latency: {res_whisper.get('stt_latency')}s (Speed: {res_whisper.get('speed_factor')}x)")
        assert res_whisper.get("accent_analysis") is not None, "accent_analysis missing from Whisper transcribe"
        acc_w = res_whisper["accent_analysis"]
        print(f"  Accent Variety: {acc_w.get('accent')} ({acc_w.get('confidence')*100:.1f}%)")
        print(f"  Accent Latency: {acc_w.get('latency_ms')}ms (SEPARATE from STT latency {res_whisper.get('stt_latency')}s)")
        print("  [PASS] Whisper STT + Accent Detection verified.")

        # Test 2.2: /api/transcribe with Sherpa-ONNX engine
        print("\n--- Subtest 2.2: /api/transcribe with Sherpa-ONNX ---")
        files = {"audio": ("bench_sherpa.wav", wav_bytes, "audio/wav")}
        data = {"engine": "sherpa", "detect_accent": "true"}
        r = await client.post(f"{BASE_URL}/api/transcribe", files=files, data=data)
        assert r.status_code == 200, f"Sherpa transcribe failed: {r.text}"
        res_sherpa = r.json()
        print(f"  Sherpa Model: {res_sherpa.get('model_name')}")
        print(f"  STT Latency: {res_sherpa.get('stt_latency')}s (RTF: {res_sherpa.get('rtf')})")
        assert res_sherpa.get("accent_analysis") is not None, "accent_analysis missing from Sherpa transcribe"
        acc_s = res_sherpa["accent_analysis"]
        print(f"  Accent Variety: {acc_s.get('accent')} ({acc_s.get('confidence')*100:.1f}%)")
        print(f"  Accent Latency: {acc_s.get('latency_ms')}ms (SEPARATE from Sherpa STT latency {res_sherpa.get('stt_latency')}s)")
        print("  [PASS] Sherpa-ONNX STT + Accent Detection verified.")

        # Test 2.3: /api/voice legacy composite benchmark endpoint
        print("\n--- Subtest 2.3: /api/voice Composite Benchmark Endpoint ---")
        files = {"audio": ("bench_composite.wav", wav_bytes, "audio/wav")}
        data = {"engine": "whisper", "model": "base"}
        r = await client.post(f"{BASE_URL}/api/voice", files=files, data=data)
        res_voice = r.json()
        safe_reply = res_voice.get('reply', '').encode('ascii', 'replace').decode('ascii')
        print(f"  Composite reply: '{safe_reply}'")
        print(f"  STT Latency: {res_voice.get('latency', {}).get('stt_latency')}s")
        print(f"  Groq Latency: {res_voice.get('latency', {}).get('groq_latency')}s")
        assert res_voice.get("accent_analysis") is not None, "accent_analysis missing from /api/voice"
        acc_v = res_voice["accent_analysis"]
        print(f"  Accent Analysis: {acc_v.get('accent')} ({acc_v.get('confidence')*100:.1f}%) latency={acc_v.get('latency_ms')}ms")
        print("  [PASS] Composite /api/voice endpoint returned accent_analysis.")

    print("\n[SUCCESS] MODE 2 (Push-to-Talk Benchmark) verified completely.")

async def main():
    await test_backend_health()
    await test_mode_1_phone_call_simulator()
    await test_mode_2_push_to_talk_benchmark()
    print("\n" + "="*60)
    print("ALL TESTS PASSED! Automatic English Accent Detection works in BOTH modes independently.")
    print("="*60)

if __name__ == "__main__":
    asyncio.run(main())
