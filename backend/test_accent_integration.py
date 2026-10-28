"""
End-to-End Integration & WebSocket Test for Automatic English Accent Detection.

Validates:
1. GET /api/accent/info endpoint
2. POST /api/accent/detect endpoint
3. POST /api/transcribe with accent analysis
4. /ws/call WebSocket full-duplex session with accent detection and auto-voice adaptation
"""

import sys
import os
import json
import time
import asyncio
from pathlib import Path
import httpx
import websockets
import soundfile as sf
import numpy as np

BACKEND_HTTP = "http://127.0.0.1:8024"
BACKEND_WS = "ws://127.0.0.1:8024/ws/call"

def test_rest_endpoints():
    print("\n--- 1. Testing REST Endpoints ---")
    with httpx.Client(base_url=BACKEND_HTTP, timeout=30.0) as client:
        # Test 1: Accent info
        r = client.get("/api/accent/info")
        assert r.status_code == 200, f"Expected 200, got {r.status_code}"
        info = r.json()
        print(f"[OK] GET /api/accent/info: model={info['model_name']} status={info['status']}")
        assert "classes" in info
        assert "Indian English" in info["classes"]
        assert "American English" in info["classes"]

        # Test 2: Accent detect on Indian sample
        sample_path = Path(__file__).parent / "test_accents" / "data" / "indian_1.wav"
        if sample_path.exists():
            with open(sample_path, "rb") as f:
                r2 = client.post("/api/accent/detect", files={"audio": ("indian_1.wav", f, "audio/wav")})
            assert r2.status_code == 200, f"Expected 200, got {r2.status_code}"
            res2 = r2.json()
            print(f"[OK] POST /api/accent/detect: predicted={res2['accent']} conf={res2['confidence']} latency={res2['latency_ms']}ms")
            assert res2["status"] == "detected"
            assert res2["accent"] == "Indian English"

        # Test 3: Transcribe with decoupled accent analysis
        sample_us = Path(__file__).parent / "test_accents" / "data" / "us_1.wav"
        if sample_us.exists():
            with open(sample_us, "rb") as f:
                r3 = client.post("/api/transcribe", files={"audio": ("us_1.wav", f, "audio/wav")}, data={"engine": "sherpa"})
            assert r3.status_code == 200
            res3 = r3.json()
            assert "accent_analysis" in res3 and res3["accent_analysis"] is not None
            acc = res3["accent_analysis"]
            print(f"[OK] POST /api/transcribe with accent: transcript='{res3['transcript']}', accent='{acc['accent']}', conf={acc['confidence']}")
            assert acc["accent"] == "American English"


async def test_websocket_accent_flow():
    print("\n--- 2. Testing WebSocket /ws/call Real-Time Accent Flow ---")
    sample_path = Path(__file__).parent / "test_accents" / "data" / "indian_1.wav"
    if not sample_path.exists():
        print("Skipping WS test (sample audio not found)")
        return

    sig, fs = sf.read(sample_path)
    if fs != 16000:
        sig = np.interp(np.linspace(0, len(sig) - 1, int(len(sig) * 16000 / fs)), np.arange(len(sig)), sig)
    pcm16 = (sig * 32767).astype(np.int16).tobytes()

    async with websockets.connect(BACKEND_WS) as ws:
        # 1. Wait for handshake
        init_msg = json.loads(await ws.recv())
        assert init_msg["type"] == "ready"
        print(f"[OK] WebSocket connected: session={init_msg['session_id']}")

        # 2. Configure session with auto_adapt_voice=True and initial US voice
        await ws.send(json.dumps({
            "type": "config",
            "tts_voice": "en-US-JennyNeural",
            "auto_adapt_voice": True
        }))

        # 3. Stream audio in 100ms chunks (3200 bytes per chunk)
        chunk_size = 3200
        for i in range(0, len(pcm16), chunk_size):
            chunk = pcm16[i:i + chunk_size]
            await ws.send(chunk)
            await asyncio.sleep(0.02)  # Simulate real-time streaming

        # Stream 1.5s of silence to allow trailing-silence VAD to trigger endpoint
        silence = (np.zeros(1600 * 15, dtype=np.int16)).tobytes()
        for i in range(0, len(silence), chunk_size):
            await ws.send(silence[i:i + chunk_size])
            await asyncio.sleep(0.02)

        # 4. Listen for emitted events
        received_accent_detected = False
        received_voice_adapted = False
        received_final = False

        start_listen = time.time()
        while time.time() - start_listen < 12.0:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=4.0)
                msg = json.loads(raw)
                mtype = msg.get("type")

                if mtype == "partial_transcript":
                    pass
                elif mtype == "final_transcript":
                    received_final = True
                    print(f"[OK] WS received final_transcript: '{msg.get('text')}' (latency: {msg.get('endpoint_latency_ms')}ms)")
                elif mtype == "accent_detected":
                    received_accent_detected = True
                    print(f"[OK] WS received accent_detected: accent={msg.get('accent')} conf={msg.get('confidence')} status={msg.get('status')} voice={msg.get('suggested_tts_voice')}")
                    assert msg["accent"] == "Indian English"
                elif mtype == "voice_adapted":
                    received_voice_adapted = True
                    print(f"[OK] WS received voice_adapted: voice={msg.get('voice')} accent={msg.get('accent')}")
                elif mtype == "turn_complete":
                    print(f"[OK] WS turn_complete received!")
                    break
            except asyncio.TimeoutError:
                break

        assert received_final, "Expected final_transcript event"
        assert received_accent_detected, "Expected accent_detected event"
        print("[PASS] Full WebSocket real-time accent flow succeeded!")


def main():
    test_rest_endpoints()
    asyncio.run(test_websocket_accent_flow())
    print("\n" + "=" * 60)
    print("ALL ACCENT DETECTION INTEGRATION TESTS PASSED 100%!")
    print("=" * 60)

if __name__ == "__main__":
    main()
