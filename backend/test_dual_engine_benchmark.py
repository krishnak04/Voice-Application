import sys
import os
import time
import httpx
import asyncio
import websockets
import json

BASE_URL = "http://127.0.0.1:8024"
SAMPLE_WAV = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "speech1.wav"))

def test_api_catalogs():
    print("=" * 60)
    print("TEST 1: API Catalogs & Unified STT Endpoints")
    print("=" * 60)
    
    with httpx.Client(base_url=BASE_URL, timeout=30.0) as client:
        # 1. Unified models endpoint
        res = client.get("/api/stt/models")
        assert res.status_code == 200, f"GET /api/stt/models failed: {res.text}"
        stt_data = res.json()
        print(f"Active Engine: {stt_data['active_engine']}")
        print(f"Whisper Models Count: {len(stt_data['whisper_models'])}")
        print(f"Sherpa Models Count: {len(stt_data['sherpa_models'])}")
        
        whisper_ids = [m["id"] for m in stt_data["whisper_models"]]
        print(f"Whisper models: {whisper_ids}")
        for required in ["tiny", "base", "small", "medium", "large-v3-turbo", "base.en"]:
            assert required in whisper_ids, f"Missing required whisper model: {required}"
            
        sherpa_ids = [m["id"] for m in stt_data["sherpa_models"]]
        print(f"Sherpa models: {sherpa_ids}")
        for required in [
            "sherpa-onnx-streaming-zipformer-en-2023-06-26",
            "sherpa-onnx-streaming-zipformer-en-2023-06-21",
            "sherpa-onnx-streaming-zipformer-en-2023-02-21",
            "sherpa-onnx-streaming-zipformer-en-20M-2023-02-17"
        ]:
            assert required in sherpa_ids, f"Missing required sherpa model: {required}"

    print(">> Unified STT Catalog Test: PASSED!\n")

def test_independent_transcription():
    print("=" * 60)
    print("TEST 2: Independent Transcription (Whisper vs Sherpa-ONNX)")
    print("=" * 60)
    
    with open(SAMPLE_WAV, "rb") as f:
        audio_bytes = f.read()
        
    with httpx.Client(base_url=BASE_URL, timeout=60.0) as client:
        # Test A: Transcribe with Faster-Whisper
        print(">> Transcribing via Faster-Whisper (base)...")
        t0 = time.time()
        res_whisper = client.post(
            "/api/transcribe",
            files={"audio": ("speech1.wav", audio_bytes, "audio/wav")},
            data={"language": "en", "model": "base", "engine": "whisper"}
        )
        t_whisper = time.time() - t0
        assert res_whisper.status_code == 200, f"Whisper transcription failed: {res_whisper.text}"
        data_whisper = res_whisper.json()
        print(f"Whisper Transcript: \"{data_whisper['transcript']}\"")
        print(f"Whisper STT Latency: {data_whisper['stt_latency']}s (RTF: {data_whisper['rtf']}, Speed: {data_whisper['speed_factor']}x)")
        print(f"Whisper Engine: {data_whisper.get('engine', 'whisper')} | Model: {data_whisper.get('model_name')}")
        
        # Test B: Transcribe with Sherpa-ONNX
        print("\n>> Transcribing via Sherpa-ONNX (Streaming Zipformer Baseline)...")
        t0 = time.time()
        res_sherpa = client.post(
            "/api/transcribe",
            files={"audio": ("speech1.wav", audio_bytes, "audio/wav")},
            data={"language": "en", "engine": "sherpa"}
        )
        t_sherpa = time.time() - t0
        assert res_sherpa.status_code == 200, f"Sherpa transcription failed: {res_sherpa.text}"
        data_sherpa = res_sherpa.json()
        print(f"Sherpa Transcript: \"{data_sherpa['transcript']}\"")
        print(f"Sherpa STT Latency: {data_sherpa['stt_latency']}s (RTF: {data_sherpa['rtf']}, Speed: {data_sherpa['speed_factor']}x)")
        print(f"Sherpa Engine: {data_sherpa.get('engine', 'sherpa')} | Model: {data_sherpa.get('model_name')}")
        
    print(">> Independent Transcription Test: PASSED!\n")

def test_engine_and_model_switching():
    print("=" * 60)
    print("TEST 3: Dynamic Model Switching without Restart")
    print("=" * 60)
    
    with httpx.Client(base_url=BASE_URL, timeout=60.0) as client:
        # Switch to whisper tiny
        print(">> Selecting Whisper tiny...")
        res = client.post("/api/stt/select", json={"engine": "whisper", "model_id": "tiny"})
        assert res.status_code == 200, f"Failed to switch to whisper tiny: {res.text}"
        print(f">> Result: {res.json()}")
        
        # Switch back to whisper base
        print(">> Switching back to Whisper base...")
        res = client.post("/api/stt/select", json={"engine": "whisper", "model_id": "base"})
        assert res.status_code == 200, f"Failed to switch to whisper base: {res.text}"
        print(f">> Result: {res.json()}")

        # Switch to sherpa 20M small
        print(">> Selecting Sherpa 20M small...")
        res = client.post("/api/stt/select", json={"engine": "sherpa", "model_id": "sherpa-onnx-streaming-zipformer-en-20M-2023-02-17"})
        assert res.status_code == 200, f"Failed to switch to sherpa 20M: {res.text}"
        print(f">> Result: {res.json()}")
        
        # Switch back to standard streaming
        print(">> Switching back to Sherpa Standard Baseline...")
        res = client.post("/api/stt/select", json={"engine": "sherpa", "model_id": "sherpa-onnx-streaming-zipformer-en-2023-06-26"})
        assert res.status_code == 200, f"Failed to switch to sherpa standard: {res.text}"
        print(f">> Result: {res.json()}")
        
    print(">> Model Switching Test: PASSED!\n")

async def test_websocket_streaming():
    print("=" * 60)
    print("TEST 4: WebSocket Full-Duplex Session & Streaming")
    print("=" * 60)
    
    ws_url = "ws://127.0.0.1:8024/ws/call"
    async with websockets.connect(ws_url) as ws:
        # Send config
        await ws.send(json.dumps({
            "type": "config",
            "tts_voice": "en-US-JennyNeural",
            "tts_engine": "edge_tts",
            "groq_model": "openai/gpt-oss-20b",
            "is_phone_call": True
        }))
        
        # Stream 1 second of PCM16 16kHz silence
        silence_pcm = b"\x00\x00" * 16000
        # Send in 2048-sample chunks
        chunk_size = 2048 * 2
        for i in range(0, len(silence_pcm), chunk_size):
            await ws.send(silence_pcm[i:i+chunk_size])
            await asyncio.sleep(0.01)
            
        print(">> WebSocket PCM chunks streamed successfully with zero exceptions.")

if __name__ == "__main__":
    test_api_catalogs()
    test_independent_transcription()
    test_engine_and_model_switching()
    asyncio.run(test_websocket_streaming())
    print("=" * 60)
    print("ALL 4 DUAL-ENGINE VERIFICATION TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 60)
