import asyncio
import aiohttp
import json
import wave
import time
import os

async def test_handshakes():
    endpoints = [
        ("Direct port 8024 (/ws/call)", "ws://127.0.0.1:8024/ws/call"),
        ("Direct port 8024 (/ws/call/)", "ws://127.0.0.1:8024/ws/call/"),
        ("Direct port 8024 (/api/ws/call)", "ws://127.0.0.1:8024/api/ws/call"),
        ("Vite Proxy port 8023 (/ws/call)", "ws://127.0.0.1:8023/ws/call"),
        ("Vite Proxy localhost:8023 (/ws/call)", "ws://localhost:8023/ws/call"),
    ]
    origins = [
        "http://localhost:8023",
        "http://127.0.0.1:8023",
        "http://61.2.142.91:8023",
    ]

    print("=== 1. TESTING WEBSOCKET HANDSHAKES & ROUTES ===")
    all_passed = True
    async with aiohttp.ClientSession() as session:
        for name, url in endpoints:
            for origin in origins:
                try:
                    async with session.ws_connect(
                        url,
                        headers={"Origin": origin},
                        timeout=aiohttp.ClientTimeout(total=8)
                    ) as ws:
                        msg = await ws.receive_json(timeout=6)
                        assert msg.get("type") in ("ready", "session_id", "connected", "call_status"), f"Unexpected msg: {msg}"
                        print(f"  [PASS] {name} | Origin: {origin} -> session: {msg.get('session_id', 'ok')[:8]}...")
                except Exception as e:
                    print(f"  [FAIL] {name} | Origin: {origin} -> {type(e).__name__}: {e}")
                    all_passed = False
    return all_passed

async def test_full_audio_stream():
    print("\n=== 2. TESTING FULL AUDIO STREAMING, PARTIAL & FINAL TRANSCRIPTS ===")
    wav_path = os.path.join(os.path.dirname(__file__), "test_accents", "data", "us_1.wav")
    assert os.path.exists(wav_path), f"Audio file not found: {wav_path}"

    with wave.open(wav_path, "rb") as wf:
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        nframes = wf.getnframes()
        raw_audio = wf.readframes(nframes)

    # 20ms chunks (320 samples @ 16kHz)
    chunk_size = 320 * sampwidth
    chunks = [raw_audio[i:i + chunk_size] for i in range(0, len(raw_audio), chunk_size)]
    silence_chunk = b"\x00" * chunk_size

    # Test via Vite proxy to be 100% sure Vite proxy and backend work end-to-end
    target_urls = [
        ("Direct Backend", "ws://127.0.0.1:8024/ws/call"),
        ("Vite Proxy", "ws://127.0.0.1:8023/ws/call"),
    ]

    for label, ws_url in target_urls:
        print(f"\n--- Testing {label} ({ws_url}) ---")
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(ws_url, headers={"Origin": "http://localhost:8023"}) as ws:
                # 1. Receive handshake
                init_msg = await ws.receive_json(timeout=5)
                print(f"  Received handshake: type={init_msg.get('type')}, session_id={init_msg.get('session_id')[:8]}...")

                # 2. Configure session
                await ws.send_json({
                    "type": "config",
                    "tts_voice": "en-US-JennyNeural",
                    "tts_engine": "edge_tts",
                    "groq_model": "openai/gpt-oss-20b"
                })

                # 3. Stream audio in background
                async def streamer():
                    # Stream voice chunks
                    for chunk in chunks:
                        await ws.send_bytes(chunk)
                        await asyncio.sleep(0.02)
                    # Stream silence chunks to trigger endpointing (100 * 20ms = 2.0s > 1.2s threshold)
                    for _ in range(100):
                        await ws.send_bytes(silence_chunk)
                        await asyncio.sleep(0.02)

                stream_task = asyncio.create_task(streamer())

                # 4. Listen for events
                received_events = []
                partial_count = 0
                final_text = None
                ai_started = False
                audio_chunks_count = 0

                deadline = time.time() + 25
                while time.time() < deadline:
                    try:
                        msg = await ws.receive(timeout=10)
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            data = json.loads(msg.data)
                            mtype = data.get("type")
                            received_events.append(mtype)

                            if mtype == "partial_transcript":
                                partial_count += 1
                                print(f"  [PARTIAL #{partial_count}] {data.get('text')}")
                            elif mtype == "final_transcript":
                                final_text = data.get("text")
                                print(f"  [FINAL] '{final_text}' (confidence={data.get('confidence')})")
                            elif mtype == "ai_response_start":
                                ai_started = True
                                print(f"  [AI START] AI generation started")
                            elif mtype == "audio_chunk":
                                audio_chunks_count += 1
                                if audio_chunks_count == 1:
                                    print(f"  [TTS AUDIO] First audio chunk received! size={len(data.get('audio', ''))}")
                            elif mtype == "ai_response_end":
                                print(f"  [AI END] AI turn finished")
                                break
                        elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                            print(f"  WebSocket closed unexpectedly: {msg.type}")
                            break
                    except asyncio.TimeoutError:
                        break

                await stream_task

                print(f"  Summary for {label}:")
                print(f"    Partials received: {partial_count}")
                print(f"    Final transcript: {final_text}")
                print(f"    AI Response started: {ai_started}")
                print(f"    TTS audio chunks: {audio_chunks_count}")

                assert partial_count > 0 or final_text, f"No transcript received on {label}!"
                assert final_text, f"No final transcript produced on {label}!"
                print(f"  --> {label} TEST PASSED!")

async def main():
    handshake_ok = await test_handshakes()
    if not handshake_ok:
        print("\nHandshake tests had failures! Exiting.")
        return 1
    await test_full_audio_stream()
    print("\nALL WEBSOCKET TESTS PASSED!")
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))
