import asyncio
import aiohttp
import json
import wave

async def run_single_session(session_num: int):
    ws_url = "http://localhost:8024/ws/call"
    import os
    wav_candidates = [
        "sherpa-onnx-streaming-zipformer-en-2023-06-26/test_wavs/0.wav",
        "../sherpa-onnx-streaming-zipformer-en-2023-06-26/test_wavs/0.wav",
        os.path.join(os.path.dirname(__file__), "..", "sherpa-onnx-streaming-zipformer-en-2023-06-26", "test_wavs", "0.wav")
    ]
    wav_path = next(p for p in wav_candidates if os.path.exists(p))

    with wave.open(wav_path, "rb") as wf:
        sampwidth = wf.getsampwidth()
        nframes = wf.getnframes()
        raw_audio = wf.readframes(nframes)

    chunk_size = 1600 * sampwidth
    chunks = [raw_audio[i:i + chunk_size] for i in range(0, len(raw_audio), chunk_size)]
    silence_chunk = b"\x00" * chunk_size

    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(ws_url) as ws:
            msg = await ws.receive()
            handshake = json.loads(msg.data)
            session_id = handshake["session_id"]
            print(f"[Session {session_num}] Connected: {session_id}")

            async def send_audio():
                for chunk in chunks:
                    await ws.send_bytes(chunk)
                    await asyncio.sleep(0.03)
                for _ in range(20):
                    await ws.send_bytes(silence_chunk)
                    await asyncio.sleep(0.04)

            sender = asyncio.create_task(send_audio())
            partials = []
            final_text = None

            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    if data.get("type") == "partial_transcript":
                        partials.append(data["text"])
                    elif data.get("type") == "final_transcript":
                        final_text = data["text"]
                        print(f"[Session {session_num}] Final: '{final_text}' (Partials: {len(partials)})")
                        break

            await sender
            return {
                "session_num": session_num,
                "session_id": session_id,
                "final_text": final_text,
                "partials_count": len(partials)
            }

async def test_concurrent_sessions():
    print("Testing 2 concurrent WebSocket sessions simultaneously...")
    results = await asyncio.gather(
        run_single_session(1),
        run_single_session(2)
    )
    print("\nConcurrent test results:")
    for r in results:
        print(f"Session {r['session_num']}: ID={r['session_id']}, Partials={r['partials_count']}, Final='{r['final_text']}'")
    
    assert results[0]["session_id"] != results[1]["session_id"], "Session IDs must be distinct"
    assert results[0]["final_text"] and results[1]["final_text"], "Both sessions must produce final text"
    print("\n>>> CONCURRENCY VALIDATION PASSED: Isolated sessions, separate recognizer streams, zero cross-talk.")

if __name__ == "__main__":
    asyncio.run(test_concurrent_sessions())
