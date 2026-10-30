import asyncio
import aiohttp
import json
import wave
import time

async def test_live_barge_in():
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

    interrupted = False

    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(ws_url) as ws:
            msg = await ws.receive()
            handshake = json.loads(msg.data)
            print("Connected:", handshake)

            await ws.send_json({
                "type": "config",
                "tts_voice": "en-IN-NeerjaNeural",
                "tts_engine": "edge_tts",
                "groq_model": "openai/gpt-oss-20b"
            })

            ai_speaking_event = asyncio.Event()

            async def send_sequence():
                # Speak first sentence
                print("[T1] Speaking initial utterance...")
                for chunk in chunks[:18]:
                    await ws.send_bytes(chunk)
                    await asyncio.sleep(0.04)
                # Silence to trigger endpoint (threshold is 1.2s)
                print("[T1] Pausing for endpoint detection...")
                for _ in range(28):
                    await ws.send_bytes(silence_chunk)
                    await asyncio.sleep(0.05)

                # Wait for AI to start answering, then barge in!
                print("[T1] Waiting for AI assistant speech...")
                try:
                    await asyncio.wait_for(ai_speaking_event.wait(), timeout=10.0)
                except asyncio.TimeoutError:
                    print("[!] Timeout waiting for AI speech event")

                await asyncio.sleep(0.4)
                print("\n>>> BARGING IN NOW: User speaks over assistant!")
                for chunk in chunks[:35]:
                    await ws.send_bytes(chunk)
                    await asyncio.sleep(0.03)

            sender = asyncio.create_task(send_sequence())

            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    mtype = data.get("type")

                    if mtype == "partial_transcript":
                        print(f"Partial: {data['text']}")
                    elif mtype == "final_transcript":
                        print(f"\nFinal: {data['text']}")
                    elif mtype == "assistant_text_delta":
                        safe_delta = data.get('delta', '').encode('ascii', 'replace').decode()
                        print(f"[AI]: {safe_delta}", end="", flush=True)
                        if not ai_speaking_event.is_set():
                            ai_speaking_event.set()
                    elif mtype == "audio_chunk":
                        print(f"\n[AI Audio Chunk #{data['chunk_index']}] ({len(data.get('audio_base64', ''))} b64 chars) Playing...", flush=True)
                        if not ai_speaking_event.is_set():
                            ai_speaking_event.set()
                    elif mtype == "turn_complete":
                        print(f"\n[Turn Complete]: {data.get('metrics', {})}")
                    elif mtype == "interrupted":
                        print(f"\n\n==========================================")
                        print(f"SUCCESS: BARGE-IN INTERRUPTION DETECTED!")
                        print(f"Event: {data}")
                        print(f"Interruption latency: {data.get('interruption_latency_ms', 0)} ms")
                        print(f"==========================================\n")
                        interrupted = True
                        break
                    elif mtype == "error":
                        print(f"Error: {data}")
                        break

            await sender

    if interrupted:
        print("Barge-in test verified and passed successfully!")

if __name__ == "__main__":
    asyncio.run(test_live_barge_in())
