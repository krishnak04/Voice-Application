import asyncio
import aiohttp
import json
import wave
import time
import os
from typing import List, Dict, Any
import numpy as np

async def run_single_turn_benchmark(session: aiohttp.ClientSession, ws_url: str, wav_path: str, turn_num: int) -> Dict[str, float]:
    """Execute a single full-duplex conversational turn and record detailed timing metrics."""
    with wave.open(wav_path, "rb") as wf:
        nchannels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        nframes = wf.getnframes()
        raw_audio = wf.readframes(nframes)

    chunk_size = 320 * sampwidth  # 20ms chunks at 16kHz, matching browser streaming cadence
    chunks = [raw_audio[i:i + chunk_size] for i in range(0, len(raw_audio), chunk_size)]
    silence_chunk = b"\x00" * chunk_size
    speech_duration_s = nframes / framerate

    metrics = {
        "turn": turn_num,
        "audio_capture_to_first_partial_ms": None,
        "audio_capture_to_final_transcript_ms": None,
        "end_of_speech_to_final_transcript_ms": None,
        "speech_start_latency_ms": None,
        "endpoint_wait_ms": None,
        "finalization_latency_ms": None,
        "final_transcript_to_first_token_ms": None,
        "final_transcript_to_first_audio_ms": None,
        "end_of_speech_to_first_audio_ms": None,
        "total_turn_duration_ms": None,
        "interruption_latency_ms": None
    }

    async with session.ws_connect(ws_url) as ws:
        # Handshake
        msg = await ws.receive()
        handshake = json.loads(msg.data)

        # Send config
        await ws.send_json({
            "type": "config",
            "tts_voice": "en-IN-NeerjaNeural",
            "tts_engine": "edge_tts",
            "groq_model": "openai/gpt-oss-20b"
        })

        t_stream_start = time.perf_counter()
        t_final_transcript = None
        end_of_speech_to_final_ms = None
        t_first_token = None
        t_first_audio = None

        async def stream_audio_feed():
            for chunk in chunks:
                await ws.send_bytes(chunk)
                await asyncio.sleep(len(chunk) / (sampwidth * framerate))
            for _ in range(80):
                await ws.send_bytes(silence_chunk)
                await asyncio.sleep(len(silence_chunk) / (sampwidth * framerate))

        stream_task = asyncio.create_task(stream_audio_feed())

        async for ws_msg in ws:
            if ws_msg.type == aiohttp.WSMsgType.TEXT:
                data = json.loads(ws_msg.data)
                mtype = data.get("type")

                if mtype == "partial_transcript":
                    if metrics["audio_capture_to_first_partial_ms"] is None:
                        metrics["audio_capture_to_first_partial_ms"] = round((time.perf_counter() - t_stream_start) * 1000, 1)

                elif mtype == "final_transcript":
                    t_final_transcript = time.perf_counter()
                    metrics["audio_capture_to_final_transcript_ms"] = round((t_final_transcript - t_stream_start) * 1000, 1)
                    metrics["end_of_speech_to_final_transcript_ms"] = data.get("endpoint_latency_ms")
                    end_of_speech_to_final_ms = data.get("endpoint_latency_ms")
                    metrics["speech_start_latency_ms"] = data.get("speech_start_latency_ms")
                    metrics["endpoint_wait_ms"] = data.get("endpoint_wait_ms")
                    metrics["finalization_latency_ms"] = data.get("finalization_latency_ms")

                elif mtype == "assistant_text_delta":
                    if t_first_token is None:
                        t_first_token = time.perf_counter()
                        if t_final_transcript:
                            metrics["final_transcript_to_first_token_ms"] = round((t_first_token - t_final_transcript) * 1000, 1)

                elif mtype == "audio_chunk":
                    if t_first_audio is None:
                        t_first_audio = time.perf_counter()
                        if t_final_transcript:
                            metrics["final_transcript_to_first_audio_ms"] = round((t_first_audio - t_final_transcript) * 1000, 1)
                        final_to_audio_ms = (t_first_audio - t_final_transcript) * 1000 if t_final_transcript else 0
                        metrics["end_of_speech_to_first_audio_ms"] = round(
                            (end_of_speech_to_final_ms or 0) + final_to_audio_ms,
                            1
                        )

                elif mtype == "turn_complete":
                    t_turn_complete = time.perf_counter()
                    if t_final_transcript:
                        metrics["total_turn_duration_ms"] = round((t_turn_complete - t_final_transcript) * 1000, 1)
                    break

                elif mtype == "error":
                    print(f"Error in turn {turn_num}: {data['message']}")
                    break

        await stream_task

        # Test interruption signal
        t_interrupt_send = time.perf_counter()
        await ws.send_json({"type": "interrupt"})
        int_reply = await ws.receive()
        if int_reply.type == aiohttp.WSMsgType.TEXT:
            int_data = json.loads(int_reply.data)
            metrics["interruption_latency_ms"] = int_data.get("interruption_latency_ms", 0.0)

    return metrics

async def run_benchmark_suite(num_runs: int = 5):
    ws_url = "ws://localhost:8024/ws/call"
    wav_candidates = [
        "sherpa-onnx-streaming-zipformer-en-2023-06-26/test_wavs/0.wav",
        "../sherpa-onnx-streaming-zipformer-en-2023-06-26/test_wavs/0.wav",
        os.path.join(os.path.dirname(__file__), "..", "sherpa-onnx-streaming-zipformer-en-2023-06-26", "test_wavs", "0.wav")
    ]
    wav_path = next(p for p in wav_candidates if os.path.exists(p))

    print(f"=== FULL-DUPLEX LATENCY BENCHMARK SUITE ({num_runs} RUNS) ===")
    print(f"Server Target: {ws_url}")
    print(f"Test Utterance: {wav_path}\n")

    results: List[Dict[str, float]] = []

    async with aiohttp.ClientSession() as session:
        for r in range(1, num_runs + 1):
            print(f"Executing Turn Run #{r}/{num_runs}...", end="", flush=True)
            res = await run_single_turn_benchmark(session, ws_url, wav_path, r)
            results.append(res)
            print(f" Done. (TTFT: {res['final_transcript_to_first_token_ms']}ms, TTFA: {res['final_transcript_to_first_audio_ms']}ms, Turn: {res['total_turn_duration_ms']}ms)")
            await asyncio.sleep(0.5)

    metric_keys = [
        ("audio_capture_to_first_partial_ms", "Audio capture -> 1st partial transcript"),
        ("speech_start_latency_ms", "Microphone speech start detection"),
        ("endpoint_wait_ms", "Speech end -> endpoint detection"),
        ("finalization_latency_ms", "Endpoint -> final transcript"),
        ("audio_capture_to_final_transcript_ms", "Audio capture -> Final transcript"),
        ("end_of_speech_to_final_transcript_ms", "End of speech -> Final transcript (VAD)"),
        ("final_transcript_to_first_token_ms", "Final transcript -> 1st LLM token (TTFT)"),
        ("final_transcript_to_first_audio_ms", "Final transcript -> 1st audio chunk (TTFA)"),
        ("end_of_speech_to_first_audio_ms", "End of speech -> 1st audible assistant voice"),
        ("total_turn_duration_ms", "Total turn response duration"),
        ("interruption_latency_ms", "User interruption -> Playback stop latency"),
    ]

    print("\n" + "=" * 76)
    print(f"{'LATENCY METRIC':<46} | {'MEDIAN (ms)':<12} | {'P95 (ms)':<10}")
    print("=" * 76)

    for key, name in metric_keys:
        vals = [r[key] for r in results if r.get(key) is not None]
        if vals:
            median_val = float(np.median(vals))
            p95_val = float(np.percentile(vals, 95))
            print(f"{name:<46} | {median_val:10.1f} ms | {p95_val:8.1f} ms")
        else:
            print(f"{name:<46} | {'N/A':>10} | {'N/A':>8}")

    print("=" * 76 + "\n")

if __name__ == "__main__":
    asyncio.run(run_benchmark_suite(num_runs=2))
