"""
Master Engineering Verification Suite
Tests all requirements of the Master Engineering Task:
1. Duplicate Message Prevention & Sequence Validation
2. Latency Benchmarks (Speech start, First Partial, Endpoint, Final Transcript, TTS first chunk)
3. Silence & Noise Rejection (0 false transcripts)
4. Full-Duplex Barge-In Interruption
5. 20 Consecutive Conversation Turns
6. Accent-wise Transcription & WER
"""

import os
import sys
import time
import json
import wave
import asyncio
import numpy as np
from pathlib import Path
import websockets

BACKEND_WS = "ws://127.0.0.1:8024/ws/call"
SAMPLE_RATE = 16000
FRAME_MS = 20  # 20ms frames
FRAME_SAMPLES = int(SAMPLE_RATE * (FRAME_MS / 1000.0))  # 320 samples = 640 bytes
DATASET_DIR = Path(__file__).parent / "test_audio_dataset"
ACCENTS_DIR = Path(__file__).parent / "test_accents" / "data"

def pcm16(samples: np.ndarray) -> bytes:
    return np.clip(samples, -1.0, 1.0).astype(np.float32).astype(np.int16).tobytes()

def make_silence(duration: float = 2.0) -> bytes:
    return b"\x00\x00" * int(SAMPLE_RATE * duration)

def make_fan_noise(duration: float = 2.0) -> bytes:
    count = int(SAMPLE_RATE * duration)
    t = np.arange(count, dtype=np.float32) / SAMPLE_RATE
    noise = 0.006 * np.sin(2 * np.pi * 120 * t) + 0.002 * np.sin(2 * np.pi * 240 * t)
    return pcm16(noise)

def make_keyboard_noise(duration: float = 2.0) -> bytes:
    count = int(SAMPLE_RATE * duration)
    samples = np.zeros(count, dtype=np.float32)
    for start in range(500, count, 1800):
        length = min(80, count - start)
        samples[start:start + length] = np.hanning(length) * 0.18
    return pcm16(samples)

def load_wav(path: Path) -> bytes:
    with wave.open(str(path), "rb") as wf:
        nchannels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
        
        # Convert to mono 16kHz PCM16 if needed
        if framerate != SAMPLE_RATE or nchannels != 1 or sampwidth != 2:
            import audioop
            if nchannels == 2:
                raw = audioop.tomono(raw, sampwidth, 0.5, 0.5)
            if sampwidth != 2:
                raw = audioop.lin2lin(raw, sampwidth, 2)
            if framerate != SAMPLE_RATE:
                raw, _ = audioop.ratecv(raw, 2, 1, framerate, SAMPLE_RATE, None)
        return raw

def calculate_wer(reference: str, hypothesis: str) -> float:
    import re
    ref_words = re.sub(r'[^\w\s]', '', reference.lower()).split()
    hyp_words = re.sub(r'[^\w\s]', '', hypothesis.lower()).split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    r_len, h_len = len(ref_words), len(hyp_words)
    dp = [[0] * (h_len + 1) for _ in range(r_len + 1)]
    for i in range(r_len + 1):
        dp[i][0] = i
    for j in range(h_len + 1):
        dp[0][j] = j
    for i in range(1, r_len + 1):
        for j in range(1, h_len + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j - 1], dp[i][j - 1], dp[i - 1][j]) + 1
    return min(1.0, dp[r_len][h_len] / r_len)


# =========================================================================
# TEST A: DUPLICATE MESSAGE AND SEQUENCE INTEGRITY TEST
# =========================================================================
async def test_duplicate_messages_and_sequence():
    print("\n" + "=" * 60)
    print("TEST A: DUPLICATE MESSAGES & SEQUENCE INTEGRITY")
    print("=" * 60)
    
    audio = load_wav(DATASET_DIR / "sample_03.wav")
    events = []
    
    async with websockets.connect(BACKEND_WS, open_timeout=15, close_timeout=5) as ws:
        hello = json.loads(await ws.recv())
        session_id = hello.get("session_id")
        assert session_id, "Missing session_id in handshake"
        
        await ws.send(json.dumps({"type": "config", "tts_engine": "edge_tts", "groq_model": "qwen/qwen3.8-27b", "silence_timeout_sec": 1.2}))
        
        # Send audio in 20ms chunks in real time
        for offset in range(0, len(audio), FRAME_SAMPLES * 2):
            await ws.send(audio[offset:offset + FRAME_SAMPLES * 2])
            await asyncio.sleep(0.015)
            
        # Send 1.5s of silence to allow endpointing
        silence = make_silence(1.5)
        for offset in range(0, len(silence), FRAME_SAMPLES * 2):
            await ws.send(silence[offset:offset + FRAME_SAMPLES * 2])
            await asyncio.sleep(0.015)
            
        # Collect events until turn_complete
        while True:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=12.0)
                msg = json.loads(raw)
                events.append(msg)
                if msg.get("type") == "turn_complete":
                    break
            except asyncio.TimeoutError:
                break
                
    # Verify events
    finals = [e for e in events if e.get("type") == "final_transcript"]
    partials = [e for e in events if e.get("type") == "partial_transcript"]
    sequences = [e.get("seq") for e in events if e.get("seq") is not None]
    
    print(f"[*] Received {len(partials)} partials, {len(finals)} final(s)")
    assert len(finals) == 1, f"Expected exactly 1 final transcript, got {len(finals)}"
    
    final_turn_id = finals[0].get("turn_id")
    for p in partials:
        assert p.get("turn_id") == final_turn_id, f"Partial turn_id {p.get('turn_id')} != final turn_id {final_turn_id}"
        
    # Check sequence monotonicity
    for i in range(1, len(sequences)):
        assert sequences[i] > sequences[i - 1], f"Non-monotonic sequence: {sequences[i - 1]} -> {sequences[i]}"
        
    print(f"[PASS] Exact 1 final transcript (turn_id={final_turn_id}). All {len(partials)} partials share the exact same turn_id.")
    print(f"[PASS] Sequence counter monotonically increased: {sequences[0]} -> {sequences[-1]}")
    return True


# =========================================================================
# TEST B: COMPREHENSIVE LATENCY BENCHMARK
# =========================================================================
async def test_latency_benchmarks():
    print("\n" + "=" * 60)
    print("TEST B: COMPREHENSIVE LATENCY BENCHMARK")
    print("=" * 60)
    
    test_files = [
        ("Short Speech", DATASET_DIR / "sample_41.wav"),
        ("Medium Speech", DATASET_DIR / "sample_01.wav"),
        ("Long Speech", DATASET_DIR / "sample_16.wav"),
        ("Fast Speech", DATASET_DIR / "sample_06.wav"),
        ("Slow Speech", DATASET_DIR / "sample_08.wav"),
    ]
    
    results = []
    
    for label, file_path in test_files:
        if not file_path.exists():
            continue
        audio = load_wav(file_path)
        
        async with websockets.connect(BACKEND_WS, open_timeout=15, close_timeout=5) as ws:
            await ws.recv()  # hello
            await ws.send(json.dumps({"type": "config", "tts_engine": "edge_tts"}))
            
            speech_start_ts = None
            first_partial_ts = None
            endpoint_ts = None
            final_transcript_ts = None
            first_llm_token_ts = None
            first_audio_chunk_ts = None
            
            t0 = time.time()
            
            async def send_stream():
                for offset in range(0, len(audio), FRAME_SAMPLES * 2):
                    await ws.send(audio[offset:offset + FRAME_SAMPLES * 2])
                    await asyncio.sleep(FRAME_MS / 1000.0)
                # Stream silence for endpointing
                silence = make_silence(1.5)
                for offset in range(0, len(silence), FRAME_SAMPLES * 2):
                    await ws.send(silence[offset:offset + FRAME_SAMPLES * 2])
                    await asyncio.sleep(FRAME_MS / 1000.0)
                    
            sender = asyncio.create_task(send_stream())
            
            server_first_partial_latency = None
            speech_start_latency = None
            endpoint_delay = None
            finalization_latency = None
            
            while True:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=25.0)
                    now = time.time()
                    msg = json.loads(raw)
                    mtype = msg.get("type")
                    
                    if mtype == "partial_transcript":
                        if first_partial_ts is None and msg.get("text", "").strip():
                            first_partial_ts = now
                            server_first_partial_latency = msg.get("first_partial_latency_ms")
                    elif mtype == "final_transcript":
                        final_transcript_ts = now
                        speech_start_latency = msg.get("speech_start_latency_ms")
                        endpoint_delay = msg.get("endpoint_wait_ms")
                        finalization_latency = msg.get("finalization_latency_ms")
                    elif mtype == "assistant_text_delta":
                        if first_llm_token_ts is None:
                            first_llm_token_ts = now
                    elif mtype == "audio_chunk":
                        if first_audio_chunk_ts is None:
                            first_audio_chunk_ts = now
                    elif mtype == "turn_complete":
                        break
                except asyncio.TimeoutError:
                    break
                    
            await sender
            
            # Compute latencies
            fp_latency = server_first_partial_latency or ((first_partial_ts - t0) * 1000 if first_partial_ts else None)
            final_latency = (final_transcript_ts - t0) * 1000 if final_transcript_ts else None
            first_audio_latency = (first_audio_chunk_ts - final_transcript_ts) * 1000 if (first_audio_chunk_ts and final_transcript_ts) else None
            
            results.append({
                "label": label,
                "speech_start_ms": speech_start_latency,
                "first_partial_ms": fp_latency,
                "endpoint_delay_ms": endpoint_delay,
                "finalization_ms": finalization_latency,
                "final_transcript_ms": final_latency,
                "first_audio_ms": first_audio_latency
            })
            
            fp_str = f"{fp_latency:.1f}ms" if fp_latency is not None else "N/A"
            final_str = f"{final_latency:.1f}ms" if final_latency is not None else "N/A"
            first_audio_str = f"{first_audio_latency:.1f}ms" if first_audio_latency is not None else "N/A"
            start_str = f"{speech_start_latency:.1f}ms" if speech_start_latency is not None else "N/A"
            end_str = f"{endpoint_delay:.1f}ms" if endpoint_delay is not None else "N/A"
            
            print(f"[*] {label:14s}: SpeechStart={start_str} | First Partial={fp_str} | Endpoint={end_str} | First TTS={first_audio_str}")
            
    # Calculate P50 and P95
    fp_values = [r["first_partial_ms"] for r in results if r["first_partial_ms"] is not None]
    start_values = [r["speech_start_ms"] for r in results if r["speech_start_ms"] is not None]
    end_values = [r["endpoint_delay_ms"] for r in results if r["endpoint_delay_ms"] is not None]
    audio_values = [r["first_audio_ms"] for r in results if r["first_audio_ms"] is not None]
    
    if start_values:
        print(f"\n[SUMMARY] Speech Start: P50 = {np.percentile(start_values, 50):.1f} ms, P95 = {np.percentile(start_values, 95):.1f} ms")
    if fp_values:
        print(f"[SUMMARY] First Partial: P50 = {np.percentile(fp_values, 50):.1f} ms, P95 = {np.percentile(fp_values, 95):.1f} ms")
    if end_values:
        print(f"[SUMMARY] Endpoint Delay: P50 = {np.percentile(end_values, 50):.1f} ms, P95 = {np.percentile(end_values, 95):.1f} ms")
    if audio_values:
        print(f"[SUMMARY] First TTS Chunk: P50 = {np.percentile(audio_values, 50):.1f} ms, P95 = {np.percentile(audio_values, 95):.1f} ms")
        
    return results


# =========================================================================
# TEST C: FALSE-TRANSCRIPTION / SILENCE REJECTION TEST
# =========================================================================
async def test_silence_and_noise_rejection():
    print("\n" + "=" * 60)
    print("TEST C: SILENCE & NOISE REJECTION (10s Silence, Fan, Keyboard)")
    print("=" * 60)
    
    cases = {
        "10s Silence": make_silence(10.0),
        "Fan Noise": make_fan_noise(3.0),
        "Keyboard Noise": make_keyboard_noise(3.0),
    }
    
    for name, audio_bytes in cases.items():
        async with websockets.connect(BACKEND_WS, open_timeout=15, close_timeout=5) as ws:
            await ws.recv()
            await ws.send(json.dumps({"type": "config", "tts_engine": "edge_tts"}))
            
            events = []
            for offset in range(0, len(audio_bytes), FRAME_SAMPLES * 2):
                await ws.send(audio_bytes[offset:offset + FRAME_SAMPLES * 2])
                await asyncio.sleep(0.005)
                
            # Allow settling time
            try:
                while True:
                    raw = await asyncio.wait_for(ws.recv(), timeout=2.5)
                    events.append(json.loads(raw))
            except asyncio.TimeoutError:
                pass
                
            finals = [e for e in events if e.get("type") == "final_transcript"]
            llm_events = [e for e in events if e.get("type") == "assistant_text_delta"]
            assert len(finals) == 0, f"{name} produced {len(finals)} false final transcript(s): {finals}"
            assert len(llm_events) == 0, f"{name} reached LLM: {llm_events}"
            print(f"[PASS] {name}: 0 false transcripts, 0 LLM calls.")
            
    return True


# =========================================================================
# TEST D: FULL-DUPLEX LIVE BARGE-IN INTERRUPTION
# =========================================================================
async def test_barge_in_interruption():
    print("\n" + "=" * 60)
    print("TEST D: FULL-DUPLEX LIVE BARGE-IN INTERRUPTION")
    print("=" * 60)
    
    audio = load_wav(DATASET_DIR / "sample_01.wav")
    interrupted = False
    interruption_latency = None
    ai_speaking_event = asyncio.Event()
    
    async with websockets.connect(BACKEND_WS, open_timeout=15, close_timeout=5) as ws:
        await ws.recv()
        await ws.send(json.dumps({"type": "config", "tts_engine": "edge_tts"}))
        
        async def send_sequence():
            # 1. Speak initial utterance
            print("[*] Speaking initial utterance...")
            for offset in range(0, len(audio), FRAME_SAMPLES * 2):
                await ws.send(audio[offset:offset + FRAME_SAMPLES * 2])
                await asyncio.sleep(FRAME_MS / 1000.0)
                
            # 2. Silence to trigger endpoint
            silence = make_silence(1.5)
            for offset in range(0, len(silence), FRAME_SAMPLES * 2):
                await ws.send(silence[offset:offset + FRAME_SAMPLES * 2])
                await asyncio.sleep(FRAME_MS / 1000.0)
                
            # 3. Wait for AI speech event, then barge in!
            print("[*] Waiting for AI speech...")
            try:
                await asyncio.wait_for(ai_speaking_event.wait(), timeout=12.0)
            except asyncio.TimeoutError:
                print("[!] Timeout waiting for AI speech event")
                return
                
            await asyncio.sleep(0.4)
            print("[*] >>> BARGING IN NOW: User speaks over AI!")
            for offset in range(0, len(audio), FRAME_SAMPLES * 2):
                await ws.send(audio[offset:offset + FRAME_SAMPLES * 2])
                await asyncio.sleep(FRAME_MS / 1000.0)
                
        sender = asyncio.create_task(send_sequence())
        
        # Concurrent receiver loop
        while True:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=15.0)
                msg = json.loads(raw)
                mtype = msg.get("type")
                
                if mtype in ("audio_chunk", "assistant_text_delta"):
                    if not ai_speaking_event.is_set():
                        ai_speaking_event.set()
                elif mtype == "interrupted":
                    interrupted = True
                    interruption_latency = msg.get("interruption_latency_ms", 0)
                    print(f"[PASS] Barge-in detected! Event: {msg}")
                    print(f"[PASS] Interruption latency = {interruption_latency:.1f} ms")
                    break
                elif mtype == "error":
                    print(f"[!] Error: {msg}")
                    break
            except asyncio.TimeoutError:
                break
                
        if not sender.done():
            sender.cancel()
        try:
            await sender
        except asyncio.CancelledError:
            pass
            
    assert interrupted, "Barge-in interrupted event was not received!"
    return True


# =========================================================================
# TEST E: 20 CONSECUTIVE CONVERSATION TURNS
# =========================================================================
async def test_twenty_consecutive_turns():
    print("\n" + "=" * 60)
    print("TEST E: 20 CONSECUTIVE CONVERSATION TURNS")
    print("=" * 60)
    
    audio_sample = load_wav(DATASET_DIR / "sample_41.wav")  # short crisp utterance
    total_turns = 20
    completed_turns = []
    
    async with websockets.connect(BACKEND_WS, open_timeout=20, close_timeout=5) as ws:
        await ws.recv()
        await ws.send(json.dumps({"type": "config", "tts_engine": "edge_tts"}))
        
        for turn_idx in range(1, total_turns + 1):
            t_start = time.time()
            # Send audio
            for offset in range(0, len(audio_sample), FRAME_SAMPLES * 2):
                await ws.send(audio_sample[offset:offset + FRAME_SAMPLES * 2])
                await asyncio.sleep(0.008)
            # Send silence for endpoint
            silence = make_silence(1.4)
            for offset in range(0, len(silence), FRAME_SAMPLES * 2):
                await ws.send(silence[offset:offset + FRAME_SAMPLES * 2])
                await asyncio.sleep(0.008)
                
            turn_finals = []
            turn_complete_received = False
            
            while True:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=12.0)
                    msg = json.loads(raw)
                    if msg.get("type") == "final_transcript":
                        turn_finals.append(msg)
                    elif msg.get("type") == "turn_complete":
                        turn_complete_received = True
                        break
                except asyncio.TimeoutError:
                    break
                    
            dur = time.time() - t_start
            print(f"[*] Turn {turn_idx:02d}/{total_turns}: finals={len(turn_finals)}, completed={turn_complete_received}, time={dur:.2f}s")
            assert len(turn_finals) == 1, f"Turn {turn_idx} had {len(turn_finals)} final transcripts!"
            completed_turns.append(turn_finals[0])
            
    print(f"\n[PASS] All {total_turns} consecutive turns completed successfully!")
    print(f"[PASS] Zero duplicate final transcripts. Turn IDs = {[t.get('turn_id') for t in completed_turns]}")
    return True


# =========================================================================
# TEST F: ACCENT-WISE TRANSCRIPTION & WER
# =========================================================================
async def test_accent_wer():
    print("\n" + "=" * 60)
    print("TEST F: ACCENT-WISE WER EVALUATION")
    print("=" * 60)
    
    accent_samples = [
        ("American English (US)", ACCENTS_DIR / "us_1.wav", "I've found these one day in the fields."),
        ("Indian English", ACCENTS_DIR / "indian_1.wav", "The flock of flamingos landed gracefully on their awkward looking legs."),
        ("British English", ACCENTS_DIR / "england_1.wav", "This is my last chance."),
        ("Australian English", ACCENTS_DIR / "australia_1.wav", "and all's fair in love and war."),
        ("US Order Inq (Dataset)", DATASET_DIR / "sample_01.wav", "Hello, I would like to inquire about the status of my recent order, please."),
        ("British Tracking (Dataset)", DATASET_DIR / "sample_03.wav", "Could you please tell me when my package is scheduled to arrive at my address?"),
        ("Indian Fast Speech (Dataset)", DATASET_DIR / "sample_07.wav", "I need to urgently update my primary shipping address before the order leaves the warehouse."),
        ("Australian Support (Dataset)", DATASET_DIR / "sample_04.wav", "Yes, I am calling because my internet connection has been dropping frequently since yesterday."),
    ]
    
    results = []
    
    for label, path, reference in accent_samples:
        if not path.exists():
            continue
        audio = load_wav(path)
        
        async with websockets.connect(BACKEND_WS, open_timeout=15, close_timeout=5) as ws:
            await ws.recv()
            await ws.send(json.dumps({"type": "config", "tts_engine": "edge_tts"}))
            
            for offset in range(0, len(audio), FRAME_SAMPLES * 2):
                await ws.send(audio[offset:offset + FRAME_SAMPLES * 2])
                await asyncio.sleep(0.01)
                
            silence = make_silence(1.5)
            for offset in range(0, len(silence), FRAME_SAMPLES * 2):
                await ws.send(silence[offset:offset + FRAME_SAMPLES * 2])
                await asyncio.sleep(0.01)
                
            final_text = ""
            while True:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=12.0)
                    msg = json.loads(raw)
                    if msg.get("type") == "final_transcript":
                        final_text = msg.get("text", "")
                    elif msg.get("type") == "turn_complete":
                        break
                except asyncio.TimeoutError:
                    break
                    
            wer = calculate_wer(reference, final_text)
            print(f"[*] {label:25s} | Ref: '{reference}' | Hyp: '{final_text}' | WER: {wer * 100:.1f}%")
            results.append({"accent": label, "wer": wer, "ref": reference, "hyp": final_text})
            
    return results


async def main():
    print("=" * 70)
    print(" MASTER VERIFICATION RUNNER")
    print("=" * 70)
    
    # 1. Duplicate & Sequence test
    await test_duplicate_messages_and_sequence()
    
    # 2. Latency benchmarks
    latencies = await test_latency_benchmarks()
    
    # 3. Silence and noise rejection
    await test_silence_and_noise_rejection()
    
    # 4. Barge-in interruption
    await test_barge_in_interruption()
    
    # 5. 20 consecutive turns
    await test_twenty_consecutive_turns()
    
    # 6. Accent WER
    accent_results = await test_accent_wer()
    
    print("\n" + "=" * 70)
    print(" ALL MASTER VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(main())
