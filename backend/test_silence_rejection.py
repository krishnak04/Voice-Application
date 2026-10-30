"""Regression tests for silence/noise rejection in the full-duplex WebSocket."""

import asyncio
import json
import math
import wave
from pathlib import Path

import numpy as np
import websockets

BACKEND_WS = "ws://127.0.0.1:8024/ws/call"
SAMPLE_RATE = 16000
FRAME_SAMPLES = 320
DATASET = Path(__file__).parent / "test_audio_dataset"


def _pcm16(samples: np.ndarray) -> bytes:
    return np.clip(samples, -1.0, 1.0).astype(np.float32).astype(np.int16).tobytes()


def _silence(duration: float = 1.6) -> bytes:
    return b"\x00\x00" * int(SAMPLE_RATE * duration)


def _fan_noise(duration: float = 1.6) -> bytes:
    count = int(SAMPLE_RATE * duration)
    t = np.arange(count, dtype=np.float32) / SAMPLE_RATE
    noise = 0.006 * np.sin(2 * np.pi * 120 * t) + 0.002 * np.sin(2 * np.pi * 240 * t)
    return _pcm16(noise)


def _keyboard_noise(duration: float = 1.6) -> bytes:
    count = int(SAMPLE_RATE * duration)
    samples = np.zeros(count, dtype=np.float32)
    for start in range(500, count, 1800):
        length = min(80, count - start)
        samples[start:start + length] = np.hanning(length) * 0.18
    return _pcm16(samples)


def _load_speech(name: str = "sample_01.wav") -> bytes:
    with wave.open(str(DATASET / name), "rb") as wav:
        return wav.readframes(wav.getnframes())


async def _run_turn(audio: bytes, wait_for_final: bool = False) -> list[dict]:
    events: list[dict] = []
    async with websockets.connect(BACKEND_WS, open_timeout=30, close_timeout=5) as ws:
        await ws.recv()
        await ws.send(json.dumps({"type": "config", "tts_engine": "edge_tts"}))

        async def send_audio() -> None:
            for offset in range(0, len(audio), FRAME_SAMPLES * 2):
                frame = audio[offset:offset + FRAME_SAMPLES * 2]
                await ws.send(frame)
                await asyncio.sleep(len(frame) / 2 / SAMPLE_RATE)

        sender = asyncio.create_task(send_audio())
        try:
            deadline = 45 if wait_for_final else 4
            while True:
                try:
                    event = json.loads(await asyncio.wait_for(ws.recv(), timeout=deadline))
                except asyncio.TimeoutError:
                    break
                events.append(event)
                if event.get("type") in ("final_transcript", "error", "turn_complete"):
                    if wait_for_final or event.get("type") == "error":
                        break
        finally:
            if not sender.done():
                sender.cancel()
            try:
                await sender
            except asyncio.CancelledError:
                pass
    return events


async def main() -> None:
    negative_cases = {
        "silence": _silence(),
        "fan_noise": _fan_noise(),
        "keyboard_noise": _keyboard_noise(),
    }

    for name, audio in negative_cases.items():
        for repeat in range(3):
            events = await _run_turn(audio)
            accepted = [e for e in events if e.get("type") == "final_transcript"]
            assert not accepted, f"{name} repeat {repeat + 1} produced {accepted}"
            assert not any(e.get("type") == "assistant_text_delta" for e in events), (
                f"{name} repeat {repeat + 1} reached Groq"
            )
        print(f"[PASS] {name}: 0 accepted transcripts across 3 repeats")

    speech_events = await _run_turn(_load_speech(), wait_for_final=True)
    finals = [e for e in speech_events if e.get("type") == "final_transcript"]
    assert len(finals) == 1, f"Expected one genuine final transcript, got {len(finals)}"
    assert finals[0].get("text", "").strip(), "Genuine speech produced an empty transcript"
    print(f"[PASS] genuine speech: {finals[0]['text']!r}")

    duplicate_events = await _run_turn(_load_speech() + _silence(2.4), wait_for_final=True)
    duplicate_finals = [e for e in duplicate_events if e.get("type") == "final_transcript"]
    assert len(duplicate_finals) == 1, f"Expected no duplicate final, got {len(duplicate_finals)}"
    print("[PASS] post-turn silence: no duplicate transcript")


if __name__ == "__main__":
    asyncio.run(main())
