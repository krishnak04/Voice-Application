"""
Diagnostic test to compare streaming incremental chunking vs full buffer transcription.
Reproduces what happens over the WebSocket in Phone Call Simulator.
"""

import sys
import os
import wave
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sherpa_service import get_sherpa_engine

def test_incremental_streaming():
    sherpa = get_sherpa_engine()
    sherpa.switch_model("sherpa-onnx-streaming-zipformer-en-2023-06-26")

    wav_path = "test_audio_dataset/sample_02.wav"
    with wave.open(wav_path, "rb") as wf:
        raw_audio = wf.readframes(wf.getnframes())

    samples = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0

    print("--- 1. Full buffer with transcribe_samples ---")
    full_transcript = sherpa.transcribe_samples(samples, 16000)
    print(f"Full transcript: '{full_transcript}'")

    print("\n--- 2. Incremental streaming via session stream (OLD WAY: no input_finished) ---")
    session_id = "test_diag_1"
    stream = sherpa.create_stream(session_id)
    
    # 2048-sample chunks as sent by ScriptProcessorNode
    chunk_size = 2048
    for i in range(0, len(samples), chunk_size):
        chunk = samples[i:i + chunk_size]
        sherpa.feed_audio(session_id, chunk)

    # Old way: get_final_transcript just called get_partial_transcript
    old_final = sherpa.get_final_transcript(session_id)
    print(f"Old final transcript (without input_finished): '{old_final}'")
    sherpa.close_stream(session_id)

    print("\n--- 3. Incremental streaming (NEW WAY: with input_finished & decode flush) ---")
    session_id_2 = "test_diag_2"
    stream_2 = sherpa.create_stream(session_id_2)
    for i in range(0, len(samples), chunk_size):
        chunk = samples[i:i + chunk_size]
        sherpa.feed_audio(session_id_2, chunk)

    # New way: signal input_finished, flush decoders
    stream_2.input_finished()
    while sherpa.recognizer.is_ready(stream_2):
        sherpa.recognizer.decode_stream(stream_2)
    new_final = sherpa.recognizer.get_result(stream_2).strip()
    print(f"New final transcript (WITH input_finished):    '{new_final}'")
    sherpa.close_stream(session_id_2)

if __name__ == "__main__":
    test_incremental_streaming()
