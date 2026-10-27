import sys
import time
import wave
from pathlib import Path
from faster_whisper import WhisperModel

if len(sys.argv) != 2:
    sys.exit(f"Usage: python {sys.argv[0]} /path/to/sample.wav")

audio = Path(sys.argv[1]).expanduser().resolve()
if not audio.is_file():
    sys.exit(f"File not found: {audio}")

try:
    with wave.open(str(audio), "rb") as wav:
        duration = wav.getnframes() / wav.getframerate()
        print(f"Duration: {duration:.2f}s")
        print(f"Sample rate: {wav.getframerate()} Hz")
        print(f"Channels: {wav.getnchannels()}")
except Exception as e:
    sys.exit(f"Cannot read WAV metadata: {e}")

print("\nLoading base.en (GPU, FP16)...")
start = time.perf_counter()
model = WhisperModel("base.en", device="cuda", compute_type="float16")
load_time = time.perf_counter() - start
print(f"Model initialization: {load_time:.3f}s")

def transcribe():
    start = time.perf_counter()
    segments, info = model.transcribe(
        str(audio),
        language="en",
        beam_size=1,
        vad_filter=False,
        condition_on_previous_text=False,
    )
    segments = list(segments)
    elapsed = time.perf_counter() - start
    text = " ".join(s.text.strip() for s in segments)
    return elapsed, text

print("\nFirst transcription...")
first_time, transcript = transcribe()
print(f"First run: {first_time:.3f}s")
print(f"Real-time factor: {first_time / duration:.3f}")
print("\nTranscript:")
print(transcript)

print("\nThree warm runs...")
times = []
for i in range(1, 4):
    elapsed, _ = transcribe()
    times.append(elapsed)
    print(f"Run {i}: {elapsed:.3f}s | RTF: {elapsed / duration:.3f}")

average = sum(times) / len(times)
print("\n========== RESULTS ==========")
print(f"Audio duration:       {duration:.3f}s")
print(f"Model initialization: {load_time:.3f}s")
print(f"First transcription:  {first_time:.3f}s")
print(f"Warm average:         {average:.3f}s")
print(f"Warm RTF:             {average / duration:.3f}")
print(f"Warm speed:           {duration / average:.2f}x real time")
print("============================")