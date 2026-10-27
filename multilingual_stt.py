
import sys
import time
from faster_whisper import WhisperModel

MODEL_NAME = "base"  # Fast multilingual model
DEVICE = "cuda"
COMPUTE_TYPE = "float16"

if len(sys.argv) < 2:
    print("Usage: python multilingual_stt.py audio.wav [language_code]")
    print("Example: python multilingual_stt.py speech1.wav hi")
    print("Omit language_code to detect the language automatically.")
    sys.exit(1)

audio_path = sys.argv[1]
language = sys.argv[2] if len(sys.argv) > 2 else None

print(f"Loading {MODEL_NAME} on GPU...")
load_start = time.perf_counter()
model = WhisperModel(
    MODEL_NAME,
    device=DEVICE,
    compute_type=COMPUTE_TYPE
)
print(f"Model load time: {time.perf_counter() - load_start:.3f}s")

start = time.perf_counter()
segments, info = model.transcribe(
    audio_path,
    language=language,
    task="transcribe",
    beam_size=1,
    vad_filter=False,
    condition_on_previous_text=False
)

# Consume all segments so the timing includes actual transcription.
text_parts = [segment.text.strip() for segment in segments]
elapsed = time.perf_counter() - start
transcript = " ".join(text_parts)

print("\n--- RESULTS ---")
print("Detected language:", info.language)
print(f"Language probability: {info.language_probability:.3f}")
print(f"Transcription time: {elapsed:.3f}s")
print("Transcript:")
print(transcript)
