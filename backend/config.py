import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env if present in backend or root
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

# Groq Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
GROQ_MAX_HISTORY_MESSAGES = int(os.getenv("GROQ_MAX_HISTORY_MESSAGES", "50"))

# Hardware & Device Detection
def _detect_hardware():
    has_cuda = False
    cuda_count = 0
    try:
        import ctranslate2
        cuda_count = ctranslate2.get_cuda_device_count()
        has_cuda = cuda_count > 0
    except Exception:
        try:
            import torch
            has_cuda = torch.cuda.is_available()
            cuda_count = torch.cuda.device_count() if has_cuda else 0
        except Exception:
            has_cuda = False
            cuda_count = 0
    return has_cuda, cuda_count

HAS_CUDA, CUDA_COUNT = _detect_hardware()

# Faster Whisper Configuration
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
# Auto-detect optimal device if not explicitly set in env or if requested cuda is unavailable
_req_whisper_dev = os.getenv("WHISPER_DEVICE", "auto").lower()
if _req_whisper_dev == "auto":
    WHISPER_DEVICE = "cuda" if HAS_CUDA else "cpu"
elif "cuda" in _req_whisper_dev and not HAS_CUDA:
    WHISPER_DEVICE = "cpu"
else:
    WHISPER_DEVICE = _req_whisper_dev

_req_compute = os.getenv("WHISPER_COMPUTE_TYPE", "auto").lower()
if _req_compute == "auto":
    WHISPER_COMPUTE_TYPE = "float16" if WHISPER_DEVICE.startswith("cuda") else "int8"
else:
    WHISPER_COMPUTE_TYPE = _req_compute

WHISPER_CPU_THREADS = int(os.getenv("WHISPER_CPU_THREADS", str(min(8, max(2, (os.cpu_count() or 4) // 2)))))

# Text-to-Speech (TTS) Configuration
TTS_ENGINE = os.getenv("TTS_ENGINE", "auto")          # 'auto', 'kokoro', 'piper', 'edge_tts'
TTS_DEVICE = os.getenv("TTS_DEVICE", "cuda:1" if CUDA_COUNT >= 2 else ("cuda" if HAS_CUDA else "cpu"))
TTS_VOICE = os.getenv("TTS_VOICE", "en-IN-NeerjaNeural")        # Default voice ID
PIPER_BINARY = os.getenv("PIPER_BINARY", "piper")
PIPER_MODEL = os.getenv("PIPER_MODEL", "en_US-lessac-medium.onnx")

# Sherpa-ONNX Streaming ASR Configuration
ROOT_DIR = Path(__file__).resolve().parent.parent
SHERPA_MODEL_DIR = os.getenv("SHERPA_MODEL_DIR", str(ROOT_DIR / "sherpa-onnx-streaming-zipformer-en-2023-06-26"))
SHERPA_USE_INT8 = os.getenv("SHERPA_USE_INT8", "true").lower() in ("true", "1", "yes")
SHERPA_NUM_THREADS = int(os.getenv("SHERPA_NUM_THREADS", str(min(8, max(2, (os.cpu_count() or 4) // 2)))))
SHERPA_RULE1_MIN_TRAILING_SILENCE = float(os.getenv("SHERPA_RULE1_MIN_TRAILING_SILENCE", "2.0"))
SHERPA_RULE2_MIN_TRAILING_SILENCE = float(os.getenv("SHERPA_RULE2_MIN_TRAILING_SILENCE", "0.8"))
SHERPA_RULE3_MIN_UTTERANCE_LENGTH = float(os.getenv("SHERPA_RULE3_MIN_UTTERANCE_LENGTH", "20.0"))
SHERPA_PROVIDER = os.getenv("SHERPA_PROVIDER", "cpu")

# Server Configuration
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8024))
CORS_ORIGINS = [
    "http://localhost:8023",
    "http://127.0.0.1:8023",
    "http://61.2.142.91:8023",
    "*"
]


