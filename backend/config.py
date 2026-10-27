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

# Faster Whisper Configuration
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cuda")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "float16")

# Text-to-Speech (TTS) Configuration
TTS_ENGINE = os.getenv("TTS_ENGINE", "auto")          # 'auto', 'kokoro', 'piper', 'edge_tts'
TTS_DEVICE = os.getenv("TTS_DEVICE", "cuda")          # 'cuda' (Kokoro uses ~300MB VRAM) or 'cpu' (0 VRAM)
TTS_VOICE = os.getenv("TTS_VOICE", "af_heart")        # Default voice ID
PIPER_BINARY = os.getenv("PIPER_BINARY", "piper")
PIPER_MODEL = os.getenv("PIPER_MODEL", "en_US-lessac-medium.onnx")

# Sherpa-ONNX Streaming ASR Configuration
ROOT_DIR = Path(__file__).resolve().parent.parent
SHERPA_MODEL_DIR = os.getenv("SHERPA_MODEL_DIR", str(ROOT_DIR / "sherpa-onnx-streaming-zipformer-en-2023-06-26"))
SHERPA_USE_INT8 = os.getenv("SHERPA_USE_INT8", "true").lower() in ("true", "1", "yes")
SHERPA_NUM_THREADS = int(os.getenv("SHERPA_NUM_THREADS", "2"))
SHERPA_RULE1_MIN_TRAILING_SILENCE = float(os.getenv("SHERPA_RULE1_MIN_TRAILING_SILENCE", "2.4"))
SHERPA_RULE2_MIN_TRAILING_SILENCE = float(os.getenv("SHERPA_RULE2_MIN_TRAILING_SILENCE", "1.0"))
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


