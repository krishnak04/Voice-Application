"""
Automatic English Accent Detection Service.

Acoustic English Accent / Variety Identification using ECAPA-TDNN trained on
Mozilla Common Voice / CommonAccent (MIT License).
Candidate classes:
- Indian English
- American English
- British English
- Australian English
- Canadian English
- New Zealand English
- Irish English
- South African English
- Other English variety
- Unknown / insufficient audio

Audio is classified directly from speech acoustics (Fbank + ECAPA-TDNN embeddings).
Strictly decoupled from STT (Whisper/Sherpa) and LLM (Groq).
"""

import os
import io
import time
import logging
import asyncio
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import numpy as np

logger = logging.getLogger("accent_service")

# Candidate Variety Codes and Display Names
VARIETY_MAPPING = {
    "indian": {
        "variety": "Indian English",
        "code": "en-IN",
        "flag": "🇮🇳",
        "tts_voice": "en-IN-NeerjaNeural"
    },
    "us": {
        "variety": "American English",
        "code": "en-US",
        "flag": "🇺🇸",
        "tts_voice": "en-US-JennyNeural"
    },
    "england": {
        "variety": "British English",
        "code": "en-GB",
        "flag": "🇬🇧",
        "tts_voice": "en-GB-SoniaNeural"
    },
    "scotland": {
        "variety": "British English",
        "code": "en-GB",
        "flag": "🇬🇧",
        "tts_voice": "en-GB-SoniaNeural"
    },
    "wales": {
        "variety": "British English",
        "code": "en-GB",
        "flag": "🇬🇧",
        "tts_voice": "en-GB-SoniaNeural"
    },
    "australia": {
        "variety": "Australian English",
        "code": "en-AU",
        "flag": "🇦🇺",
        "tts_voice": "en-AU-NatashaNeural"
    },
    "canada": {
        "variety": "Canadian English",
        "code": "en-CA",
        "flag": "🇨🇦",
        "tts_voice": "en-CA-ClaraNeural"
    },
    "newzealand": {
        "variety": "New Zealand English",
        "code": "en-NZ",
        "flag": "🇳🇿",
        "tts_voice": "en-NZ-MitchellNeural"
    },
    "ireland": {
        "variety": "Irish English",
        "code": "en-IE",
        "flag": "🇮🇪",
        "tts_voice": "en-IE-EmilyNeural"
    },
    "african": {
        "variety": "South African English",
        "code": "en-ZA",
        "flag": "🇿🇦",
        "tts_voice": "en-ZA-LeahNeural"
    },
    "southatlandtic": {
        "variety": "South African English",
        "code": "en-ZA",
        "flag": "🇿🇦",
        "tts_voice": "en-ZA-LeahNeural"
    },
    # Other English varieties
    "malaysia": {
        "variety": "Other English variety",
        "code": "en-OTHER",
        "flag": "🌐",
        "tts_voice": "en-US-JennyNeural"
    },
    "singapore": {
        "variety": "Other English variety",
        "code": "en-OTHER",
        "flag": "🌐",
        "tts_voice": "en-US-JennyNeural"
    },
    "philippines": {
        "variety": "Other English variety",
        "code": "en-OTHER",
        "flag": "🌐",
        "tts_voice": "en-US-JennyNeural"
    },
    "hongkong": {
        "variety": "Other English variety",
        "code": "en-OTHER",
        "flag": "🌐",
        "tts_voice": "en-US-JennyNeural"
    },
    "bermuda": {
        "variety": "Other English variety",
        "code": "en-OTHER",
        "flag": "🌐",
        "tts_voice": "en-US-JennyNeural"
    }
}

MIN_AUDIO_DURATION_SEC = 1.5  # Need at least 1.5s speech audio for reliable acoustic representation
CONFIDENCE_THRESHOLD = 0.28   # 16-class uniform random baseline is 0.0625 (6.25%)


class AccentSessionTracker:
    """Tracks audio and smooths accent predictions across multiple turns in a session."""
    def __init__(self, session_id: str, max_buffer_duration_sec: float = 12.0):
        self.session_id = session_id
        self.audio_samples: List[np.ndarray] = []
        self.total_samples = 0
        self.max_samples = int(max_buffer_duration_sec * 16000)
        self.history: List[Dict[str, Any]] = []
        self.current_accent: Optional[str] = None
        self.current_code: Optional[str] = None
        self.current_confidence: float = 0.0
        self.current_status: str = "detecting"

    def append_audio(self, pcm_16k_mono: np.ndarray):
        """Append float32 audio samples at 16kHz to the session buffer."""
        if pcm_16k_mono is None or len(pcm_16k_mono) == 0:
            return
        self.audio_samples.append(pcm_16k_mono)
        self.total_samples += len(pcm_16k_mono)
        # Prune older chunks if buffer exceeds max_samples
        while self.total_samples > self.max_samples and len(self.audio_samples) > 1:
            removed = self.audio_samples.pop(0)
            self.total_samples -= len(removed)

    def get_accumulated_audio(self) -> Optional[np.ndarray]:
        """Returns the accumulated audio as a single 1D numpy float32 array."""
        if not self.audio_samples:
            return None
        return np.concatenate(self.audio_samples)

    def update_prediction(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply temporal smoothing (hysteresis) to prevent rapid flipping between turns.
        """
        self.history.append(prediction)
        status = prediction.get("status")

        if status != "detected":
            # If not enough confidence or audio, preserve current stable accent if we had one
            if self.current_accent:
                return {
                    "accent": self.current_accent,
                    "code": self.current_code,
                    "flag": prediction.get("flag", "🌐"),
                    "confidence": self.current_confidence,
                    "status": "detected",
                    "sample_duration": prediction.get("sample_duration", 0.0),
                    "model": prediction.get("model"),
                    "device": prediction.get("device"),
                    "latency_ms": prediction.get("latency_ms", 0.0),
                    "tts_voice": prediction.get("tts_voice"),
                    "is_smoothed": True
                }
            return prediction

        new_accent = prediction["accent"]
        new_conf = prediction["confidence"]

        if self.current_accent is None:
            # First solid detection
            self.current_accent = new_accent
            self.current_code = prediction["code"]
            self.current_confidence = new_conf
            self.current_status = "detected"
        elif self.current_accent == new_accent:
            # Reinforce existing prediction with running exponential average
            self.current_confidence = round(0.7 * self.current_confidence + 0.3 * new_conf, 3)
        else:
            # Different accent proposed: check recent history before switching
            recent_same = [h for h in self.history[-3:] if h.get("accent") == new_accent and h.get("status") == "detected"]
            # Switch only if at least 2 consecutive turns suggest new accent or new confidence is significantly higher
            if len(recent_same) >= 2 or new_conf > (self.current_confidence + 0.25):
                self.current_accent = new_accent
                self.current_code = prediction["code"]
                self.current_confidence = new_conf
            else:
                # Keep current accent to avoid jitter
                prediction["accent"] = self.current_accent
                prediction["code"] = self.current_code
                prediction["is_smoothed"] = True

        return prediction


class AccentDetectionService:
    """
    Automatic Audio-based English Accent Detector using ECAPA-TDNN.
    Decoupled from STT and LLM.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(AccentDetectionService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_dir: Optional[str] = None, device: str = "cpu"):
        if getattr(self, "_initialized", False):
            return

        self.device = device
        self.model_name = "Jzuluaga/accent-id-commonaccent_ecapa"
        self.model_license = "MIT"
        self.classifier = None
        self._load_lock = asyncio.Lock()
        self.sessions: Dict[str, AccentSessionTracker] = {}

        # Resolve local cache directory
        if model_dir:
            self.savedir = Path(model_dir)
        else:
            self.savedir = Path(__file__).parent / "tmp_accent_model"
        self.savedir.mkdir(parents=True, exist_ok=True)

        self._initialize_classifier()
        self._initialized = True

    def _initialize_classifier(self):
        """Load the SpeechBrain ECAPA-TDNN accent classifier."""
        try:
            from speechbrain.inference.classifiers import EncoderClassifier
            from speechbrain.utils.fetching import LocalStrategy

            logger.info(f"Loading Accent Classifier '{self.model_name}' on {self.device}...")
            # Check if model checkpoints already downloaded in savedir
            required_files = ["hyperparams.yaml", "classifier.ckpt", "embedding_model.ckpt"]
            has_local = all((self.savedir / f).exists() for f in required_files)

            if has_local:
                logger.info(f"Found local accent model files in {self.savedir}")
                self.classifier = EncoderClassifier.from_hparams(
                    source=str(self.savedir),
                    run_opts={"device": self.device}
                )
            else:
                logger.info(f"Downloading accent model from HuggingFace to {self.savedir}...")
                self.classifier = EncoderClassifier.from_hparams(
                    source=self.model_name,
                    savedir=str(self.savedir),
                    run_opts={"device": self.device},
                    local_strategy=LocalStrategy.COPY
                )

            logger.info("Accent Classifier successfully initialized!")
        except Exception as e:
            logger.error(f"Failed to load Accent Classifier: {e}", exc_info=True)
            self.classifier = None

    def is_available(self) -> bool:
        return self.classifier is not None

    def get_session(self, session_id: str) -> AccentSessionTracker:
        """Get or create session tracker for a WebSocket / caller session."""
        if session_id not in self.sessions:
            self.sessions[session_id] = AccentSessionTracker(session_id)
        return self.sessions[session_id]

    def clear_session(self, session_id: str):
        if session_id in self.sessions:
            del self.sessions[session_id]

    def _extract_pcm16k(self, audio_data: Any) -> Tuple[Optional[np.ndarray], float]:
        """
        Convert various audio input formats (filepath, bytes, numpy array, wav) into 16kHz float32 mono array.
        Returns: (samples, duration_in_seconds)
        """
        try:
            import soundfile as sf

            if isinstance(audio_data, (str, Path)):
                filepath = str(audio_data)
                if os.path.isfile(filepath):
                    data, sample_rate = sf.read(filepath)
                    if len(data.shape) > 1:
                        data = data.mean(axis=1)
                    data = data.astype(np.float32)
                    if sample_rate != 16000:
                        num_target_samples = int(len(data) * 16000 / sample_rate)
                        indices = np.linspace(0, len(data) - 1, num_target_samples)
                        data = np.interp(indices, np.arange(len(data)), data).astype(np.float32)
                    duration = len(data) / 16000.0
                    return data, duration

            if isinstance(audio_data, np.ndarray):
                if audio_data.dtype in (np.int16, np.int32):
                    audio_float = audio_data.astype(np.float32) / 32768.0
                else:
                    audio_float = audio_data.astype(np.float32)
                duration = len(audio_float) / 16000.0
                return audio_float, duration

            if isinstance(audio_data, (bytes, bytearray)):
                try:
                    with io.BytesIO(audio_data) as bio:
                        data, sample_rate = sf.read(bio)
                        if len(data.shape) > 1:
                            data = data.mean(axis=1)  # convert stereo to mono
                        data = data.astype(np.float32)
                        if sample_rate != 16000:
                            num_target_samples = int(len(data) * 16000 / sample_rate)
                            indices = np.linspace(0, len(data) - 1, num_target_samples)
                            data = np.interp(indices, np.arange(len(data)), data).astype(np.float32)
                        duration = len(data) / 16000.0
                        return data, duration
                except Exception:
                    # Raw PCM16 16kHz bytes
                    pcm_int16 = np.frombuffer(audio_data, dtype=np.int16)
                    audio_float = pcm_int16.astype(np.float32) / 32768.0
                    duration = len(audio_float) / 16000.0
                    return audio_float, duration

        except Exception as e:
            logger.warning(f"Error converting audio in accent detector: {e}")
            return None, 0.0

        return None, 0.0

    def detect_accent_sync(
        self,
        audio_data: Any,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Synchronous audio-based English accent detection.
        Decoupled from STT.
        """
        t0 = time.perf_counter()

        samples, duration = self._extract_pcm16k(audio_data)

        # If session_id provided, accumulate in session buffer for multi-turn smoothing
        session = self.get_session(session_id) if session_id else None
        if session and samples is not None:
            session.append_audio(samples)
            # Use accumulated audio if current turn is short
            if duration < MIN_AUDIO_DURATION_SEC:
                accum_samples = session.get_accumulated_audio()
                if accum_samples is not None and len(accum_samples) > len(samples):
                    samples = accum_samples
                    duration = len(samples) / 16000.0

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        # 1. Check for insufficient audio
        if samples is None or duration < MIN_AUDIO_DURATION_SEC:
            result = {
                "accent": "Unknown / insufficient audio",
                "code": "unknown",
                "flag": "❓",
                "confidence": 0.0,
                "status": "insufficient_audio",
                "sample_duration": round(duration, 2),
                "model": self.model_name,
                "license": self.model_license,
                "device": self.device,
                "latency_ms": latency_ms,
                "tts_voice": "en-US-JennyNeural",
                "details": f"Audio length ({duration:.2f}s) is under the {MIN_AUDIO_DURATION_SEC}s minimum requirement for acoustic accent estimation."
            }
            if session:
                return session.update_prediction(result)
            return result

        # 2. Check if classifier is loaded
        if not self.classifier:
            return {
                "accent": "Unknown / model unavailable",
                "code": "unknown",
                "flag": "⚠️",
                "confidence": 0.0,
                "status": "error",
                "sample_duration": round(duration, 2),
                "model": self.model_name,
                "license": self.model_license,
                "device": self.device,
                "latency_ms": latency_ms,
                "tts_voice": "en-US-JennyNeural",
                "details": "Accent classifier is not initialized on the server."
            }

        # 3. Run Inference on audio tensor
        try:
            import torch

            # Truncate to maximum 10 seconds for bounded inference latency (<350ms)
            max_samples = 16000 * 10
            if len(samples) > max_samples:
                samples = samples[:max_samples]

            sig_tensor = torch.from_numpy(samples).unsqueeze(0).float()
            if self.device != "cpu":
                sig_tensor = sig_tensor.to(self.device)

            inf_start = time.perf_counter()
            with torch.no_grad():
                out = self.classifier.classify_batch(sig_tensor)
                # out is (scores, best_score, index, text_lab)
                scores = out[0].squeeze().cpu().numpy()
                best_score = float(out[1].squeeze().cpu().item())
                best_index = int(out[2].squeeze().cpu().item())
                predicted_label = str(out[3][0]).strip().lower()

            inf_latency_ms = round((time.perf_counter() - inf_start) * 1000, 2)
            total_latency_ms = round((time.perf_counter() - t0) * 1000, 2)

            # Map to candidate classes
            variety_info = VARIETY_MAPPING.get(predicted_label, {
                "variety": "Other English variety",
                "code": "en-OTHER",
                "flag": "🌐",
                "tts_voice": "en-US-JennyNeural"
            })

            # Check confidence threshold
            if best_score < CONFIDENCE_THRESHOLD:
                result = {
                    "accent": "Unknown / uncertain",
                    "code": "uncertain",
                    "flag": "❓",
                    "confidence": round(best_score, 3),
                    "status": "uncertain",
                    "candidate_variety": variety_info["variety"],
                    "raw_label": predicted_label,
                    "sample_duration": round(duration, 2),
                    "model": self.model_name,
                    "license": self.model_license,
                    "device": self.device,
                    "latency_ms": total_latency_ms,
                    "inference_latency_ms": inf_latency_ms,
                    "tts_voice": variety_info["tts_voice"],
                    "details": f"Confidence score ({best_score:.3f}) below threshold ({CONFIDENCE_THRESHOLD})"
                }
            else:
                result = {
                    "accent": variety_info["variety"],
                    "code": variety_info["code"],
                    "flag": variety_info["flag"],
                    "confidence": round(best_score, 3),
                    "status": "detected",
                    "raw_label": predicted_label,
                    "sample_duration": round(duration, 2),
                    "model": self.model_name,
                    "license": self.model_license,
                    "device": self.device,
                    "latency_ms": total_latency_ms,
                    "inference_latency_ms": inf_latency_ms,
                    "tts_voice": variety_info["tts_voice"],
                    "details": f"Successfully identified {variety_info['variety']} with {int(best_score * 100)}% acoustic confidence."
                }

            if session:
                return session.update_prediction(result)
            return result

        except Exception as e:
            logger.error(f"Inference error during accent detection: {e}", exc_info=True)
            return {
                "accent": "Unknown / classification error",
                "code": "error",
                "flag": "⚠️",
                "confidence": 0.0,
                "status": "error",
                "sample_duration": round(duration, 2),
                "model": self.model_name,
                "license": self.model_license,
                "device": self.device,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
                "tts_voice": "en-US-JennyNeural",
                "details": str(e)
            }

    async def detect_accent_async(
        self,
        audio_data: Any,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Asynchronous wrapper that runs accent detection in a threadpool without blocking the event loop."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.detect_accent_sync, audio_data, session_id)


# Singleton getter
_accent_instance: Optional[AccentDetectionService] = None

def get_accent_detector() -> AccentDetectionService:
    global _accent_instance
    if _accent_instance is None:
        _accent_instance = AccentDetectionService(device="cpu")
    return _accent_instance
