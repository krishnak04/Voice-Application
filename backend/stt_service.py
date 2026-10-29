import os
import io
import time
import tempfile
import logging
import ctypes
import numpy as np
from faster_whisper import WhisperModel
import config

# Patch PyAV if installed to prevent 'metadata_errors' TypeError in newer PyAV versions
try:
    import av
    _orig_av_open = av.open
    def _safe_av_open(*args, **kwargs):
        kwargs.pop("metadata_errors", None)
        return _orig_av_open(*args, **kwargs)
    av.open = _safe_av_open
except Exception:
    pass

logger = logging.getLogger("stt_service")

def _preload_cuda_libraries():
    """Attempt to preload pip-installed nvidia CUDA 12 libraries on Linux."""
    if os.name != "posix":
        return

    # 1. Preload nvJitLink (required by cuBLAS 12.9+)
    try:
        import nvidia.nvjitlink.lib
        jit_dir = os.path.dirname(nvidia.nvjitlink.lib.__file__)
        for fname in sorted(os.listdir(jit_dir)):
            if fname.startswith("libnvJitLink") and ".so" in fname:
                try:
                    ctypes.CDLL(os.path.join(jit_dir, fname), mode=ctypes.RTLD_GLOBAL)
                except Exception:
                    pass
    except Exception:
        pass

    # 2. Preload cuBLAS Lt and cuBLAS
    try:
        import nvidia.cublas.lib
        cublas_dir = os.path.dirname(nvidia.cublas.lib.__file__)
        for lib in ["libcublasLt.so.12", "libcublas.so.12"]:
            path = os.path.join(cublas_dir, lib)
            if os.path.exists(path):
                try:
                    ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                    logger.debug(f"Preloaded {lib}")
                except Exception as e:
                    logger.debug(f"Could not preload {lib}: {e}")
    except Exception:
        pass

    # 3. Preload cuDNN
    try:
        import nvidia.cudnn.lib
        cudnn_dir = os.path.dirname(nvidia.cudnn.lib.__file__)
        for fname in sorted(os.listdir(cudnn_dir)):
            if fname.startswith("libcudnn") and ".so" in fname:
                path = os.path.join(cudnn_dir, fname)
                try:
                    ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                except Exception:
                    pass
    except Exception:
        pass

from typing import Optional

# 22 Official Indian Languages Mapping (Number, Name, Code)
INDIAN_LANGUAGES_MAP = {
    "1": "hi", "hindi": "hi", "hi": "hi",
    "2": "bn", "bengali": "bn", "bn": "bn",
    "3": "ta", "tamil": "ta", "ta": "ta",
    "4": "te", "telugu": "te", "te": "te",
    "5": "mr", "marathi": "mr", "mr": "mr",
    "6": "gu", "gujarati": "gu", "gu": "gu",
    "7": "kn", "kannada": "kn", "kn": "kn",
    "8": "ml", "malayalam": "ml", "ml": "ml",
    "9": "pa", "punjabi": "pa", "pa": "pa",
    "10": "ur", "urdu": "ur", "ur": "ur",
    "11": "as", "assamese": "as", "as": "as",
    "12": "or", "odia": "or", "oriya": "or",
    "13": "ne", "nepali": "ne", "ne": "ne",
    "14": "sa", "sanskrit": "sa", "sa": "sa",
    "15": "sd", "sindhi": "sd", "sd": "sd",
    "16": "ks", "kashmiri": "ks", "ks": "ks",
    "17": "kok", "konkani": "kok",
    "18": "mni", "manipuri": "mni", "meitei": "mni",
    "19": "brx", "bodo": "brx",
    "20": "doi", "dogri": "doi",
    "21": "sat", "santali": "sat",
    "22": "mai", "maithili": "mai",
}

WHISPER_MODELS_CATALOG = {
    "tiny": {
        "id": "tiny",
        "name": "Whisper Tiny",
        "description": "Ultra-fast ~39M parameters, lowest memory footprint (~75MB VRAM / RAM).",
        "params": "~39M",
        "languages": "multilingual",
        "type": "offline",
        "category": "Small",
        "is_english_only": False,
        "is_supported": True,
    },
    "base": {
        "id": "base",
        "name": "Whisper Base",
        "description": "Standard fast baseline ~74M parameters. Balanced speed and multi-lingual accuracy.",
        "params": "~74M",
        "languages": "multilingual",
        "type": "offline",
        "category": "Standard",
        "is_english_only": False,
        "is_supported": True,
    },
    "small": {
        "id": "small",
        "name": "Whisper Small",
        "description": "High accuracy ~244M parameters for complex acoustic environments.",
        "params": "~244M",
        "languages": "multilingual",
        "type": "offline",
        "category": "Large",
        "is_english_only": False,
        "is_supported": True,
    },
    "medium": {
        "id": "medium",
        "name": "Whisper Medium",
        "description": "Production ~769M parameters for high-fidelity multi-lingual and Indic recognition.",
        "params": "~769M",
        "languages": "multilingual",
        "type": "offline",
        "category": "Large",
        "is_english_only": False,
        "is_supported": True,
    },
    "large-v3-turbo": {
        "id": "large-v3-turbo",
        "name": "Whisper Large-v3-Turbo",
        "description": "State-of-the-art ~809M parameters with optimized decoder depth (4 layers) for rapid inference.",
        "params": "~809M",
        "languages": "multilingual",
        "type": "offline",
        "category": "Large",
        "is_english_only": False,
        "is_supported": True,
    },
    "base.en": {
        "id": "base.en",
        "name": "Whisper Base.en",
        "description": "English-specialized ~74M parameters with enhanced phonetics for US/UK/IN accents.",
        "params": "~74M",
        "languages": "en",
        "type": "offline",
        "category": "Standard",
        "is_english_only": True,
        "is_supported": True,
    },
    "small.en": {
        "id": "small.en",
        "name": "Whisper Small.en",
        "description": "English-specialized ~244M parameters for superior accuracy across diverse English accents.",
        "params": "~244M",
        "languages": "en",
        "type": "offline",
        "category": "Large",
        "is_english_only": True,
        "is_supported": True,
    },
}

class STTEngine:
    def __init__(self):
        self.device = config.WHISPER_DEVICE
        self.compute_type = config.WHISPER_COMPUTE_TYPE
        self.default_model_name = config.WHISPER_MODEL
        self.models: dict[str, WhisperModel] = {}
        # Preload the default model on startup
        self.get_model(self.default_model_name)

    def _fallback_to_cpu(self, model_name: str, reason: str) -> WhisperModel:
        logger.warning(f"CUDA execution failed for {model_name} ({reason}). Falling back to CPU/int8...")
        self.device = "cpu"
        self.compute_type = "int8"
        cpu_model = WhisperModel(
            model_name,
            device="cpu",
            compute_type="int8"
        )
        self.models[model_name] = cpu_model
        logger.info(f"Loaded fallback model {model_name} on CPU (int8)")
        return cpu_model

    def get_model(self, model_name: Optional[str] = None) -> tuple[WhisperModel, str]:
        """Get or dynamically load and cache a Whisper model by name."""
        name = (model_name or self.default_model_name).strip()
        if not name:
            name = self.default_model_name

        if name in self.models:
            return self.models[name], name

        _preload_cuda_libraries()
        start = time.perf_counter()
        logger.info(f"Loading Whisper model '{name}' on {self.device} ({self.compute_type})...")
        try:
            kwargs = {
                "device": self.device,
                "compute_type": self.compute_type
            }
            if self.device == "cpu":
                kwargs["cpu_threads"] = getattr(config, "WHISPER_CPU_THREADS", 6)

            model = WhisperModel(name, **kwargs)
            if self.device.startswith("cuda"):
                # Warm-up verification
                dummy = np.zeros(1600, dtype=np.float32)
                _ = list(model.transcribe(dummy, language="en")[0])
                logger.info(f"CUDA verification & warm-up successful for {name}.")
            self.models[name] = model
        except Exception as e:
            model = self._fallback_to_cpu(name, str(e))

        load_time = time.perf_counter() - start
        logger.info(f"Model '{name}' ready in {load_time:.3f}s on {self.device} ({self.compute_type})")
        return model, name

    def get_loaded_models(self) -> list[str]:
        return list(self.models.keys())

    def list_models(self) -> list[dict]:
        """List all verified Whisper models with loaded and active status."""
        result = []
        for model_id, meta in WHISPER_MODELS_CATALOG.items():
            is_loaded = model_id in self.models
            is_active = (model_id == self.default_model_name)
            status = "Ready" if is_loaded else "Installed"
            result.append({
                "id": model_id,
                "name": meta["name"],
                "description": meta["description"],
                "params": meta["params"],
                "languages": meta["languages"],
                "type": meta["type"],  # 'offline'
                "category": meta["category"],  # 'Small' | 'Standard' | 'Large'
                "is_english_only": meta["is_english_only"],
                "is_downloaded": True,
                "downloaded": True,
                "is_loaded": is_loaded,
                "status": status,
                "is_active": is_active,
                "device": self.device,
                "compute_type": self.compute_type
            })
        return result

    def select_model(self, model_name: str) -> dict:
        """Safely switch and preload the active Whisper model."""
        model_id = model_name.strip()
        if model_id not in WHISPER_MODELS_CATALOG:
            return {
                "success": False,
                "message": f"Model '{model_id}' is not in the verified Whisper catalog."
            }
        try:
            model, name = self.get_model(model_id)
            self.default_model_name = name
            return {
                "success": True,
                "message": f"Switched active Whisper model to '{name}'.",
                "active_model": name,
                "device": self.device,
                "compute_type": self.compute_type
            }
        except Exception as e:
            logger.error(f"Failed to load Whisper model '{model_id}': {e}")
            return {
                "success": False,
                "message": f"Failed to load Whisper model '{model_id}': {str(e)}"
            }

    def transcribe_audio_bytes(
        self,
        audio_bytes: bytes,
        file_ext: str = ".webm",
        language: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        if len(audio_bytes) == 0:
            return {"transcript": "", "language": language or "en", "duration": 0.0, "stt_latency": 0.0, "rtf": 0.0, "model_name": model_name or self.default_model_name}

        # Handle raw PCM16 audio bytes directly without requiring container header
        is_container = (
            audio_bytes.startswith(b"RIFF")
            or audio_bytes.startswith(b"\x1a\x45\xdf\xa3")
            or audio_bytes.startswith(b"ID3")
            or audio_bytes.startswith(b"\xff\xfb")
            or audio_bytes.startswith(b"OggS")
            or audio_bytes.startswith(b"fLaC")
        )
        if not is_container and (len(audio_bytes) % 2 == 0):
            pcm_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            return self.transcribe_file(pcm_np, language=language, model_name=model_name)

        # 1. High-speed in-memory transcription (avoids disk write/read latency)
        try:
            bio = io.BytesIO(audio_bytes)
            return self.transcribe_file(bio, language=language, model_name=model_name)
        except Exception as in_mem_err:
            logger.debug(f"In-memory transcription fallback to tempfile: {in_mem_err}")

        # 2. Resilient temporary file fallback
        with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            return self.transcribe_file(tmp_path, language=language, model_name=model_name)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    def transcribe_file(
        self,
        audio_source,
        language: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        # Normalize language: None or "auto" triggers Whisper's automatic language detection
        lang_param = None
        if language and language.strip().lower() not in ("auto", "none", "0", ""):
            clean_lang = language.strip().lower()
            lang_param = INDIAN_LANGUAGES_MAP.get(clean_lang, clean_lang)

        # Resolve requested model
        req_model = model_name or self.default_model_name
        if req_model == "sherpa-streaming" or (req_model and req_model.startswith("sherpa-")):
            try:
                from sherpa_service import get_sherpa_engine
                sherpa = get_sherpa_engine()
                if req_model.startswith("sherpa-") and req_model != sherpa.active_model_id:
                    sherpa.switch_model(req_model)
                if sherpa.is_available():
                    return sherpa.transcribe_audio_source(audio_source)
            except Exception as e:
                logger.warning(f"Sherpa file transcription failed ({e}). Falling back to {self.default_model_name}...")
            req_model = self.default_model_name

        # If an English-only model (e.g. base.en) is active but a non-English language is requested,
        # automatically adapt to the multilingual counterpart (e.g. base)
        if req_model.endswith(".en") and lang_param and lang_param != "en":
            multilingual_alt = req_model[:-3]  # strip .en -> base, small, etc.
            logger.info(f"Language '{lang_param}' requested with English-only model '{req_model}'. Switching to multilingual '{multilingual_alt}'.")
            req_model = multilingual_alt

        if req_model.endswith(".en") and not lang_param:
            lang_param = "en"

        model, active_model_name = self.get_model(req_model)

        # 1. Decode audio source to 16kHz mono float32 with telemetry
        start_time = time.perf_counter()
        samples_np = None
        orig_rate = 16000
        orig_channels = 1

        if isinstance(audio_source, np.ndarray):
            samples_np = audio_source.astype(np.float32)
        else:
            if isinstance(audio_source, (bytes, bytearray)):
                is_container = (
                    audio_source.startswith(b"RIFF")
                    or audio_source.startswith(b"\x1a\x45\xdf\xa3")
                    or audio_source.startswith(b"ID3")
                    or audio_source.startswith(b"\xff\xfb")
                    or audio_source.startswith(b"OggS")
                    or audio_source.startswith(b"fLaC")
                )
                if not is_container and (len(audio_source) % 2 == 0):
                    samples_np = np.frombuffer(audio_source, dtype=np.int16).astype(np.float32) / 32768.0
                else:
                    audio_input = io.BytesIO(audio_source)
            elif isinstance(audio_source, io.BytesIO):
                raw = audio_source.getvalue()
                is_container = (
                    raw.startswith(b"RIFF")
                    or raw.startswith(b"\x1a\x45\xdf\xa3")
                    or raw.startswith(b"ID3")
                    or raw.startswith(b"\xff\xfb")
                    or raw.startswith(b"OggS")
                    or raw.startswith(b"fLaC")
                )
                if not is_container and (len(raw) % 2 == 0):
                    samples_np = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                else:
                    audio_input = audio_source
            elif isinstance(audio_source, str) and not os.path.exists(audio_source):
                raise FileNotFoundError(f"Audio file not found: {audio_source}")
            else:
                audio_input = audio_source

            if samples_np is None:
                import av
                try:
                    container = av.open(audio_input)
                    audio_stream = next(s for s in container.streams if s.type == 'audio')
                    orig_rate = getattr(audio_stream, 'rate', 16000)
                    orig_channels = getattr(audio_stream, 'channels', 1)
                    resampler = av.AudioResampler(format='fltp', layout='mono', rate=16000)
                    decoded_frames = []
                    for frame in container.decode(audio_stream):
                        for rf in resampler.resample(frame):
                            decoded_frames.append(rf.to_ndarray()[0])
                    for rf in resampler.resample(None):
                        decoded_frames.append(rf.to_ndarray()[0])
                    container.close()
                    if decoded_frames:
                        samples_np = np.concatenate(decoded_frames)
                    else:
                        samples_np = np.zeros(0, dtype=np.float32)
                except Exception as decode_err:
                    logger.debug(f"PyAV decode error ({decode_err}), using audio_source directly.")
                    samples_np = audio_source

        # Telemetry & Normalization if samples array was extracted
        if isinstance(samples_np, np.ndarray) and len(samples_np) > 0:
            audio_dur = len(samples_np) / 16000.0
            peak_val = float(np.max(np.abs(samples_np)))
            peak_dbfs = round(20 * float(np.log10(max(peak_val, 1e-5))), 1)
            rms_val = round(float(np.sqrt(np.mean(samples_np**2))), 4)
            logger.info(
                f"[Whisper STT Audio Telemetry] duration={audio_dur:.2f}s, in_rate={orig_rate}Hz, "
                f"in_channels={orig_channels}, peak={peak_dbfs} dBFS, rms={rms_val}, model={active_model_name}"
            )

            # Normalization: DC bias removal and natural headroom leveling
            mean_v = np.mean(samples_np)
            samples_np = samples_np - mean_v
            if peak_val > 0.98:
                samples_np = samples_np * (0.95 / peak_val)
            elif peak_val < 0.04 and peak_val > 1e-4:
                # Conservative boost only for faint signals, strictly capped to 2.0x
                samples_np = np.clip(samples_np * min(2.0, 0.15 / max(rms_val, 1e-4)), -1.0, 1.0)
            transcribe_target = samples_np
        else:
            transcribe_target = audio_source

        beam_size = getattr(config, "WHISPER_BEAM_SIZE", 3)
        vad_params = dict(
            threshold=0.35,
            min_speech_duration_ms=200,
            min_silence_duration_ms=400,
            speech_pad_ms=200
        )

        try:
            try:
                segments, info = model.transcribe(
                    transcribe_target,
                    language=lang_param,
                    beam_size=beam_size,
                    patience=1.0,
                    vad_filter=True,
                    vad_parameters=vad_params,
                    condition_on_previous_text=False
                )
                segment_list = list(segments)
            except ValueError as ve:
                logger.warning(f"Language code '{lang_param}' raised ValueError ({ve}). Retrying with auto-detect...")
                segments, info = model.transcribe(
                    transcribe_target,
                    language=None,
                    beam_size=beam_size,
                    patience=1.0,
                    vad_filter=True,
                    vad_parameters=vad_params,
                    condition_on_previous_text=False
                )
                segment_list = list(segments)

            # Fallback if VAD filter was overly aggressive on soft audible speech
            if not any(s.text.strip() for s in segment_list) and isinstance(samples_np, np.ndarray) and len(samples_np) > 4000:
                peak_check = float(np.max(np.abs(samples_np)))
                if peak_check > 0.015:
                    logger.info("VAD filter produced empty transcript on audible audio. Retrying without VAD filter...")
                    segments, info = model.transcribe(
                        transcribe_target,
                        language=lang_param,
                        beam_size=beam_size,
                        patience=1.0,
                        vad_filter=False,
                        condition_on_previous_text=False
                    )
                    segment_list = list(segments)
        except Exception as e:
            if self.device == "cuda":
                logger.error(f"CUDA transcription error with {active_model_name}: {e}. Switching to CPU fallback...")
                model = self._fallback_to_cpu(active_model_name, str(e))
                try:
                    segments, info = model.transcribe(
                        transcribe_target,
                        language=lang_param,
                        beam_size=beam_size,
                        patience=1.0,
                        vad_filter=True,
                        vad_parameters=vad_params,
                        condition_on_previous_text=False
                    )
                    segment_list = list(segments)
                except ValueError:
                    segments, info = model.transcribe(
                        transcribe_target,
                        language=None,
                        beam_size=beam_size,
                        patience=1.0,
                        vad_filter=True,
                        vad_parameters=vad_params,
                        condition_on_previous_text=False
                    )
                    segment_list = list(segments)
            else:
                raise e

        elapsed_time = time.perf_counter() - start_time
        transcript = " ".join(s.text.strip() for s in segment_list)

        duration = getattr(info, "duration", 0.0)
        if duration <= 0.0:
            if isinstance(samples_np, np.ndarray) and len(samples_np) > 0:
                duration = len(samples_np) / 16000.0
            else:
                duration = max(elapsed_time, 0.01)

        rtf = elapsed_time / duration if duration > 0 else 0.0
        speed_factor = duration / elapsed_time if elapsed_time > 0 else 0.0

        detected_lang = getattr(info, "language", lang_param or "en")
        lang_prob = getattr(info, "language_probability", 1.0)

        return {
            "transcript": transcript,
            "language": detected_lang,
            "language_probability": round(lang_prob, 3) if lang_prob is not None else 1.0,
            "duration": round(duration, 3),
            "stt_latency": round(elapsed_time, 3),
            "rtf": round(rtf, 3),
            "speed_factor": round(speed_factor, 2),
            "device": self.device,
            "model_name": active_model_name
        }

# Global Singleton instance
stt_engine = None

def get_stt_engine() -> STTEngine:
    global stt_engine
    if stt_engine is None:
        stt_engine = STTEngine()
    return stt_engine

