import os
import io
import gc
import time
import subprocess
import logging
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np
import config
from sherpa_models_catalog import OFFICIAL_SHERPA_ENGLISH_MODELS, find_model_files

logger = logging.getLogger("sherpa_service")

class SherpaStreamingEngine:
    """
    High-performance speech recognition engine powered by Sherpa-ONNX.
    Supports official Small, Standard, and Large English models, with clean separation
    between causal Streaming Zipformer transducers and bidirectional Offline models.
    """
    def __init__(self):
        self.root_dir = Path(config.ROOT_DIR).resolve()
        self.models_dir = (self.root_dir / "sherpa_models").resolve()
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.use_int8 = config.SHERPA_USE_INT8
        self.num_threads = config.SHERPA_NUM_THREADS
        self.provider = config.SHERPA_PROVIDER
        self.rule1_silence = config.SHERPA_RULE1_MIN_TRAILING_SILENCE
        self.rule2_silence = config.SHERPA_RULE2_MIN_TRAILING_SILENCE
        self.rule3_length = config.SHERPA_RULE3_MIN_UTTERANCE_LENGTH
        self.sample_rate = 16000

        # Model state
        self.active_model_id = "sherpa-onnx-streaming-zipformer-en-2023-06-26"
        self.model_name = self.active_model_id
        self.model_type = "streaming"
        self.model_category = "Standard"
        self.is_streaming = True
        self.is_loaded = False
        self.load_error = None
        self.recognizer = None
        self.streaming_recognizer = None
        self.offline_recognizers: Dict[str, Any] = {}

        # Active streaming sessions (for session isolation & safe switching)
        self.active_streams: Dict[str, Any] = {}
        # OnlineRecognizer instances are shared by all sessions. Sherpa stream
        # objects remain isolated, while decode calls are serialized safely.
        self._recognizer_lock = threading.RLock()

        # Preload the default baseline model
        self.load_model(self.active_model_id)

    def find_model_directory(self, model_id: str) -> Optional[Path]:
        """Check both sherpa_models/ and the project root for model directory."""
        candidates = [
            self.models_dir / model_id,
            self.root_dir / model_id
        ]
        for candidate in candidates:
            if candidate.is_dir():
                return candidate
        return None

    def list_models(self) -> List[Dict[str, Any]]:
        """List all verified English Sherpa-ONNX models with disk status and metadata."""
        model_list = []
        for model_id, meta in OFFICIAL_SHERPA_ENGLISH_MODELS.items():
            model_dir = self.find_model_directory(model_id)
            is_downloaded = False
            disk_size_mb = 0.0

            if model_dir:
                files = find_model_files(model_dir, self.use_int8)
                if files["is_valid"]:
                    is_downloaded = True
                    try:
                        total_bytes = sum(f.stat().st_size for f in model_dir.glob("**/*") if f.is_file())
                        disk_size_mb = round(total_bytes / (1024 * 1024), 1)
                    except Exception:
                        pass

            if is_downloaded:
                if model_id == self.active_model_id and self.is_loaded:
                    status = "Ready"
                elif self.load_error and model_id == self.active_model_id:
                    status = "Error"
                else:
                    status = "Installed"
            else:
                status = "Not Installed"

            item = {
                "id": model_id,
                "name": meta["name"],
                "language": meta["language"],
                "lang_code": meta["lang_code"],
                "type": meta["type"],  # 'streaming' | 'offline'
                "category": meta["category"],  # 'Small' | 'Standard' | 'Large'
                "params": meta["params"],
                "dataset": meta["dataset"],
                "streaming_supported": meta["streaming_supported"],
                "offline_supported": meta["offline_supported"],
                "is_downloaded": is_downloaded,
                "downloaded": is_downloaded,
                "status": status,
                "is_active": (model_id == self.active_model_id and self.is_loaded),
                "disk_size_mb": disk_size_mb,
                "license": meta["license"],
                "description": meta["description"]
            }
            model_list.append(item)
        return model_list

    def load_model(self, model_id: str) -> bool:
        """
        Dynamically load a specified Sherpa-ONNX model.
        Properly selects OnlineRecognizer for streaming models or OfflineRecognizer for offline models.
        """
        try:
            import sherpa_onnx
        except ImportError as e:
            self.load_error = f"sherpa_onnx package not installed: {e}"
            logger.error(self.load_error)
            return False

        meta = OFFICIAL_SHERPA_ENGLISH_MODELS.get(model_id)
        if not meta:
            self.load_error = f"Model '{model_id}' is not in the verified catalog."
            logger.error(self.load_error)
            return False

        model_dir = self.find_model_directory(model_id)
        if not model_dir:
            self.load_error = f"Model '{model_id}' is not downloaded. Please download it first."
            logger.warning(self.load_error)
            return False

        files = find_model_files(model_dir, use_int8=self.use_int8)
        if not files["is_valid"]:
            self.load_error = f"Required ONNX files missing or incomplete in {model_dir}"
            logger.error(self.load_error)
            return False

        try:
            logger.info(f"Loading Sherpa-ONNX model '{model_id}' ({meta['type'].upper()} {meta['category']}) from {model_dir}...")
            start_t = time.perf_counter()

            if meta["type"] == "streaming":
                online_rec = sherpa_onnx.OnlineRecognizer.from_transducer(
                    tokens=str(files["tokens"]),
                    encoder=str(files["encoder"]),
                    decoder=str(files["decoder"]),
                    joiner=str(files["joiner"]),
                    num_threads=self.num_threads,
                    sample_rate=16000,
                    feature_dim=80,
                    enable_endpoint_detection=True,
                    rule1_min_trailing_silence=self.rule1_silence,
                    rule2_min_trailing_silence=self.rule2_silence,
                    rule3_min_utterance_length=self.rule3_length,
                    decoding_method="greedy_search",
                    provider=self.provider
                )
                self.recognizer = online_rec
                self.streaming_recognizer = online_rec
                self.is_streaming = True
            else:
                offline_rec = sherpa_onnx.OfflineRecognizer.from_transducer(
                    tokens=str(files["tokens"]),
                    encoder=str(files["encoder"]),
                    decoder=str(files["decoder"]),
                    joiner=str(files["joiner"]),
                    num_threads=self.num_threads,
                    decoding_method="greedy_search"
                )
                self.recognizer = offline_rec
                self.offline_recognizers[model_id] = offline_rec
                self.is_streaming = False

            elapsed = time.perf_counter() - start_t
            self.active_model_id = model_id
            self.model_name = model_id
            self.model_type = meta["type"]
            self.model_category = meta["category"]
            self.is_loaded = True
            self.load_error = None
            logger.info(f"Sherpa-ONNX model '{model_id}' ready in {elapsed:.2f}s!")
            return True

        except Exception as e:
            self.load_error = f"Failed to instantiate {meta['type'].capitalize()}Recognizer for '{model_id}': {e}"
            logger.error(self.load_error)
            return False

    def switch_model(self, model_id: str) -> Dict[str, Any]:
        """
        Safely switch active model, protecting active WebSocket sessions and releasing memory.
        """
        if len(self.active_streams) > 0:
            msg = f"Cannot switch model while {len(self.active_streams)} active voice call session(s) are in progress."
            logger.warning(msg)
            return {"success": False, "message": msg}

        if model_id == self.active_model_id and self.is_loaded:
            return {
                "success": True,
                "message": f"Model '{model_id}' is already active.",
                "active_model": self.get_info()
            }

        # Release current recognizer
        self.recognizer = None
        self.is_loaded = False
        gc.collect()

        success = self.load_model(model_id)
        if success:
            return {
                "success": True,
                "message": f"Switched active model to '{model_id}' ({self.model_category} {self.model_type}).",
                "active_model": self.get_info()
            }
        else:
            # Fallback to default baseline
            logger.warning("Falling back to default baseline streaming model...")
            self.load_model("sherpa-onnx-streaming-zipformer-en-2023-06-26")
            return {
                "success": False,
                "message": f"Failed to switch model: {self.load_error}. Reverted to baseline.",
                "active_model": self.get_info()
            }

    def download_model(self, model_id: str) -> Dict[str, Any]:
        """Download and extract a verified model from official releases."""
        meta = OFFICIAL_SHERPA_ENGLISH_MODELS.get(model_id)
        if not meta:
            return {"success": False, "message": f"Unknown model '{model_id}'."}

        existing_dir = self.find_model_directory(model_id)
        if existing_dir:
            files = find_model_files(existing_dir, self.use_int8)
            if files["is_valid"]:
                return {"success": True, "message": f"Model '{model_id}' is already downloaded and verified."}

        tar_url = meta.get("download_url")
        tar_filename = meta.get("tar_filename")
        if not tar_url or not tar_filename:
            return {"success": False, "message": f"No download URL available for '{model_id}'."}

        tar_path = self.models_dir / tar_filename
        logger.info(f"Downloading {tar_filename} from {tar_url}...")

        try:
            # Download using curl.exe
            cmd_download = ["curl.exe", "-L", "-o", str(tar_path), tar_url]
            res_dl = subprocess.run(cmd_download, cwd=str(self.models_dir), capture_output=True, text=True, timeout=600)
            if res_dl.returncode != 0:
                raise RuntimeError(f"curl download failed: {res_dl.stderr}")

            # Extract using tar -xjf
            logger.info(f"Extracting {tar_filename}...")
            cmd_extract = ["tar", "-xjf", str(tar_path)]
            res_ex = subprocess.run(cmd_extract, cwd=str(self.models_dir), capture_output=True, text=True, timeout=300)
            if res_ex.returncode != 0:
                raise RuntimeError(f"tar extraction failed: {res_ex.stderr}")

            # Remove tar archive
            if tar_path.exists():
                tar_path.unlink()

            # Verify files
            extracted_dir = self.find_model_directory(model_id)
            if not extracted_dir or not find_model_files(extracted_dir, self.use_int8)["is_valid"]:
                raise RuntimeError("Extraction completed but required ONNX files are missing or incomplete.")

            logger.info(f"Model '{model_id}' successfully installed!")
            return {"success": True, "message": f"Model '{model_id}' downloaded and installed successfully."}

        except Exception as e:
            logger.error(f"Error downloading model '{model_id}': {e}")
            if tar_path.exists():
                try: tar_path.unlink()
                except Exception: pass
            return {"success": False, "message": f"Download failed: {str(e)}"}

    def is_available(self) -> bool:
        return self.is_loaded and (self.recognizer is not None or self.streaming_recognizer is not None)

    def get_streaming_recognizer(self):
        """Return dedicated online recognizer for uninterrupted real-time streaming."""
        if self.streaming_recognizer is not None:
            return self.streaming_recognizer
        if self.is_streaming and self.recognizer is not None:
            return self.recognizer
        # Lazily initialize default standard streaming model if needed
        default_model = "sherpa-onnx-streaming-zipformer-en-2023-06-26"
        model_dir = self.find_model_directory(default_model)
        if model_dir:
            try:
                import sherpa_onnx
                files = find_model_files(model_dir, self.use_int8)
                if files["is_valid"]:
                    self.streaming_recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
                        tokens=str(files["tokens"]),
                        encoder=str(files["encoder"]),
                        decoder=str(files["decoder"]),
                        joiner=str(files["joiner"]),
                        num_threads=self.num_threads,
                        sample_rate=16000,
                        feature_dim=80,
                        enable_endpoint_detection=True,
                        rule1_min_trailing_silence=self.rule1_silence,
                        rule2_min_trailing_silence=self.rule2_silence,
                        rule3_min_utterance_length=self.rule3_length,
                        decoding_method="greedy_search",
                        provider=self.provider
                    )
                    logger.info("Initialized dedicated standard streaming recognizer.")
            except Exception as e:
                logger.warning(f"Could not load dedicated streaming recognizer: {e}")
        return self.streaming_recognizer

    def is_streaming_available(self) -> bool:
        return self.get_streaming_recognizer() is not None

    def get_info(self) -> dict:
        return {
            "name": self.model_name,
            "active_model_id": self.active_model_id,
            "type": self.model_type,
            "category": self.model_category,
            "is_streaming": self.is_streaming,
            "status": "ready" if self.is_available() else "unavailable",
            "model_dir": str(self.find_model_directory(self.active_model_id) or self.models_dir),
            "use_int8": self.use_int8,
            "provider": self.provider,
            "threads": self.num_threads,
            "rule2_silence": self.rule2_silence,
            "error": self.load_error,
            "active_sessions_count": len(self.active_streams)
        }

    # -------------------------------------------------------------
    # STREAMING FULL-DUPLEX WEBSOCKET API (Session Isolation)
    # -------------------------------------------------------------
    def create_stream(self, session_id: str):
        """Create a dedicated OnlineStream for a specific WebSocket connection."""
        rec = self.get_streaming_recognizer()
        if not rec:
            raise RuntimeError(f"Sherpa-ONNX streaming recognizer is not available: {self.load_error}")

        stream = rec.create_stream()
        self.active_streams[session_id] = stream
        logger.debug(f"Created Sherpa streaming stream for session: {session_id}")
        return stream

    def get_stream(self, session_id: str):
        return self.active_streams.get(session_id)

    def feed_audio(self, session_id: str, audio_samples: np.ndarray, sample_rate: int = 16000) -> dict:
        """Accept an incremental audio chunk for an active streaming session."""
        rec = self.get_streaming_recognizer()
        if not rec:
            return {"text": "", "is_endpoint": False, "is_ready": False}

        stream = self.active_streams.get(session_id)
        if stream is None:
            stream = self.create_stream(session_id)

        feed_start = time.perf_counter()
        with self._recognizer_lock:
            stream.accept_waveform(sample_rate, audio_samples)
            feed_elapsed = time.perf_counter() - feed_start
            decode_start = time.perf_counter()
            while rec.is_ready(stream):
                rec.decode_stream(stream)
            decode_elapsed = time.perf_counter() - decode_start
            is_endpoint = rec.is_endpoint(stream)
            current_text = rec.get_result(stream)

        return {
            "text": current_text.strip() if current_text else "",
            "is_endpoint": is_endpoint,
            "is_ready": True,
            "feed_ms": round(feed_elapsed * 1000, 3),
            "decode_ms": round(decode_elapsed * 1000, 3)
        }

    def reset_stream(self, session_id: str):
        """Reset internal stream state after an endpoint turn."""
        stream = self.active_streams.get(session_id)
        rec = self.get_streaming_recognizer()
        if stream and rec:
            with self._recognizer_lock:
                rec.reset(stream)

    def get_partial_transcript(self, session_id: str) -> str:
        """Retrieve current partial transcript."""
        stream = self.active_streams.get(session_id)
        rec = self.get_streaming_recognizer()
        if stream and rec:
            with self._recognizer_lock:
                res = rec.get_result(stream)
            return res.strip() if res else ""
        return ""

    def is_endpoint(self, session_id: str) -> bool:
        """Check if endpoint trailing silence threshold is reached."""
        stream = self.active_streams.get(session_id)
        rec = self.get_streaming_recognizer()
        if stream and rec:
            with self._recognizer_lock:
                return rec.is_endpoint(stream)
        return False

    def get_final_transcript(self, session_id: str) -> str:
        """
        Finalize speech turn: feeds 0.2s silence tail to flush remaining lookahead
        acoustic frames in streaming Zipformer transducer, signals input_finished,
        and decodes all remaining tokens.
        """
        stream = self.active_streams.get(session_id)
        rec = self.get_streaming_recognizer()
        if stream and rec:
            try:
                with self._recognizer_lock:
                    silence_tail = np.zeros(int(self.sample_rate * 0.2), dtype=np.float32)
                    stream.accept_waveform(self.sample_rate, silence_tail)
                    stream.input_finished()
                    while rec.is_ready(stream):
                        rec.decode_stream(stream)
                    res = rec.get_result(stream)
            except Exception as e:
                logger.warning(f"Error flushing Sherpa stream for {session_id}: {e}")
                res = ""
            return res.strip() if res else ""
        return ""


    def close_stream(self, session_id: str):
        """Clean up stream when session terminates."""
        if session_id in self.active_streams:
            del self.active_streams[session_id]
            logger.debug(f"Closed Sherpa stream for session: {session_id}")

    # -------------------------------------------------------------
    # BATCH / FILE AUDIO TRANSCRIPTION (Supports Streaming & Offline)
    # -------------------------------------------------------------
    def transcribe_samples(self, samples: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe an entire sample buffer with Sherpa-ONNX (Streaming or Offline)."""
        if not self.is_available():
            raise RuntimeError(f"Sherpa-ONNX not available: {self.load_error}")

        if len(samples) == 0:
            return ""

        # Normalize audio: remove DC bias and calibrate gain with conservative headroom
        mean_v = np.mean(samples)
        samples = samples - mean_v
        peak_v = float(np.max(np.abs(samples))) if len(samples) > 0 else 0.0
        if peak_v > 0.98:
            samples = samples * (0.95 / peak_v)
        elif peak_v < 0.04 and peak_v > 1e-4:
            # Conservative boost only for faint signals, strictly capped to 2.0x
            rms_val = float(np.sqrt(np.mean(samples**2)))
            samples = np.clip(samples * min(2.0, 0.15 / max(rms_val, 1e-4)), -1.0, 1.0)

        if self.is_streaming:
            # Online stream processed in causal frames with head priming
            stream = self.recognizer.create_stream()
            silence_head = np.zeros(int(sample_rate * 0.12), dtype=np.float32)
            stream.accept_waveform(sample_rate, silence_head)

            chunk_size = 640
            for i in range(0, len(samples), chunk_size):
                chunk = samples[i:i + chunk_size]
                stream.accept_waveform(sample_rate, chunk)
                while self.recognizer.is_ready(stream):
                    self.recognizer.decode_stream(stream)

            # Flush lookahead context window
            silence_tail = np.zeros(int(sample_rate * 0.25), dtype=np.float32)
            stream.accept_waveform(sample_rate, silence_tail)
            stream.input_finished()
            while self.recognizer.is_ready(stream):
                self.recognizer.decode_stream(stream)

            result = self.recognizer.get_result(stream)
            return result.strip() if result else ""
        else:
            # OfflineRecognizer processes entire waveform
            stream = self.recognizer.create_stream()
            stream.accept_waveform(sample_rate, samples)
            self.recognizer.decode_stream(stream)
            return stream.result.text.strip()

    def _get_offline_recognizer(self, model_id: str):
        """Load and cache an offline recognizer without changing live streaming state."""
        cached = self.offline_recognizers.get(model_id)
        if cached is not None:
            return cached

        import sherpa_onnx
        meta = OFFICIAL_SHERPA_ENGLISH_MODELS.get(model_id)
        if not meta or meta["type"] != "offline":
            raise ValueError(f"'{model_id}' is not a verified offline Sherpa model.")
        model_dir = self.find_model_directory(model_id)
        if not model_dir:
            raise FileNotFoundError(f"Offline Sherpa model '{model_id}' is not installed.")
        files = find_model_files(model_dir, use_int8=self.use_int8)
        if not files["is_valid"]:
            raise RuntimeError(f"Offline Sherpa model '{model_id}' is incomplete.")

        recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            tokens=str(files["tokens"]),
            encoder=str(files["encoder"]),
            decoder=str(files["decoder"]),
            joiner=str(files["joiner"]),
            num_threads=self.num_threads,
            decoding_method="greedy_search"
        )
        self.offline_recognizers[model_id] = recognizer
        logger.info("Cached offline Sherpa recognizer '%s' for finalization.", model_id)
        return recognizer

    def transcribe_offline_model(self, audio_source, model_id: str) -> dict:
        """Transcribe complete audio with a cached offline model, preserving streaming state."""
        import io
        import av

        if isinstance(audio_source, np.ndarray):
            samples = audio_source.astype(np.float32)
        elif isinstance(audio_source, (bytes, bytearray)) and len(audio_source) % 2 == 0 and not audio_source.startswith(
            (b"RIFF", b"\x1a\x45\xdf\xa3", b"ID3", b"\xff\xfb", b"OggS", b"fLaC")
        ):
            samples = np.frombuffer(audio_source, dtype=np.int16).astype(np.float32) / 32768.0
        else:
            source = io.BytesIO(audio_source) if isinstance(audio_source, (bytes, bytearray)) else audio_source
            container = av.open(source)
            audio_stream = next(s for s in container.streams if s.type == "audio")
            resampler = av.AudioResampler(format="fltp", layout="mono", rate=16000)
            decoded = []
            for frame in container.decode(audio_stream):
                decoded.extend(rf.to_ndarray()[0] for rf in resampler.resample(frame))
            decoded.extend(rf.to_ndarray()[0] for rf in resampler.resample(None))
            container.close()
            samples = np.concatenate(decoded) if decoded else np.zeros(0, dtype=np.float32)

        if len(samples) == 0:
            return {"transcript": "", "model": model_id, "latency": 0.0}

        recognizer = self._get_offline_recognizer(model_id)
        stream = recognizer.create_stream()
        stream.accept_waveform(16000, samples)
        started = time.perf_counter()
        recognizer.decode_stream(stream)
        transcript = stream.result.text.strip()
        latency = round(time.perf_counter() - started, 3)
        return {"transcript": transcript, "model": model_id, "latency": latency}

    def transcribe_audio_source(self, audio_source) -> dict:
        """
        Transcribe an audio file, BytesIO, or audio bytes using the active Sherpa model.
        Decodes any format using av or soundfile to 16kHz mono float32.
        """
        import av
        if not self.is_available():
            raise RuntimeError(f"Sherpa-ONNX engine is not available: {self.load_error}")

        if isinstance(audio_source, (str, Path)) and not os.path.exists(audio_source):
            raise FileNotFoundError(f"Audio file not found: {audio_source}")

        start_time = time.perf_counter()
        samples_np = None
        orig_rate = 16000
        orig_channels = 1

        if isinstance(audio_source, np.ndarray):
            samples_np = audio_source.astype(np.float32)
        elif isinstance(audio_source, (bytes, bytearray)):
            import io
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
                audio_source = io.BytesIO(audio_source)
        elif hasattr(audio_source, "getvalue"):
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

        if samples_np is None:
            container = av.open(audio_source)
            audio_stream = next(s for s in container.streams if s.type == 'audio')
            resampler = av.AudioResampler(format='fltp', layout='mono', rate=16000)
            samples = []
            for frame in container.decode(audio_stream):
                for rf in resampler.resample(frame):
                    samples.append(rf.to_ndarray()[0])
            for rf in resampler.resample(None):
                samples.append(rf.to_ndarray()[0])
            orig_rate = getattr(audio_stream, 'rate', 16000)
            orig_channels = getattr(audio_stream, 'channels', 1)
            container.close()

            if samples:
                samples_np = np.concatenate(samples)
            else:
                samples_np = np.zeros(0, dtype=np.float32)

        duration = len(samples_np) / 16000.0
        peak_val = float(np.max(np.abs(samples_np))) if len(samples_np) > 0 else 0.0
        peak_dbfs = round(20 * float(np.log10(max(peak_val, 1e-5))), 1)
        rms_val = round(float(np.sqrt(np.mean(samples_np**2))), 4) if len(samples_np) > 0 else 0.0
        logger.info(
            f"[Sherpa STT Audio Telemetry] duration={duration:.2f}s, in_rate={orig_rate}Hz, "
            f"in_channels={orig_channels}, peak={peak_dbfs} dBFS, rms={rms_val}, model={self.active_model_id}"
        )

        t_asr_start = time.perf_counter()
        transcript = self.transcribe_samples(samples_np, sample_rate=16000)
        asr_time = time.perf_counter() - t_asr_start
        total_latency = round(time.perf_counter() - start_time, 3)

        duration = len(samples_np) / 16000.0
        rtf = round(total_latency / max(duration, 0.001), 3)
        speed = round(duration / max(total_latency, 0.001), 2)

        return {
            "transcript": transcript,
            "language": "en",
            "language_probability": 1.0,
            "duration": round(duration, 3),
            "stt_latency": total_latency,
            "rtf": rtf,
            "speed_factor": speed,
            "device": self.provider,
            "model_name": self.model_name,
            "model_category": self.model_category,
            "model_type": self.model_type
        }


# Global singleton
sherpa_engine: Optional[SherpaStreamingEngine] = None

def get_sherpa_engine() -> SherpaStreamingEngine:
    global sherpa_engine
    if sherpa_engine is None:
        sherpa_engine = SherpaStreamingEngine()
    return sherpa_engine
