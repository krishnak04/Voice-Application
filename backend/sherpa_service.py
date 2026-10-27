import os
import io
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import config

logger = logging.getLogger("sherpa_service")

class SherpaStreamingEngine:
    """
    High-performance streaming speech recognition engine powered by Sherpa-ONNX.
    Uses streaming Zipformer transducer models for sub-100ms first-partial latency
    and integrated endpoint detection for full-duplex conversational voice telephony.
    """
    def __init__(self):
        self.model_dir = Path(config.SHERPA_MODEL_DIR).resolve()
        self.use_int8 = config.SHERPA_USE_INT8
        self.num_threads = config.SHERPA_NUM_THREADS
        self.provider = config.SHERPA_PROVIDER
        self.rule1_silence = config.SHERPA_RULE1_MIN_TRAILING_SILENCE
        self.rule2_silence = config.SHERPA_RULE2_MIN_TRAILING_SILENCE
        self.rule3_length = config.SHERPA_RULE3_MIN_UTTERANCE_LENGTH
        self.sample_rate = 16000

        self.recognizer = None

        self.active_streams: Dict[str, Any] = {}
        self.model_name = "sherpa-onnx-streaming-zipformer-en-2023-06-26"
        self.is_loaded = False
        self.load_error = None

        self._initialize_recognizer()

    def _initialize_recognizer(self):
        try:
            import sherpa_onnx
        except ImportError as e:
            self.load_error = f"sherpa_onnx package not installed: {e}"
            logger.error(self.load_error)
            return

        if not self.model_dir.exists():
            self.load_error = f"Model directory not found at {self.model_dir}"
            logger.warning(self.load_error)
            return

        tokens = self.model_dir / "tokens.txt"
        
        # Check int8 models first if configured
        if self.use_int8:
            enc = self.model_dir / "encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx"
            dec = self.model_dir / "decoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx"
            join = self.model_dir / "joiner-epoch-99-avg-1-chunk-16-left-128.int8.onnx"
        else:
            enc = dec = join = None

        # Fallback to standard onnx if int8 not found or not selected
        if not (enc and enc.exists() and dec and dec.exists() and join and join.exists()):
            enc = self.model_dir / "encoder-epoch-99-avg-1-chunk-16-left-128.onnx"
            dec = self.model_dir / "decoder-epoch-99-avg-1-chunk-16-left-128.onnx"
            join = self.model_dir / "joiner-epoch-99-avg-1-chunk-16-left-128.onnx"

        if not (tokens.exists() and enc.exists() and dec.exists() and join.exists()):
            self.load_error = f"Required ONNX files missing in {self.model_dir}"
            logger.error(self.load_error)
            return

        try:
            logger.info(f"Loading Sherpa-ONNX streaming model from {self.model_dir} (int8={enc.name.endswith('.int8.onnx')})...")
            start_t = time.perf_counter()
            self.recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
                tokens=str(tokens),
                encoder=str(enc),
                decoder=str(dec),
                joiner=str(join),
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
            elapsed = time.perf_counter() - start_t
            self.is_loaded = True
            self.load_error = None
            logger.info(f"Sherpa-ONNX streaming Zipformer loaded successfully in {elapsed:.2f}s!")
        except Exception as e:
            self.load_error = f"Failed to instantiate OnlineRecognizer: {e}"
            logger.error(self.load_error)

    def is_available(self) -> bool:
        return self.is_loaded and self.recognizer is not None

    def get_info(self) -> dict:
        return {
            "name": self.model_name,
            "status": "ready" if self.is_available() else "unavailable",
            "model_dir": str(self.model_dir),
            "use_int8": self.use_int8,
            "provider": self.provider,
            "threads": self.num_threads,
            "rule2_silence": self.rule2_silence,
            "error": self.load_error
        }

    def create_stream(self, session_id: str):
        """Create a dedicated OnlineStream for a specific WebSocket connection."""
        if not self.is_available():
            raise RuntimeError(f"Sherpa-ONNX recognizer is not available: {self.load_error}")
        stream = self.recognizer.create_stream()
        self.active_streams[session_id] = stream
        logger.debug(f"Created Sherpa streaming stream for session: {session_id}")
        return stream

    def get_stream(self, session_id: str):
        return self.active_streams.get(session_id)

    def feed_audio(self, session_id: str, audio_samples: np.ndarray, sample_rate: int = 16000) -> dict:
        """
        Accept an incremental audio chunk (float32 array normalized to [-1.0, 1.0]).
        Decodes ready frames and checks endpoint detection.
        """
        if not self.is_available():
            return {"text": "", "is_endpoint": False, "is_ready": False}

        stream = self.active_streams.get(session_id)
        if stream is None:
            stream = self.create_stream(session_id)

        # Accept the waveform
        stream.accept_waveform(sample_rate, audio_samples)

        # Decode frames while recognizer has sufficient frames ready
        while self.recognizer.is_ready(stream):
            self.recognizer.decode_stream(stream)

        is_endpoint = self.recognizer.is_endpoint(stream)
        current_text = self.recognizer.get_result(stream)

        return {
            "text": current_text.strip() if current_text else "",
            "is_endpoint": is_endpoint,
            "is_ready": True
        }

    def reset_stream(self, session_id: str):
        """Reset the internal state of a stream after an endpoint or conversational turn."""
        stream = self.active_streams.get(session_id)
        if stream and self.recognizer:
            self.recognizer.reset(stream)

    def get_partial_transcript(self, session_id: str) -> str:
        """Retrieve the current partial decoded text for an active stream."""
        stream = self.active_streams.get(session_id)
        if stream and self.recognizer:
            res = self.recognizer.get_result(stream)
            return res.strip() if res else ""
        return ""

    def is_endpoint(self, session_id: str) -> bool:
        """Check if endpoint (turn completion / trailing silence) is reached."""
        stream = self.active_streams.get(session_id)
        if stream and self.recognizer:
            return self.recognizer.is_endpoint(stream)
        return False

    def get_final_transcript(self, session_id: str) -> str:
        """Retrieve the finalized utterance text for a stream."""
        return self.get_partial_transcript(session_id)

    def close_stream(self, session_id: str):
        """Remove and clean up stream when session terminates."""
        if session_id in self.active_streams:
            del self.active_streams[session_id]
            logger.debug(f"Closed Sherpa stream for session: {session_id}")


    def transcribe_samples(self, samples: np.ndarray, sample_rate: int = 16000) -> str:
        """Helper to transcribe an entire sample buffer with Sherpa-ONNX."""
        if not self.is_available():
            raise RuntimeError(f"Sherpa-ONNX not available: {self.load_error}")

        stream = self.recognizer.create_stream()
        chunk_size = 1600
        for i in range(0, len(samples), chunk_size):
            chunk = samples[i:i+chunk_size]
            stream.accept_waveform(sample_rate, chunk)
            while self.recognizer.is_ready(stream):
                self.recognizer.decode_stream(stream)

        stream.input_finished()
        while self.recognizer.is_ready(stream):
            self.recognizer.decode_stream(stream)
        result = self.recognizer.get_result(stream)
        return result.strip() if result else ""

    def transcribe_audio_source(self, audio_source) -> dict:
        """
        Transcribe an audio file, BytesIO, or audio bytes using Sherpa-ONNX streaming Zipformer.
        Decodes any format using av or soundfile to 16kHz mono float32.
        """
        import av
        if not self.is_available():
            raise RuntimeError(f"Sherpa-ONNX streaming engine is not available: {self.load_error}")

        if isinstance(audio_source, (str, Path)) and not os.path.exists(audio_source):
            raise FileNotFoundError(f"Audio file not found: {audio_source}")

        start_time = time.perf_counter()
        container = av.open(audio_source)
        audio_stream = next(s for s in container.streams if s.type == 'audio')
        resampler = av.AudioResampler(format='fltp', layout='mono', rate=16000)
        samples = []
        for frame in container.decode(audio_stream):
            for rf in resampler.resample(frame):
                samples.append(rf.to_ndarray()[0])
        container.close()

        if samples:
            samples_np = np.concatenate(samples)
        else:
            samples_np = np.zeros(0, dtype=np.float32)

        decode_time = time.perf_counter() - start_time
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
            "model_name": self.model_name
        }


# Global singleton
sherpa_engine: Optional[SherpaStreamingEngine] = None

def get_sherpa_engine() -> SherpaStreamingEngine:
    global sherpa_engine
    if sherpa_engine is None:
        sherpa_engine = SherpaStreamingEngine()
    return sherpa_engine
