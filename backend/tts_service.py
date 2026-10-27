import io
import time
import os
import wave
import logging
import asyncio
from typing import Optional, Tuple
import config

logger = logging.getLogger("tts_service")

# Available voices metadata
DEFAULT_VOICES = {
    "kokoro": [
        {"id": "af_heart", "name": "Heart (US Female - Warm & Natural)", "lang": "en", "accent": "US"},
        {"id": "af_bella", "name": "Bella (US Female - Cheerful)", "lang": "en", "accent": "US"},
        {"id": "af_sarah", "name": "Sarah (US Female - Soft & Calm)", "lang": "en", "accent": "US"},
        {"id": "am_adam", "name": "Adam (US Male - Professional)", "lang": "en", "accent": "US"},
        {"id": "am_michael", "name": "Michael (US Male - Calm)", "lang": "en", "accent": "US"},
        {"id": "bf_emma", "name": "Emma (UK Female)", "lang": "en", "accent": "UK"},
        {"id": "bm_george", "name": "George (UK Male)", "lang": "en", "accent": "UK"},
    ],
    "piper": [
        {"id": "en_US-lessac-medium", "name": "Lessac (US Female - Fast Telephony)", "lang": "en", "accent": "US"},
        {"id": "en_US-bryce-medium", "name": "Bryce (US Male - Crisp)", "lang": "en", "accent": "US"},
        {"id": "en_US-amy-medium", "name": "Amy (US Female - Clear)", "lang": "en", "accent": "US"},
    ],
    "edge_tts": [
        {"id": "en-IN-NeerjaNeural", "name": "Neerja (Indian English Female - Expressive)", "lang": "en-IN", "accent": "IN"},
        {"id": "en-IN-PrabhatNeural", "name": "Prabhat (Indian English Male - Confident)", "lang": "en-IN", "accent": "IN"},
        {"id": "hi-IN-SwaraNeural", "name": "Swara (Hindi Female - Clear)", "lang": "hi", "accent": "IN-Hindi"},
        {"id": "hi-IN-MadhurNeural", "name": "Madhur (Hindi Male - Natural)", "lang": "hi", "accent": "IN-Hindi"},
        {"id": "en-US-JennyNeural", "name": "Jenny (US Female - Neural)", "lang": "en", "accent": "US"},
        {"id": "en-US-GuyNeural", "name": "Guy (US Male - Neural)", "lang": "en", "accent": "US"},
        {"id": "en-US-AriaNeural", "name": "Aria (US Female - Expressive)", "lang": "en", "accent": "US"},
    ]
}

class TTSEngine:
    def __init__(self):
        self.preferred_engine = getattr(config, "TTS_ENGINE", "auto")
        self.device = getattr(config, "TTS_DEVICE", "cuda")  # or cpu
        self.default_voice = getattr(config, "TTS_VOICE", "af_heart")
        
        self.kokoro_pipeline = None
        self.piper_voice = None
        self.active_engine = "none"

        self._initialize_best_engine()

    def _initialize_best_engine(self):
        """Detect and initialize the best available TTS backend."""
        # 1. Try Kokoro-82M (Best quality, ultra-low VRAM ~300MB)
        if self.preferred_engine in ("auto", "kokoro"):
            try:
                from kokoro import KPipeline
                lang_code = 'a'  # American English default
                logger.info(f"Loading Kokoro-82M TTS on {self.device}...")
                self.kokoro_pipeline = KPipeline(lang_code=lang_code)
                self.active_engine = "kokoro"
                logger.info("Kokoro-82M TTS engine successfully initialized!")
                return
            except Exception as e:
                logger.debug(f"Kokoro not loaded: {e}")

        # 2. Try Piper TTS (0 VRAM, ultra-fast CPU telephony)
        if self.preferred_engine in ("auto", "piper"):
            try:
                import piper
                self.active_engine = "piper"
                logger.info("Piper TTS engine available!")
                return
            except Exception as e:
                logger.debug(f"Piper not loaded: {e}")

        # 3. Try Edge-TTS (0 VRAM, zero CPU, cloud neural)
        if self.preferred_engine in ("auto", "edge_tts", "edge"):
            try:
                import edge_tts
                self.active_engine = "edge_tts"
                logger.info("Edge-TTS engine available!")
                return
            except Exception as e:
                logger.debug(f"Edge-TTS not loaded: {e}")

        logger.warning(
            "No TTS engine installed yet. TTS calls will use an in-memory tone fallback. "
            "To enable realistic voice, run: pip install kokoro soundfile OR pip install edge-tts"
        )
        self.active_engine = "fallback"

    def get_info(self) -> dict:
        all_voices = list(DEFAULT_VOICES.get(self.active_engine, []))
        if self.active_engine != "edge_tts":
            for v in DEFAULT_VOICES.get("edge_tts", []):
                if not any(existing["id"] == v["id"] for existing in all_voices):
                    all_voices.append(v)
        return {
            "active_engine": self.active_engine,
            "preferred_engine": self.preferred_engine,
            "device": self.device,
            "default_voice": self.default_voice,
            "available_voices": all_voices
        }

    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
        pitch: Optional[str] = None,
        engine: Optional[str] = None
    ) -> dict:
        """
        Synthesize text to audio with speed and pitch support.
        """
        clean_text = text.strip()
        if not clean_text:
            return self._generate_silence_wav()

        req_engine = (engine or "").lower().strip()
        active_voice = voice or self.default_voice

        # Smart Engine Resolution
        if not req_engine or req_engine == "auto":
            voice_lower = active_voice.lower()
            if "neural" in voice_lower or voice_lower.startswith(("en-in", "hi-in", "ta-in", "te-in", "bn-in", "mr-in", "gu-in", "en-us", "en-gb", "en-ca", "en-au", "en-ie")):
                resolved_engine = "edge_tts"
            elif any(active_voice.startswith(p) for p in ("af_", "am_", "bf_", "bm_")):
                resolved_engine = "kokoro" if (self.kokoro_pipeline is not None or self.active_engine == "kokoro") else "edge_tts"
            elif any(p in voice_lower for p in ("lessac", "bryce", "amy", "medium")):
                resolved_engine = "piper"
            else:
                resolved_engine = self.active_engine
        else:
            resolved_engine = req_engine

        start_time = time.perf_counter()

        # Route 1: Kokoro-82M
        if resolved_engine == "kokoro":
            try:
                return self._synthesize_kokoro(clean_text, active_voice, speed, start_time)
            except Exception as e:
                logger.error(f"Kokoro synthesis failed: {e}. Falling back to Edge-TTS...")
                try:
                    fallback_voice = "en-IN-NeerjaNeural" if "in" in active_voice.lower() else "en-US-JennyNeural"
                    return self._synthesize_edge_tts(clean_text, fallback_voice, speed, pitch, start_time)
                except Exception:
                    pass

        # Route 2: Edge-TTS (Primary Neural Voice Provider)
        if resolved_engine in ("edge_tts", "edge"):
            try:
                return self._synthesize_edge_tts(clean_text, active_voice, speed, pitch, start_time)
            except Exception as e:
                logger.error(f"Edge-TTS synthesis failed: {e}")

        # Route 3: Piper
        if resolved_engine == "piper":
            try:
                return self._synthesize_piper(clean_text, active_voice, speed, start_time)
            except Exception as e:
                logger.error(f"Piper synthesis failed: {e}")

        # Final Fallback
        return self._synthesize_fallback(clean_text, start_time)

    async def async_synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
        pitch: Optional[str] = None,
        engine: Optional[str] = None
    ) -> dict:
        """
        Native asynchronous synthesis with pitch and rate modulation.
        """
        clean_text = text.strip()
        if not clean_text:
            return self._generate_silence_wav()

        req_engine = (engine or "").lower().strip()
        active_voice = voice or self.default_voice

        if not req_engine or req_engine == "auto":
            voice_lower = active_voice.lower()
            if "neural" in voice_lower or voice_lower.startswith(("en-in", "hi-in", "ta-in", "te-in", "bn-in", "mr-in", "gu-in", "en-us", "en-gb", "en-ca", "en-au", "en-ie")):
                resolved_engine = "edge_tts"
            elif any(active_voice.startswith(p) for p in ("af_", "am_", "bf_", "bm_")):
                resolved_engine = "kokoro" if (self.kokoro_pipeline is not None or self.active_engine == "kokoro") else "edge_tts"
            elif any(p in voice_lower for p in ("lessac", "bryce", "amy", "medium")):
                resolved_engine = "piper"
            else:
                resolved_engine = self.active_engine
        else:
            resolved_engine = req_engine

        start_time = time.perf_counter()

        if resolved_engine in ("edge_tts", "edge"):
            try:
                return await self._async_synthesize_edge_tts(clean_text, active_voice, speed, pitch, start_time)
            except Exception as e:
                logger.error(f"Async Edge-TTS synthesis failed: {e}")

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.synthesize(clean_text, active_voice, speed, pitch, resolved_engine))

    async def _async_synthesize_edge_tts(self, text: str, voice: str, speed: float, pitch: Optional[str], start_time: float) -> dict:
        """Native async Edge-TTS synthesis with rate and pitch controls."""
        import edge_tts

        edge_voice = voice if "Neural" in voice else "en-US-JennyNeural"
        rate_str = f"+{int((speed - 1.0) * 100)}%" if speed >= 1.0 else f"-{int((1.0 - speed) * 100)}%"
        pitch_str = pitch if (pitch and pitch.strip()) else "+0Hz"

        communicate = edge_tts.Communicate(text, voice=edge_voice, rate=rate_str, pitch=pitch_str)
        mp3_buffer = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                mp3_buffer.write(chunk["data"])

        audio_bytes = mp3_buffer.getvalue()
        elapsed = time.perf_counter() - start_time
        est_duration = max(round(len(audio_bytes) / 6000.0, 2), 0.5)

        return {
            "audio_bytes": audio_bytes,
            "media_type": "audio/mpeg",
            "format": "mp3",
            "tts_latency": round(elapsed, 3),
            "engine": "edge-tts",
            "voice": edge_voice,
            "pitch": pitch_str,
            "duration": est_duration
        }

    def _synthesize_edge_tts(self, text: str, voice: str, speed: float, pitch: Optional[str], start_time: float) -> dict:
        """Edge-TTS streaming neural voice synthesis (synchronous wrapper)."""
        import edge_tts

        edge_voice = voice if "Neural" in voice else "en-US-JennyNeural"
        rate_str = f"+{int((speed - 1.0) * 100)}%" if speed >= 1.0 else f"-{int((1.0 - speed) * 100)}%"
        pitch_str = pitch if (pitch and pitch.strip()) else "+0Hz"

        async def _run_edge():
            communicate = edge_tts.Communicate(text, voice=edge_voice, rate=rate_str, pitch=pitch_str)
            mp3_buffer = io.BytesIO()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    mp3_buffer.write(chunk["data"])
            return mp3_buffer.getvalue()

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import nest_asyncio
                nest_asyncio.apply()
                audio_bytes = loop.run_until_complete(_run_edge())
            else:
                audio_bytes = loop.run_until_complete(_run_edge())
        except Exception:
            new_loop = asyncio.new_event_loop()
            audio_bytes = new_loop.run_until_complete(_run_edge())
            new_loop.close()

        elapsed = time.perf_counter() - start_time
        est_duration = max(round(len(audio_bytes) / 6000.0, 2), 0.5)

        return {
            "audio_bytes": audio_bytes,
            "media_type": "audio/mpeg",
            "format": "mp3",
            "tts_latency": round(elapsed, 3),
            "engine": "edge-tts",
            "voice": edge_voice,
            "pitch": pitch_str,
            "duration": est_duration
        }

    def _synthesize_kokoro(self, text: str, voice: str, speed: float, start_time: float) -> dict:
        """Kokoro-82M synthesis via PyTorch/SoundFile."""
        import soundfile as sf
        import numpy as np

        if self.kokoro_pipeline is None:
            from kokoro import KPipeline
            self.kokoro_pipeline = KPipeline(lang_code='a')

        kokoro_voice = voice if voice.startswith(("af_", "am_", "bf_", "bm_")) else "af_heart"
        
        # Kokoro yields (graphemes, phonemes, audio_tensor)
        generator = self.kokoro_pipeline(text, voice=kokoro_voice, speed=speed, split_pattern=r'\n+')
        all_audio = []
        sample_rate = 24000

        for _, _, audio in generator:
            if audio is not None:
                if hasattr(audio, "cpu"):
                    audio = audio.cpu().numpy()
                all_audio.append(audio)

        if not all_audio:
            raise RuntimeError("Kokoro produced empty audio chunks")

        combined = np.concatenate(all_audio)
        duration = len(combined) / sample_rate

        out_buffer = io.BytesIO()
        sf.write(out_buffer, combined, sample_rate, format='WAV', subtype='PCM_16')
        audio_bytes = out_buffer.getvalue()

        elapsed = time.perf_counter() - start_time
        return {
            "audio_bytes": audio_bytes,
            "media_type": "audio/wav",
            "format": "wav",
            "tts_latency": round(elapsed, 3),
            "engine": "kokoro-82m",
            "voice": kokoro_voice,
            "duration": round(duration, 3)
        }

    def _synthesize_piper(self, text: str, voice: str, speed: float, start_time: float) -> dict:
        """Piper CPU VITS synthesis."""
        # If piper binary or python package is configured
        import subprocess
        piper_bin = getattr(config, "PIPER_BINARY", "piper")
        piper_model = getattr(config, "PIPER_MODEL", "en_US-lessac-medium.onnx")

        cmd = [piper_bin, "--model", piper_model, "--output_raw"]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        raw_pcm, err = proc.communicate(input=text.encode("utf-8"))

        if proc.returncode != 0:
            raise RuntimeError(f"Piper error: {err.decode('utf-8')}")

        # Wrap raw PCM in WAV container (22050 Hz, 16-bit mono)
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(22050)
            wf.writeframes(raw_pcm)

        audio_bytes = wav_buffer.getvalue()
        duration = len(raw_pcm) / (22050 * 2)
        elapsed = time.perf_counter() - start_time

        return {
            "audio_bytes": audio_bytes,
            "media_type": "audio/wav",
            "format": "wav",
            "tts_latency": round(elapsed, 3),
            "engine": "piper",
            "voice": voice or "en_US-lessac",
            "duration": round(duration, 3)
        }

    def _synthesize_fallback(self, text: str, start_time: float) -> dict:
        """Graceful mock sound so client receives audible audio even before installing packages."""
        import math
        import struct

        sample_rate = 16000
        duration = min(max(len(text) * 0.05, 0.8), 3.0)  # estimate length
        num_samples = int(sample_rate * duration)

        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            # Create a pleasant chime/telephony beep
            for i in range(num_samples):
                t = i / sample_rate
                # 440 Hz gentle sine wave decaying
                envelope = math.exp(-t * 2.0)
                sample = int(10000 * math.sin(2 * math.pi * 440 * t) * envelope)
                wf.writeframesraw(struct.pack("<h", max(-32768, min(32767, sample))))

        elapsed = time.perf_counter() - start_time
        return {
            "audio_bytes": wav_buffer.getvalue(),
            "media_type": "audio/wav",
            "format": "wav",
            "tts_latency": round(elapsed, 3),
            "engine": "tone_fallback",
            "voice": "beep",
            "duration": round(duration, 2)
        }

    def _generate_silence_wav(self) -> dict:
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframesraw(b"\x00" * 3200)  # 0.1s silence
        return {
            "audio_bytes": wav_buffer.getvalue(),
            "media_type": "audio/wav",
            "format": "wav",
            "tts_latency": 0.001,
            "engine": "silence",
            "voice": "none",
            "duration": 0.1
        }

    def get_human_voices(self) -> dict:
        """
        Returns verified real human voices supported by the active Edge-TTS provider.
        Categorized by demographics (Adult Female, Adult Male, Child / Youth, Mature / Narrator),
        with pitch support, accents, and natural preview sentences.
        """
        HUMAN_VOICES = [
            # 👶 CHILD / YOUTH VOICES (Real Microsoft Edge-TTS Youth Models)
            {
                "id": "en-US-AnaNeural",
                "name": "Ana",
                "full_name": "Microsoft Ana Online (Natural) - English (United States)",
                "gender": "Female",
                "locale": "en-US",
                "locale_name": "English (United States)",
                "accent": "US",
                "category": "Child / Youth",
                "age_group": "Youth / Child",
                "style": "Lively, clear, cheerful youth girl voice",
                "engine": "edge_tts",
                "pitch_supported": True,
                "sample_text": "Hello! My name is Ana. I am a natural human youth voice. What can we learn today?"
            },
            {
                "id": "en-GB-MaisieNeural",
                "name": "Maisie",
                "full_name": "Microsoft Maisie Online (Natural) - English (United Kingdom)",
                "gender": "Female",
                "locale": "en-GB",
                "locale_name": "English (United Kingdom)",
                "accent": "UK",
                "category": "Child / Youth",
                "age_group": "Youth / Child",
                "style": "Friendly, expressive youth British girl",
                "engine": "edge_tts",
                "pitch_supported": True,
                "sample_text": "Hi there! I am Maisie, a youth British voice. Lovely to speak with you today!"
            },

            # 👩 ADULT FEMALE VOICES (Natural, Professional & Conversational)
            {
                "id": "en-US-JennyNeural",
                "name": "Jenny",
                "full_name": "Microsoft Jenny Online (Natural) - English (United States)",
                "gender": "Female",
                "locale": "en-US",
                "locale_name": "English (United States)",
                "accent": "US",
                "category": "Adult Female",
                "age_group": "Adult",
                "style": "Warm, considerate, balanced conversational human voice",
                "engine": "edge_tts",
                "pitch_supported": True,
                "sample_text": "Hello, thank you for calling. My name is Jenny. How can I assist you today?"
            },
            {
                "id": "en-US-AvaNeural",
                "name": "Ava",
                "full_name": "Microsoft Ava Online (Natural) - English (United States)",
                "gender": "Female",
                "locale": "en-US",
                "locale_name": "English (United States)",
                "accent": "US",
                "category": "Adult Female",
                "age_group": "Adult",
                "style": "Expressive, caring, modern copilot voice",
                "engine": "edge_tts",
                "pitch_supported": True,
                "sample_text": "Hi there! I'm Ava. I'm right here to help you get things done smoothly."
            },
            {
                "id": "en-US-EmmaNeural",
                "name": "Emma",
                "full_name": "Microsoft Emma Online (Natural) - English (United States)",
                "gender": "Female",
                "locale": "en-US",
                "locale_name": "English (United States)",
                "accent": "US",
                "category": "Adult Female",
                "age_group": "Adult",
                "style": "Cheerful, crisp, articulate customer service",
                "engine": "edge_tts",
                "pitch_supported": True,
                "sample_text": "Good day! I am Emma. I'd be delighted to help answer any questions you have."
            },
            {
                "id": "en-US-AriaNeural",
                "name": "Aria",
                "full_name": "Microsoft Aria Online (Natural) - English (United States)",
                "gender": "Female",
                "locale": "en-US",
                "locale_name": "English (United States)",
                "accent": "US",
                "category": "Adult Female",
                "age_group": "Adult",
                "style": "Positive, confident, professional executive voice",
                "engine": "edge_tts",
                "pitch_supported": True,
                "sample_text": "Welcome to our customer service line. My name is Aria, and I am ready to assist."
            },
            {
                "id": "en-US-MichelleNeural",
                "name": "Michelle",
                "full_name": "Microsoft Michelle Online (Natural) - English (United States)",
                "gender": "Female",
                "locale": "en-US",
                "locale_name": "English (United States)",
                "accent": "US",
                "category": "Adult Female",
                "age_group": "Adult",
                "style": "Pleasant, friendly, engaging storytelling tone",
                "engine": "edge_tts",
                "pitch_supported": True,
                "sample_text": "Hello, this is Michelle. I hope you are having a wonderful day so far."
            },
            {
                "id": "en-IN-NeerjaNeural",
                "name": "Neerja",
                "full_name": "Microsoft Neerja Online (Natural) - English (India)",
                "gender": "Female",
                "locale": "en-IN",
                "locale_name": "English (India)",
                "accent": "IN",
                "category": "Adult Female",
                "age_group": "Adult",
                "style": "Warm, natural, clear Indian English conversational voice",
                "engine": "edge_tts",
                "pitch_supported": True,
                "sample_text": "Namaste! Thank you for calling. My name is Neerja. How may I help you with your inquiry?"
            },
            {
                "id": "en-IN-NeerjaExpressiveNeural",
                "name": "Neerja Expressive",
                "full_name": "Microsoft Neerja Expressive Online (Natural) - English (India)",
                "gender": "Female",
                "locale": "en-IN",
                "locale_name": "English (India)",
                "accent": "IN",
                "category": "Adult Female",
                "age_group": "Adult",
                "style": "Dynamic, vibrant, expressive Indian English voice",
                "engine": "edge_tts",
                "pitch_supported": True,
                "sample_text": "Hello! I am Neerja Expressive. Let's make this conversation effortless and helpful."
            },
            {
                "id": "en-GB-LibbyNeural",
                "name": "Libby",
                "full_name": "Microsoft Libby Online (Natural) - English (United Kingdom)",
                "gender": "Female",
                "locale": "en-GB",
                "locale_name": "English (United Kingdom)",
                "accent": "UK",
                "category": "Adult Female",
                "age_group": "Adult",
                "style": "Polite, professional, polished British English",
                "engine": "edge_tts",
                "pitch_supported": True,
                "sample_text": "Good morning. My name is Libby. Please let me know how I can be of assistance."
            },
            {
                "id": "en-GB-SoniaNeural",
                "name": "Sonia",
                "full_name": "Microsoft Sonia Online (Natural) - English (United Kingdom)",
                "gender": "Female",
                "locale": "en-GB",
                "locale_name": "English (United Kingdom)",
                "accent": "UK",
                "category": "Adult Female",
                "age_group": "Adult",
                "style": "Calm, friendly, pleasant British English",
                "engine": "edge_tts",
                "pitch_supported": True,
                "sample_text": "Hello there! Sonia speaking. How may I direct your call today?"
            },
            {
                "id": "en-CA-ClaraNeural",
                "name": "Clara",
                "full_name": "Microsoft Clara Online (Natural) - English (Canada)",
                "gender": "Female",
                "locale": "en-CA",
                "locale_name": "English (Canada)",
                "accent": "CA",
                "category": "Adult Female",
                "age_group": "Adult",
                "style": "Friendly, authentic Canadian English",
                "engine": "edge_tts",
                "pitch_supported": True,
                "sample_text": "Hi, thanks for reaching out! Clara here. Let me know what you need today."
            },
            {
                "id": "en-AU-NatashaNeural",
                "name": "Natasha",
                "full_name": "Microsoft Natasha Online (Natural) - English (Australia)",
                "gender": "Female",
                "locale": "en-AU",
                "locale_name": "English (Australia)",
                "accent": "AU",
                "category": "Adult Female",
                "age_group": "Adult",
                "style": "Natural, welcoming Australian accent",
                "engine": "edge_tts",
                "pitch_supported": True,
                "sample_text": "G'day! Natasha here. How can I give you a hand today?"
            },
            {
                "id": "en-IE-EmilyNeural",
                "name": "Emily",
                "full_name": "Microsoft Emily Online (Natural) - English (Ireland)",
                "gender": "Female",
                "locale": "en-IE",
                "locale_name": "English (Ireland)",
                "accent": "IE",
                "category": "Adult Female",
                "age_group": "Adult",
                "style": "Gentle, lyrical, articulate Irish accent",
                "engine": "edge_tts",
                "pitch_supported": True,
                "sample_text": "Hello, welcome! I am Emily. It is a pleasure to speak with you today."
            },
            {
                "id": "hi-IN-SwaraNeural",
                "name": "Swara",
                "full_name": "Microsoft Swara Online (Natural) - Hindi (India)",
                "gender": "Female",
                "locale": "hi-IN",
                "locale_name": "Hindi (India)",
                "accent": "IN-Hindi",
                "category": "Adult Female",
                "age_group": "Adult",
                "style": "Natural, clear, articulate Hindi voice",
                "engine": "edge_tts",
                "pitch_supported": True,
                "sample_text": "नमस्ते! मैं स्वरा हूँ। आज मैं आपकी क्या सहायता कर सकती हूँ?"
            },

            # 👨 ADULT MALE VOICES (Conversational, Professional & Natural)
            {
                "id": "en-US-GuyNeural",
                "name": "Guy",
                "full_name": "Microsoft Guy Online (Natural) - English (United States)",
                "gender": "Male",
                "locale": "en-US",
                "locale_name": "English (United States)",
                "accent": "US",
                "category": "Adult Male",
                "age_group": "Adult",
                "style": "Casual, natural, confident everyday conversational voice",
                "engine": "edge_tts",
                "pitch_supported": True,
                "sample_text": "Hey there! Guy here. Glad to be connected with you. How can I help?"
            },
            {
                "id": "en-US-AndrewNeural",
                "name": "Andrew",
                "full_name": "Microsoft Andrew Online (Natural) - English (United States)",
                "gender": "Male",
                "locale": "en-US",
                "locale_name": "English (United States)",
                "accent": "US",
                "category": "Adult Male",
                "age_group": "Adult",
                "style": "Warm, confident, authentic, trustworthy tone",
                "engine": "edge_tts",
                "pitch_supported": True,
                "sample_text": "Hello! I am Andrew. I'm here to ensure you get accurate, quick assistance."
            },
            {
                "id": "en-US-BrianNeural",
                "name": "Brian",
                "full_name": "Microsoft Brian Online (Natural) - English (United States)",
                "gender": "Male",
                "locale": "en-US",
                "locale_name": "English (United States)",
                "accent": "US",
                "category": "Adult Male",
                "age_group": "Adult",
                "style": "Approachable, sincere, calm, relaxed voice",
                "engine": "edge_tts",
                "pitch_supported": True,
                "sample_text": "Hi, thanks for reaching out today. I am Brian, ready to assist."
            },
            {
                "id": "en-IN-PrabhatNeural",
                "name": "Prabhat",
                "full_name": "Microsoft Prabhat Online (Natural) - English (India)",
                "gender": "Male",
                "locale": "en-IN",
                "locale_name": "English (India)",
                "accent": "IN",
                "category": "Adult Male",
                "age_group": "Adult",
                "style": "Direct, confident, natural Indian English male voice",
                "engine": "edge_tts",
                "pitch_supported": True,
                "sample_text": "Hello! I am Prabhat. Please let me know how I can support your query today."
            },
            {
                "id": "en-GB-RyanNeural",
                "name": "Ryan",
                "full_name": "Microsoft Ryan Online (Natural) - English (United Kingdom)",
                "gender": "Male",
                "locale": "en-GB",
                "locale_name": "English (United Kingdom)",
                "accent": "UK",
                "category": "Adult Male",
                "age_group": "Adult",
                "style": "Engaging, conversational, friendly British male",
                "engine": "edge_tts",
                "pitch_supported": True,
                "sample_text": "Cheers for calling! I am Ryan. What can I do for you today?"
            },
            {
                "id": "en-GB-ThomasNeural",
                "name": "Thomas",
                "full_name": "Microsoft Thomas Online (Natural) - English (United Kingdom)",
                "gender": "Male",
                "locale": "en-GB",
                "locale_name": "English (United Kingdom)",
                "accent": "UK",
                "category": "Adult Male",
                "age_group": "Adult",
                "style": "Dignified, precise, formal British English",
                "engine": "edge_tts",
                "pitch_supported": True,
                "sample_text": "Good day. My name is Thomas. I will be pleased to assist with your requirements."
            },
            {
                "id": "en-CA-LiamNeural",
                "name": "Liam",
                "full_name": "Microsoft Liam Online (Natural) - English (Canada)",
                "gender": "Male",
                "locale": "en-CA",
                "locale_name": "English (Canada)",
                "accent": "CA",
                "category": "Adult Male",
                "age_group": "Adult",
                "style": "Energetic, clear Canadian English male",
                "engine": "edge_tts",
                "pitch_supported": True,
                "sample_text": "Hey there! Liam here. How's it going? Let me know how I can help."
            },
            {
                "id": "en-AU-WilliamMultilingualNeural",
                "name": "William",
                "full_name": "Microsoft William Online (Natural) - English (Australia)",
                "gender": "Male",
                "locale": "en-AU",
                "locale_name": "English (Australia)",
                "accent": "AU",
                "category": "Adult Male",
                "age_group": "Adult",
                "style": "Authentic Australian male, balanced and clear",
                "engine": "edge_tts",
                "pitch_supported": True,
                "sample_text": "Hello, William speaking. Ready to answer any questions you might have."
            },
            {
                "id": "en-IE-ConnorNeural",
                "name": "Connor",
                "full_name": "Microsoft Connor Online (Natural) - English (Ireland)",
                "gender": "Male",
                "locale": "en-IE",
                "locale_name": "English (Ireland)",
                "accent": "IE",
                "category": "Adult Male",
                "age_group": "Adult",
                "style": "Articulate, friendly Irish male accent",
                "engine": "edge_tts",
                "pitch_supported": True,
                "sample_text": "Hello there! Connor here from customer service. How can I help you today?"
            },
            {
                "id": "hi-IN-MadhurNeural",
                "name": "Madhur",
                "full_name": "Microsoft Madhur Online (Natural) - Hindi (India)",
                "gender": "Male",
                "locale": "hi-IN",
                "locale_name": "Hindi (India)",
                "accent": "IN-Hindi",
                "category": "Adult Male",
                "age_group": "Adult",
                "style": "Calm, articulate, natural Hindi male",
                "engine": "edge_tts",
                "pitch_supported": True,
                "sample_text": "नमस्कार! मैं मधुर हूँ। आपकी सहायता करने में मुझे खुशी होगी।"
            }
        ]

        categories = sorted(list(set(v["category"] for v in HUMAN_VOICES)))
        accents = sorted(list(set(v["accent"] for v in HUMAN_VOICES)))

        return {
            "total": len(HUMAN_VOICES),
            "categories": categories,
            "accents": accents,
            "voices": HUMAN_VOICES
        }

# Global singleton
tts_engine = None

def get_tts_engine() -> TTSEngine:
    global tts_engine
    if tts_engine is None:
        tts_engine = TTSEngine()
    return tts_engine
