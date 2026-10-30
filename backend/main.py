import io
import time
import json
import base64
import logging
import uuid
import asyncio
import re
import numpy as np
from contextlib import asynccontextmanager
from typing import Optional, List
from collections import deque
from fastapi import FastAPI, UploadFile, File, Form, Query, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import config
from schemas import (
    TranscriptionResponse,
    ChatRequest,
    ChatResponse,
    VoiceResponse,
    LatencyMetrics,
    HealthResponse,
    TTSRequest,
    TTSResponse,
    CallTurnResponse,
    HumanVoicesResponse,
    SherpaModelsResponse,
    SherpaSelectRequest,
    SherpaDownloadRequest,
    WhisperSelectRequest,
    STTSelectRequest,
    STTModelsResponse,
    AccentDetectionResponse,
    AccentInfoResponse
)
from stt_service import get_stt_engine
from groq_service import get_groq_engine
from tts_service import get_tts_engine
from sherpa_service import get_sherpa_engine
from accent_service import get_accent_detector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("voice_api")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing Voice Server engines on startup...")
    stt = get_stt_engine()
    groq = get_groq_engine()
    tts = get_tts_engine()
    sherpa = get_sherpa_engine()
    logger.info(
        f"Ready: STT ({stt.default_model_name} on {stt.device}), "
        f"Sherpa ({sherpa.model_name if sherpa.is_available() else 'disabled'}), "
        f"Groq ({groq.model}), TTS ({tts.active_engine} on {tts.device})"
    )
    yield
    logger.info("Shutting down Voice API server...")

app = FastAPI(
    title="Real-Time Voice Calling & Speech API",
    description="Full-duplex voice pipeline combining Sherpa-ONNX streaming ASR, Faster-Whisper, Groq LLM, and Low-VRAM TTS.",
    version="3.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 1. Health & Server Info
# ==========================================
@app.get("/api/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    stt = get_stt_engine()
    groq = get_groq_engine()
    tts = get_tts_engine()
    sherpa = get_sherpa_engine()
    return HealthResponse(
        status="online",
        whisper_model=stt.default_model_name,
        device=stt.device,
        compute_type=stt.compute_type,
        loaded_models=stt.get_loaded_models(),
        groq_configured=bool(groq.api_key),
        groq_model=groq.model,
        tts_engine=tts.active_engine,
        tts_device=tts.device,
        tts_voice=tts.default_voice,
        sherpa_available=sherpa.is_available(),
        sherpa_model=sherpa.model_name if sherpa.is_available() else None,
        sherpa_active_model=sherpa.active_model_id if sherpa.is_available() else None,
        sherpa_type=sherpa.model_type if sherpa.is_available() else "streaming",
        sherpa_category=sherpa.model_category if sherpa.is_available() else "Standard"
    )

@app.get("/api/stt/models", tags=["Speech to Text"])
def get_all_stt_models(engine: Optional[str] = None):
    """Return verified models for both Whisper and Sherpa-ONNX engines."""
    stt = get_stt_engine()
    sherpa = get_sherpa_engine()
    whisper_data = {
        "engine": "whisper",
        "name": "Faster-Whisper (OpenAI Architecture)",
        "active_model": stt.default_model_name,
        "device": stt.device,
        "compute_type": stt.compute_type,
        "models": stt.list_models()
    }
    sherpa_data = {
        "engine": "sherpa",
        "name": "Sherpa-ONNX (Next-Gen Kaldi Zipformer)",
        "active_model": sherpa.active_model_id,
        "active_type": sherpa.model_type,
        "active_category": sherpa.model_category,
        "available": sherpa.is_available(),
        "models": sherpa.list_models()
    }
    if engine == "whisper":
        return whisper_data
    elif engine == "sherpa":
        return sherpa_data
    return {
        "engines": ["whisper", "sherpa"],
        "active_engine": "sherpa" if sherpa.is_available() and sherpa.is_streaming else "whisper",
        "whisper": whisper_data,
        "sherpa": sherpa_data,
        "whisper_models": whisper_data["models"],
        "sherpa_models": sherpa_data["models"]
    }

@app.post("/api/stt/select", tags=["Speech to Text"])
def select_stt_model(req: STTSelectRequest):
    """Unified STT model selector: safely switches either Whisper or Sherpa-ONNX model."""
    target_engine = (req.engine or "whisper").lower().strip()
    if target_engine == "whisper":
        stt = get_stt_engine()
        res = stt.select_model(req.model_id)
        if not res.get("success"):
            raise HTTPException(status_code=400, detail=res.get("message"))
        return {
            "success": True,
            "engine": "whisper",
            "model_id": res.get("active_model"),
            "message": res.get("message")
        }
    elif target_engine in ("sherpa", "sherpa-onnx"):
        sherpa = get_sherpa_engine()
        res = sherpa.switch_model(req.model_id)
        if not res.get("success"):
            raise HTTPException(status_code=400, detail=res.get("message"))
        return {
            "success": True,
            "engine": "sherpa",
            "model_id": sherpa.active_model_id,
            "message": res.get("message")
        }
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported STT engine: '{req.engine}'. Use 'whisper' or 'sherpa'.")

@app.get("/api/whisper/models", tags=["Speech to Text"])
def get_whisper_models():
    """List all verified Faster-Whisper models with loaded and active status."""
    stt = get_stt_engine()
    return {
        "models": stt.list_models(),
        "active_model": stt.default_model_name,
        "device": stt.device,
        "compute_type": stt.compute_type
    }

@app.post("/api/whisper/models/select", tags=["Speech to Text"])
def select_whisper_model(req: WhisperSelectRequest):
    """Safely switch and preload the active Faster-Whisper model."""
    stt = get_stt_engine()
    res = stt.select_model(req.model_id)
    if not res.get("success"):
        raise HTTPException(status_code=400, detail=res.get("message"))
    return res

@app.get("/api/sherpa/info", tags=["Speech to Text"])
def get_sherpa_info():
    """Return Sherpa-ONNX streaming engine metadata and configuration."""
    sherpa = get_sherpa_engine()
    return sherpa.get_info()

@app.get("/api/sherpa/models", tags=["Speech to Text"])
def get_sherpa_models():
    """List all verified English Sherpa-ONNX models with disk status, size category, and type."""
    sherpa = get_sherpa_engine()
    return {
        "models": sherpa.list_models(),
        "active_model": sherpa.active_model_id,
        "active_type": sherpa.model_type,
        "active_category": sherpa.model_category
    }

@app.post("/api/sherpa/models/select", tags=["Speech to Text"])
def select_sherpa_model(req: SherpaSelectRequest):
    """Safely switch the active Sherpa-ONNX model, validating files and active sessions."""
    sherpa = get_sherpa_engine()
    res = sherpa.switch_model(req.model_id)
    if not res.get("success"):
        raise HTTPException(status_code=400, detail=res.get("message"))
    return res

@app.post("/api/sherpa/models/download", tags=["Speech to Text"])
def download_sherpa_model(req: SherpaDownloadRequest):
    """Download and extract a verified English Sherpa-ONNX model from official releases."""
    sherpa = get_sherpa_engine()
    res = sherpa.download_model(req.model_id)
    if not res.get("success"):
        raise HTTPException(status_code=500, detail=res.get("message"))
    return res

@app.get("/api/sherpa/benchmark", tags=["Speech to Text"])
def get_sherpa_benchmark():
    """Return empirical benchmark comparisons for installed Sherpa-ONNX English models."""
    try:
        from benchmark_sherpa_models import run_benchmark
        results = run_benchmark()
        return {"status": "success", "benchmarks": results}
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ==========================================
# 2. Pure Voice-to-Text (STT) & Accent APIs
# ==========================================
@app.get("/api/accent/info", response_model=AccentInfoResponse, tags=["Accent Detection"])
def get_accent_info():
    """Return model architecture, candidate variety classes, and status of acoustic accent detector."""
    detector = get_accent_detector()
    return AccentInfoResponse(
        model_name=detector.model_name,
        license=detector.model_license,
        device=detector.device,
        classes=[
            "Indian English",
            "American English",
            "British English",
            "Australian English",
            "Canadian English",
            "New Zealand English",
            "Irish English",
            "South African English",
            "Other English variety",
            "Unknown / insufficient audio"
        ],
        min_audio_seconds=1.5,
        confidence_threshold=0.28,
        status="ready" if detector.is_available() else "unavailable"
    )

@app.post("/api/accent/detect", response_model=AccentDetectionResponse, tags=["Accent Detection"])
async def detect_accent_endpoint(
    audio: UploadFile = File(..., description="Audio recording (.wav, .webm, etc.)"),
    session_id: Optional[str] = Form(None, description="Optional WebSocket session ID for smoothing")
):
    """
    Automatic audio-based English accent / variety estimation.
    Acoustic classification using ECAPA-TDNN trained on Common Voice CommonAccent.
    """
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Audio file is empty")
    detector = get_accent_detector()
    return await detector.detect_accent_async(audio_bytes, session_id=session_id)

@app.post("/api/transcribe", response_model=TranscriptionResponse, tags=["Speech to Text"])
async def transcribe_audio(
    audio: UploadFile = File(..., description="Audio file to transcribe (.wav, .webm, .mp3, .m4a, .ogg)"),
    language: Optional[str] = Form(None, description="Language code (e.g. 'en', 'hi', 'auto')"),
    model: Optional[str] = Form(None, description="Model name (e.g. 'base', 'small', 'medium')"),
    engine: Optional[str] = Form(None, description="STT engine ('whisper' or 'sherpa')"),
    detect_accent: Optional[bool] = Form(True, description="Whether to also perform acoustic accent analysis")
):
    """Standalone Voice-to-Text endpoint supporting Faster-Whisper, Sherpa-ONNX, and decoupled accent estimation."""
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Audio payload is empty")

    filename = audio.filename or "audio.webm"
    ext = "." + filename.split(".")[-1] if "." in filename else ".webm"

    # Start decoupled accent detection concurrently if requested
    accent_task = None
    if detect_accent:
        detector = get_accent_detector()
        if detector.is_available():
            accent_task = asyncio.create_task(detector.detect_accent_async(audio_bytes))

    # Route based on engine selection or model prefix
    use_sherpa = (engine == "sherpa") or (model and (model.startswith("sherpa-") or model == "sherpa-streaming"))

    if use_sherpa:
        sherpa = get_sherpa_engine()
        if not sherpa.is_available():
            raise HTTPException(status_code=503, detail="Sherpa-ONNX engine is not available.")
        # If a specific sherpa model was passed and differs from active, switch or transcribe
        if model and model.startswith("sherpa-") and model != sherpa.active_model_id:
            sherpa.switch_model(model)
        stt_result = sherpa.transcribe_audio_source(audio_bytes)
    else:
        stt = get_stt_engine()
        stt_result = stt.transcribe_audio_bytes(
            audio_bytes=audio_bytes,
            file_ext=ext,
            language=language,
            model_name=model
        )

    # Collect accent detection result if task was launched
    accent_res = None
    if accent_task:
        try:
            accent_dict = await accent_task
            accent_res = AccentDetectionResponse(**accent_dict)
        except Exception as e:
            logger.warning(f"Decoupled accent detection warning: {e}")

    return TranscriptionResponse(
        transcript=stt_result["transcript"],
        language=stt_result["language"],
        language_probability=stt_result["language_probability"],
        duration=stt_result["duration"],
        stt_latency=stt_result["stt_latency"],
        rtf=stt_result["rtf"],
        speed_factor=stt_result["speed_factor"],
        device=stt_result["device"],
        model_name=stt_result["model_name"],
        accent_analysis=accent_res
    )

# ==========================================
# 3. Dedicated Groq LLM API
# ==========================================
@app.post("/api/chat", response_model=ChatResponse, tags=["Groq LLM"])
def chat_with_groq(req: ChatRequest):
    """Standalone Groq LLM text completion with conversation history support."""
    if not req.message or not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    history_dicts = [m.model_dump() for m in req.history] if req.history else None

    groq = get_groq_engine()
    groq_result = groq.generate_reply(
        user_text=req.message,
        api_key=req.groq_api_key,
        model_name=req.groq_model,
        history=history_dicts,
        system_prompt=req.system_prompt
    )

    return ChatResponse(
        reply=groq_result["reply"],
        groq_latency=groq_result["groq_latency"],
        model=groq_result["model"]
    )

# ==========================================
# 4. Text-to-Speech (TTS) APIs
# ==========================================
@app.post("/api/tts", response_model=TTSResponse, tags=["Text to Speech"])
def synthesize_speech(req: TTSRequest):
    """
    Standalone Text-to-Speech endpoint.
    Returns Base64 audio + latency metrics.
    """
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    tts = get_tts_engine()
    result = tts.synthesize(
        text=req.text,
        voice=req.voice,
        speed=req.speed or 1.0,
        pitch=req.pitch,
        engine=req.engine
    )

    b64_audio = base64.b64encode(result["audio_bytes"]).decode("utf-8")

    return TTSResponse(
        audio_base64=b64_audio,
        format=result["format"],
        media_type=result["media_type"],
        tts_latency=result["tts_latency"],
        duration=result["duration"],
        engine=result["engine"],
        voice=result["voice"],
        pitch=result.get("pitch", "+0Hz")
    )

@app.get("/api/tts/voices", response_model=HumanVoicesResponse, tags=["Text to Speech"])
def get_human_voices():
    """
    Return all verified, natural human voices categorized by demographics:
    Male, Female, Child / Youth, and Mature / Narrator, with pitch and accent support.
    """
    tts = get_tts_engine()
    return tts.get_human_voices()

@app.get("/api/tts/stream", tags=["Text to Speech"])
def stream_speech(
    text: str = Query(..., description="Text to synthesize"),
    voice: Optional[str] = Query(None, description="Voice ID"),
    speed: Optional[float] = Query(1.0, description="Speed factor"),
    pitch: Optional[str] = Query(None, description="Pitch modification string (e.g. +10Hz, -15Hz)")
):
    """Direct audio stream endpoint (plays in browser audio tag or VLC)."""
    tts = get_tts_engine()
    result = tts.synthesize(text=text, voice=voice, speed=speed or 1.0, pitch=pitch)
    return StreamingResponse(
        io.BytesIO(result["audio_bytes"]),
        media_type=result["media_type"]
    )

@app.get("/api/tts/info", tags=["Text to Speech"])
def get_tts_info():
    """Return active TTS engine details and available voices."""
    tts = get_tts_engine()
    return tts.get_info()

# ==========================================
# 5. Full Voice Calling Pipeline (Telephony Turn)
# Audio In -> STT -> Groq (Context) -> TTS -> Audio Out
# ==========================================
@app.post("/api/call/turn", response_model=CallTurnResponse, tags=["Voice Calling"])
async def voice_call_turn(
    audio: UploadFile = File(..., description="User voice audio recording (.webm or .wav)"),
    history: Optional[str] = Form(None, description="JSON string of past conversation messages"),
    language: Optional[str] = Form(None, description="Language code for STT"),
    whisper_model: Optional[str] = Form(None, description="Whisper model name"),
    stt_engine: Optional[str] = Form(None, description="STT engine ('whisper' or 'sherpa')"),
    groq_api_key: Optional[str] = Form(None, description="Groq API key"),
    groq_model: Optional[str] = Form(None, description="Groq model name"),
    tts_voice: Optional[str] = Form(None, description="TTS voice ID"),
    tts_pitch: Optional[str] = Form(None, description="TTS voice pitch"),
    tts_engine: Optional[str] = Form(None, description="TTS engine name")
):
    """
    Simulated Telephone Call Turn:
    1. Transcribes incoming caller audio with Faster-Whisper or Sherpa-ONNX.
    2. Passes transcript and history to Groq LLM with telephony prompt.
    3. Synthesizes AI voice with ultra-low VRAM TTS.
    4. Returns spoken voice audio, transcripts, and latency metrics in a single round-trip.
    """
    total_start = time.perf_counter()
    stt = get_stt_engine()
    groq = get_groq_engine()
    tts = get_tts_engine()
    sherpa = get_sherpa_engine()
    accent_detector = get_accent_detector()

    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Audio file is empty")

    accent_task = None
    if accent_detector.is_available():
        accent_task = asyncio.create_task(accent_detector.detect_accent_async(audio_bytes))

    filename = audio.filename or "call_audio.webm"
    ext = "." + filename.split(".")[-1] if "." in filename else ".webm"

    # Route STT to selected engine
    use_sherpa = (stt_engine == "sherpa") or (whisper_model and (whisper_model.startswith("sherpa-") or whisper_model == "sherpa-streaming"))
    if use_sherpa and sherpa.is_available():
        if whisper_model and whisper_model.startswith("sherpa-") and whisper_model != sherpa.active_model_id:
            sherpa.switch_model(whisper_model)
        stt_result = sherpa.transcribe_audio_source(audio_bytes)
    else:
        stt_result = stt.transcribe_audio_bytes(
            audio_bytes=audio_bytes,
            file_ext=ext,
            language=language,
            model_name=whisper_model
        )
    user_transcript = stt_result["transcript"].strip()

    # Parse conversation history
    history_list = []
    if history:
        try:
            history_list = json.loads(history)
            if not isinstance(history_list, list):
                history_list = []
        except Exception:
            history_list = []

    # 2. Groq LLM Phone Persona Reply & TTS
    if user_transcript:
        groq_result = groq.generate_reply(
            user_text=user_transcript,
            api_key=groq_api_key,
            model_name=groq_model,
            history=history_list,
            is_phone_call=True
        )
        ai_reply = groq_result["reply"]
        groq_latency = groq_result["groq_latency"]

        # 3. Low-VRAM Text-to-Speech
        tts_result = await tts.async_synthesize(
            text=ai_reply,
            voice=tts_voice,
            speed=1.05,
            engine=tts_engine
        )
        b64_audio = base64.b64encode(tts_result["audio_bytes"]).decode("utf-8")
        audio_format = tts_result["format"]
        tts_latency = tts_result["tts_latency"]
        tts_engine_name = tts_result["engine"]
        tts_voice_name = tts_result["voice"]
    else:
        # In continuous calling, if silence or ambient noise produced no speech,
        # return empty reply without synthesizing audio or interrupting caller.
        ai_reply = ""
        groq_latency = 0.0
        b64_audio = ""
        audio_format = "wav"
        tts_latency = 0.0
        tts_engine_name = tts.active_engine
        tts_voice_name = tts_voice or tts.default_voice

    total_server_time = round(time.perf_counter() - total_start, 3)

    latency = LatencyMetrics(
        audio_duration=stt_result["duration"],
        stt_latency=stt_result["stt_latency"],
        rtf=stt_result["rtf"],
        speed_factor=stt_result["speed_factor"],
        groq_latency=groq_latency,
        tts_latency=tts_latency,
        total_server_time=total_server_time
    )

    accent_res = None
    if accent_task:
        try:
            accent_dict = await accent_task
            accent_res = AccentDetectionResponse(**accent_dict)
        except Exception as e:
            logger.warning(f"Turn accent estimation error: {e}")

    return CallTurnResponse(
        transcript=user_transcript,
        reply=ai_reply,
        audio_base64=b64_audio,
        audio_format=audio_format,
        latency=latency,
        device=stt_result["device"],
        whisper_model=stt_result["model_name"],
        tts_engine=tts_engine_name,
        tts_voice=tts_voice_name,
        accent_analysis=accent_res
    )

# ==========================================
# 6. Backward-Compatible Benchmark Endpoint
# ==========================================
@app.post("/api/voice", response_model=VoiceResponse, tags=["Composite Pipeline"])
async def process_voice(
    audio: UploadFile = File(...),
    language: Optional[str] = Form(None),
    model: Optional[str] = Form(None),
    engine: Optional[str] = Form(None),
    groq_api_key: Optional[str] = Form(None),
    groq_model: Optional[str] = Form(None)
):
    """Legacy composite endpoint combining STT and Groq for benchmarking."""
    request_start = time.perf_counter()
    stt = get_stt_engine()
    groq = get_groq_engine()
    sherpa = get_sherpa_engine()
    accent_detector = get_accent_detector()

    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Audio payload is empty")

    accent_task = None
    if accent_detector.is_available():
        accent_task = asyncio.create_task(accent_detector.detect_accent_async(audio_bytes))

    filename = audio.filename or "recording.webm"
    ext = "." + filename.split(".")[-1] if "." in filename else ".webm"

    use_sherpa = (engine == "sherpa") or (model and (model.startswith("sherpa-") or model == "sherpa-streaming"))
    if use_sherpa and sherpa.is_available():
        if model and model.startswith("sherpa-") and model != sherpa.active_model_id:
            sherpa.switch_model(model)
        stt_result = sherpa.transcribe_audio_source(audio_bytes)
    else:
        stt_result = stt.transcribe_audio_bytes(
            audio_bytes=audio_bytes,
            file_ext=ext,
            language=language,
            model_name=model
        )
    transcript = stt_result["transcript"]

    if transcript.strip():
        groq_result = groq.generate_reply(
            user_text=transcript,
            api_key=groq_api_key,
            model_name=groq_model
        )
    else:
        groq_result = {
            "reply": "[No audible speech detected. Please speak closer to the microphone]",
            "groq_latency": 0.0,
            "model": groq_model or config.GROQ_MODEL
        }

    total_server_time = round(time.perf_counter() - request_start, 3)

    latency = LatencyMetrics(
        audio_duration=stt_result["duration"],
        stt_latency=stt_result["stt_latency"],
        rtf=stt_result["rtf"],
        speed_factor=stt_result["speed_factor"],
        groq_latency=groq_result["groq_latency"],
        tts_latency=0.0,
        total_server_time=total_server_time
    )

    accent_res = None
    if accent_task:
        try:
            accent_dict = await accent_task
            accent_res = AccentDetectionResponse(**accent_dict)
        except Exception as e:
            logger.warning(f"Voice benchmark accent estimation error: {e}")

    return VoiceResponse(
        transcript=transcript,
        reply=groq_result["reply"],
        latency=latency,
        device=stt_result["device"],
        model_name=stt_result["model_name"],
        accent_analysis=accent_res
    )

# ==========================================
# 7. Real-Time Full-Duplex WebSocket Endpoint
# Sherpa-ONNX Streaming ASR -> Groq LLM -> Chunked TTS -> Client Audio
# Supports continuous mic capture, partial transcription & barge-in interruption
# ==========================================
@app.websocket("/ws/call")
async def websocket_call_endpoint(websocket: WebSocket):
    """
    Real-Time Full-Duplex Conversational Voice WebSocket:
    - Accepts continuous 16kHz PCM16 audio chunks from browser microphone.
    - Emits real-time partial & final transcripts via Sherpa-ONNX streaming Zipformer.
    - Dispatches finalized user utterances to Groq LLM with conversation history.
    - Synthesizes incremental sentence-level audio chunks via TTS.
    - Supports user barge-in (interruption): immediately halts active response,
      notifies client to stop audio playback, and switches to new user turn.
    """
    await websocket.accept()
    session_id = f"call_{uuid.uuid4().hex[:8]}"
    sherpa = get_sherpa_engine()
    groq = get_groq_engine()
    tts = get_tts_engine()

    has_sherpa_stream = sherpa.is_available() and sherpa.is_streaming
    stream = None
    if has_sherpa_stream:
        stream = sherpa.create_stream(session_id)
        logger.info(f"[{session_id}] WebSocket call connected. Sherpa-ONNX stream established.")
    else:
        logger.info(f"[{session_id}] WebSocket call connected. Using Faster-Whisper / ASR engine.")

    # Session settings
    config_state = {
        "stt_engine": "whisper",
        "stt_model": "base.en",
        "stt_language": "auto",
        "tts_voice": "en-IN-NeerjaNeural",
        "tts_engine": "edge_tts",
        "tts_pitch": "+0Hz",
        "groq_model": config.GROQ_MODEL,
        "system_prompt": None,
        "is_phone_call": True,
        "tts_speed": 1.05,
        "auto_adapt_voice": True
    }

    accent_detector = get_accent_detector()
    accent_session = accent_detector.get_session(session_id)
    turn_audio_bytes = bytearray()
    pre_roll_chunks = deque(maxlen=8)  # ~1000ms pre-roll to catch soft voiceless initial consonants
    speech_started = False
    speech_start_time = None
    last_speech_time = None

    history: List[dict] = []
    active_turn_id = 0
    current_ai_task: Optional[asyncio.Task] = None
    is_ai_speaking = False
    ai_turn_start_time: float = 0.0
    last_partial = ""
    first_partial_sent = False
    last_first_partial_latency = None

    # Send initial handshake
    await websocket.send_json({
        "type": "ready",
        "session_id": session_id,
        "sample_rate": sherpa.sample_rate,
        "model": sherpa.model_name
    })

    async def cancel_active_ai_response(reason: str = "barge_in"):
        nonlocal current_ai_task, is_ai_speaking
        if current_ai_task and not current_ai_task.done():
            logger.info(f"[{session_id}] Cancelling active AI turn due to {reason}")
            current_ai_task.cancel()
            try:
                await current_ai_task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error(f"[{session_id}] Error cancelling AI task: {e}")
        is_ai_speaking = False
        current_ai_task = None

    async def execute_ai_turn(turn_id: int, user_text: str, t_endpoint: float, stt_metrics: dict = None):
        nonlocal is_ai_speaking
        is_ai_speaking = True
        t_llm_start = time.perf_counter()
        first_token_time = None
        first_audio_time = None
        tts_first_chunk_latency = None
        total_tts_time = 0.0
        chunk_idx = 0
        full_reply_text = ""

        # Pipelined concurrent TTS queue: allows Groq to stream unblocked
        # while TTS synthesizes ready clauses concurrently
        synth_queue = asyncio.Queue()
        tts_worker_task = None

        async def tts_worker():
            nonlocal first_audio_time, tts_first_chunk_latency, total_tts_time, chunk_idx
            try:
                while True:
                    item = await synth_queue.get()
                    if item is None:
                        synth_queue.task_done()
                        break

                    text_snippet, is_final_chunk = item
                    logger.info(f"[{session_id}] TTS worker synthesizing: '{text_snippet}' (engine={config_state['tts_engine']}, voice={config_state['tts_voice']})")
                    t_synth_start = time.perf_counter()
                    synth_res = await tts.async_synthesize(
                        text=text_snippet,
                        voice=config_state["tts_voice"],
                        speed=config_state["tts_speed"],
                        pitch=config_state.get("tts_pitch"),
                        engine=config_state["tts_engine"]
                    )
                    chunk_latency = synth_res.get("tts_latency", round(time.perf_counter() - t_synth_start, 3))
                    logger.info(f"[{session_id}] TTS synthesized {len(synth_res.get('audio_bytes', b''))} bytes in {chunk_latency}s")
                    total_tts_time += chunk_latency
                    if tts_first_chunk_latency is None:
                        tts_first_chunk_latency = chunk_latency
                    if first_audio_time is None:
                        first_audio_time = round(time.perf_counter() - t_endpoint, 3)

                    audio_b64 = base64.b64encode(synth_res["audio_bytes"]).decode("utf-8")
                    chunk_idx += 1
                    await websocket.send_json({
                        "type": "audio_chunk",
                        "turn_id": turn_id,
                        "chunk_index": chunk_idx,
                        "audio_base64": audio_b64,
                        "format": synth_res["format"],
                        "text_snippet": text_snippet,
                        "is_final": is_final_chunk,
                        "metrics": {
                            "time_to_first_token_ms": int((first_token_time or 0) * 1000),
                            "time_to_first_audio_ms": int((first_audio_time or 0) * 1000),
                            "tts_chunk_latency_ms": int(chunk_latency * 1000)
                        }
                    })
                    synth_queue.task_done()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"[{session_id}] TTS worker error: {e}")

        tts_worker_task = asyncio.create_task(tts_worker())

        try:
            sentence_buffer = ""
            async for chunk in groq.async_stream_reply(
                user_text=user_text,
                model_name=config_state["groq_model"],
                history=history,
                system_prompt=config_state["system_prompt"],
                is_phone_call=config_state["is_phone_call"]
            ):
                delta = chunk.get("delta", "")
                is_final = chunk.get("is_final", False)

                if delta:
                    if first_token_time is None:
                        first_token_time = round(time.perf_counter() - t_llm_start, 3)

                    # Send text delta to client UI immediately
                    await websocket.send_json({
                        "type": "assistant_text_delta",
                        "turn_id": turn_id,
                        "delta": delta,
                        "time_to_first_token_ms": int(first_token_time * 1000) if first_token_time else None
                    })

                    sentence_buffer += delta
                    full_reply_text += delta

                    # Optimized clause & sentence boundary matching
                    parts = re.split(r'([.?!;\n]+|,\s+|:\s+)', sentence_buffer)
                    if len(parts) > 2:
                        candidate_phrase = (parts[0] + parts[1]).strip()
                        word_count = len(candidate_phrase.split())
                        is_terminal = any(candidate_phrase.endswith(p) for p in ('.', '?', '!', ';', '\n'))
                        is_clause = word_count >= 4

                        if is_terminal or is_clause:
                            ready_sentence = candidate_phrase
                            sentence_buffer = "".join(parts[2:]).strip()
                            await synth_queue.put((ready_sentence, False))

                if is_final:
                    break

            # Synthesize any remaining sentence buffer
            remaining_text = sentence_buffer.strip()
            if remaining_text:
                await synth_queue.put((remaining_text, True))

            # Signal worker that all sentences are queued and wait for completion
            await synth_queue.put(None)
            await tts_worker_task

            clean_reply = full_reply_text.strip()
            if not clean_reply and user_text.strip():
                clean_reply = "I'm sorry, I didn't quite catch that. Could you please repeat?"
                try:
                    synth_res = await tts.async_synthesize(
                        text=clean_reply,
                        voice=config_state["tts_voice"],
                        speed=config_state["tts_speed"],
                        pitch=config_state.get("tts_pitch"),
                        engine=config_state["tts_engine"]
                    )
                    chunk_latency = synth_res.get("tts_latency", 0.3)
                    total_tts_time += chunk_latency
                    if tts_first_chunk_latency is None:
                        tts_first_chunk_latency = chunk_latency
                    if first_audio_time is None:
                        first_audio_time = round(time.perf_counter() - t_endpoint, 3)

                    audio_b64 = base64.b64encode(synth_res["audio_bytes"]).decode("utf-8")
                    chunk_idx += 1
                    await websocket.send_json({
                        "type": "audio_chunk",
                        "turn_id": turn_id,
                        "chunk_index": chunk_idx,
                        "audio_base64": audio_b64,
                        "format": synth_res["format"],
                        "text_snippet": clean_reply,
                        "is_final": True,
                        "metrics": {
                            "time_to_first_token_ms": int((first_token_time or 0) * 1000),
                            "time_to_first_audio_ms": int((first_audio_time or 0) * 1000),
                            "tts_chunk_latency_ms": int(chunk_latency * 1000)
                        }
                    })
                except Exception as synth_err:
                    logger.warning(f"[{session_id}] Fallback synthesis warning: {synth_err}")

            if user_text and clean_reply:
                history.append({"role": "user", "content": user_text})
                history.append({"role": "assistant", "content": clean_reply})

            t_turn_complete = time.perf_counter()
            total_elapsed = round(t_turn_complete - t_endpoint, 3)
            groq_completion_ms = round((t_turn_complete - t_llm_start) * 1000)
            time_to_first_audio_ms = int(first_audio_time * 1000) if first_audio_time else None
            tts_total_ms = round(total_tts_time * 1000)
            voice_to_voice_rtt_ms = time_to_first_audio_ms or int(total_elapsed * 1000)

            await websocket.send_json({
                "type": "turn_complete",
                "turn_id": turn_id,
                "full_reply": clean_reply,
                "total_chunks": chunk_idx,
                "metrics": {
                    "first_partial_latency_ms": stt_metrics.get("first_partial_latency_ms") if stt_metrics else None,
                    "endpoint_latency_ms": stt_metrics.get("endpoint_latency_ms") if stt_metrics else None,
                    "audio_duration_sec": stt_metrics.get("audio_duration_sec") if stt_metrics else None,
                    "rtf": stt_metrics.get("rtf") if stt_metrics else None,
                    "time_to_first_token_ms": int(first_token_time * 1000) if first_token_time else None,
                    "groq_completion_ms": groq_completion_ms,
                    "time_to_first_audio_ms": time_to_first_audio_ms,
                    "tts_first_chunk_ms": int(tts_first_chunk_latency * 1000) if tts_first_chunk_latency else None,
                    "tts_total_ms": tts_total_ms,
                    "total_turn_latency_ms": int(total_elapsed * 1000),
                    "voice_to_voice_rtt_ms": voice_to_voice_rtt_ms
                }
            })

        except asyncio.CancelledError:
            logger.info(f"[{session_id}] AI turn {turn_id} cancelled.")
            if tts_worker_task and not tts_worker_task.done():
                tts_worker_task.cancel()
            raise
        except Exception as e:
            logger.error(f"[{session_id}] AI turn error: {e}")
            if tts_worker_task and not tts_worker_task.done():
                tts_worker_task.cancel()
            await websocket.send_json({
                "type": "error",
                "turn_id": turn_id,
                "message": f"AI response error: {str(e)}"
            })
        finally:
            is_ai_speaking = False

    try:
        while True:
            message = await websocket.receive()

            if "bytes" in message and message["bytes"]:
                raw_bytes = message["bytes"]
                audio_np = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                rms = float(np.sqrt(np.mean(audio_np**2))) if len(audio_np) > 0 else 0.0
                now_t = time.perf_counter()

                # Speech activity tracking: only buffer and start turn timer when user actually speaks
                if not speech_started:
                    pre_roll_chunks.append(raw_bytes)
                    if rms > 0.005:
                        speech_started = True
                        speech_start_time = now_t
                        last_speech_time = now_t
                        for pb in pre_roll_chunks:
                            turn_audio_bytes.extend(pb)
                        pre_roll_chunks.clear()
                else:
                    turn_audio_bytes.extend(raw_bytes)
                    if rms > 0.005:
                        last_speech_time = now_t

                # Feed audio chunk to Sherpa streaming recognizer if active
                partial = ""
                if has_sherpa_stream:
                    sherpa.feed_audio(session_id, audio_np)
                    partial = sherpa.get_partial_transcript(session_id)

                if partial:
                    if not speech_started:
                        speech_started = True
                        speech_start_time = now_t
                        for pb in pre_roll_chunks:
                            turn_audio_bytes.extend(pb)
                        pre_roll_chunks.clear()
                    last_speech_time = now_t

                if partial and partial != last_partial:
                    last_partial = partial
                    if not first_partial_sent:
                        first_partial_latency = int((now_t - (speech_start_time or now_t)) * 1000)
                        last_first_partial_latency = first_partial_latency
                        first_partial_sent = True
                    else:
                        first_partial_latency = None

                    await websocket.send_json({
                        "type": "partial_transcript",
                        "text": partial,
                        "turn_id": active_turn_id,
                        "first_partial_latency_ms": first_partial_latency
                    })

                    # Barge-in: if AI is speaking and user speaks substantial fresh utterance
                    if is_ai_speaking or (current_ai_task and not current_ai_task.done()):
                        time_since_ai_start = now_t - ai_turn_start_time
                        if time_since_ai_start > 0.3:
                            words = partial.strip().split()
                            if len(words) >= 2 or len(partial.strip()) >= 5:
                                interrupt_start = time.perf_counter()
                                await cancel_active_ai_response("user_spoken_partial")
                                interrupt_latency_ms = round((time.perf_counter() - interrupt_start) * 1000, 2)
                                await websocket.send_json({
                                    "type": "interrupted",
                                    "turn_id": active_turn_id,
                                    "reason": "user_speech_detected",
                                    "interruption_latency_ms": interrupt_latency_ms,
                                    "partial_text": partial
                                })

                # Endpoint detection: tuned conversational trailing silence (1.2s)
                has_speech = bool(last_partial.strip()) or speech_started
                is_vad_endpoint = (sherpa.is_endpoint(session_id) and bool(last_partial.strip())) if has_sherpa_stream else False
                trailing_silence_threshold = getattr(config, "SHERPA_RULE2_MIN_TRAILING_SILENCE", 1.2)
                has_trailing_silence = (
                    has_speech
                    and last_speech_time is not None
                    and (now_t - last_speech_time >= trailing_silence_threshold)
                    and not is_ai_speaking
                )

                if (is_vad_endpoint or has_trailing_silence) and len(turn_audio_bytes) > 0:
                    active_engine = config_state.get("stt_engine", "whisper").lower()
                    active_model = config_state.get("stt_model")
                    target_lang = config_state.get("stt_language")
                    utterance_audio = bytes(turn_audio_bytes)

                    final_text = ""
                    # 1. Faster-Whisper turn finalization
                    is_whisper = (active_engine == "whisper") or (active_model and not str(active_model).startswith("sherpa"))
                    if is_whisper:
                        stt = get_stt_engine()
                        whisper_res = stt.transcribe_audio_bytes(
                            audio_bytes=utterance_audio,
                            file_ext=".wav",
                            language=target_lang if target_lang not in ("auto", "none", None) else None,
                            model_name=active_model
                        )
                        final_text = whisper_res.get("transcript", "").strip()
                        if not final_text and last_partial and len(last_partial.strip().split()) >= 2:
                            final_text = last_partial.strip()
                    else:
                        # 2. Sherpa-ONNX model finalization (streaming or offline)
                        sherpa = get_sherpa_engine()
                        if active_model and str(active_model).startswith("sherpa-") and active_model != sherpa.active_model_id:
                            sherpa.switch_model(active_model)

                        if sherpa.is_streaming:
                            # Streaming Zipformer with lookahead silence flush
                            final_text = sherpa.get_final_transcript(session_id).strip()
                        else:
                            # Offline Sherpa model (GigaSpeech, Large, Small)
                            sherpa_res = sherpa.transcribe_audio_source(utterance_audio)
                            final_text = sherpa_res.get("transcript", "").strip()

                        if not final_text and last_partial and len(last_partial.strip().split()) >= 2:
                            final_text = last_partial.strip()

                    if sherpa.is_available() and sherpa.is_streaming:
                        sherpa.reset_stream(session_id)
                    last_partial = ""
                    first_partial_sent = False

                    if final_text:
                        t_endpoint = time.perf_counter()
                        utterance_audio = bytes(turn_audio_bytes)
                        audio_duration_sec = round(len(utterance_audio) / 32000.0, 3)
                        utt_np = np.frombuffer(utterance_audio, dtype=np.int16).astype(np.float32) / 32768.0 if len(utterance_audio) > 0 else np.zeros(0, dtype=np.float32)
                        peak_val = float(np.max(np.abs(utt_np))) if len(utt_np) > 0 else 0.0
                        peak_dbfs = round(20 * float(np.log10(max(peak_val, 1e-5))), 1)
                        rms_val = round(float(np.sqrt(np.mean(utt_np**2))), 4) if len(utt_np) > 0 else 0.0
                        # Real endpointing delay is from speech completion to endpoint declaration
                        endpoint_latency_ms = round((t_endpoint - (last_speech_time or speech_start_time or t_endpoint)) * 1000, 1)
                        stt_rtf = round((endpoint_latency_ms / 1000.0) / max(audio_duration_sec, 0.001), 3) if audio_duration_sec > 0 else None

                        # Reset turn trackers for next utterance
                        speech_started = False
                        speech_start_time = None
                        last_speech_time = None
                        turn_audio_bytes.clear()
                        pre_roll_chunks.clear()

                        active_turn_id += 1
                        turn_id = active_turn_id
                        logger.info(
                            f"[{session_id}] Sherpa finalized turn {turn_id}: '{final_text}' "
                            f"(audio={audio_duration_sec}s, peak={peak_dbfs} dBFS, rms={rms_val}, "
                            f"endpoint={endpoint_latency_ms}ms, vad={is_vad_endpoint}, silence={has_trailing_silence})"
                        )

                        stt_metrics = {
                            "first_partial_latency_ms": last_first_partial_latency,
                            "endpoint_latency_ms": endpoint_latency_ms,
                            "audio_duration_sec": audio_duration_sec,
                            "rtf": stt_rtf
                        }

                        # Reset partial trackers for next turn
                        last_first_partial_latency = None
                        first_partial_sent = False

                        await websocket.send_json({
                            "type": "final_transcript",
                            "text": final_text,
                            "turn_id": turn_id,
                            "endpoint_latency_ms": endpoint_latency_ms,
                            "audio_duration_sec": audio_duration_sec,
                            "rtf": stt_rtf
                        })

                        # Cancel any previous active task
                        await cancel_active_ai_response("new_turn_start")

                        # Spawn assistant response task immediately
                        ai_turn_start_time = time.perf_counter()
                        current_ai_task = asyncio.create_task(
                            execute_ai_turn(turn_id, final_text, t_endpoint, stt_metrics)
                        )

                        # Concurrently run automatic accent estimation in the background
                        async def process_turn_accent(audio_chunk: bytes, turn_num: int):
                            try:
                                acc_res = await accent_detector.detect_accent_async(audio_chunk, session_id=session_id)
                                logger.info(f"[{session_id}] Accent turn {turn_num}: {acc_res['accent']} ({int(acc_res['confidence']*100)}%, {acc_res['status']})")
                                await websocket.send_json({
                                    "type": "accent_detected",
                                    "turn_id": turn_num,
                                    "accent": acc_res["accent"],
                                    "code": acc_res["code"],
                                    "flag": acc_res.get("flag", "🌐"),
                                    "confidence": acc_res["confidence"],
                                    "status": acc_res["status"],
                                    "sample_duration": acc_res["sample_duration"],
                                    "latency_ms": acc_res["latency_ms"],
                                    "suggested_tts_voice": acc_res.get("tts_voice"),
                                    "details": acc_res.get("details", ""),
                                    "is_smoothed": acc_res.get("is_smoothed", False)
                                })

                                # Auto voice adaptation if enabled and accent was reliably detected
                                if config_state.get("auto_adapt_voice", True) and acc_res["status"] == "detected":
                                    target_voice = acc_res.get("tts_voice")
                                    if target_voice and target_voice != config_state["tts_voice"]:
                                        old_voice = config_state["tts_voice"]
                                        config_state["tts_voice"] = target_voice
                                        logger.info(f"[{session_id}] Auto-adapted response voice: {old_voice} -> {target_voice} ({acc_res['accent']})")
                                        await websocket.send_json({
                                            "type": "voice_adapted",
                                            "voice": target_voice,
                                            "accent": acc_res["accent"],
                                            "flag": acc_res.get("flag", "🌐")
                                        })
                            except Exception as e:
                                logger.warning(f"[{session_id}] Accent detection background error: {e}")

                        if utterance_audio:
                            asyncio.create_task(process_turn_accent(utterance_audio, turn_id))

            elif "text" in message and message["text"]:
                data = json.loads(message["text"])
                msg_type = data.get("type")

                if msg_type == "config":
                    if "tts_voice" in data: config_state["tts_voice"] = data["tts_voice"]
                    if "tts_engine" in data: config_state["tts_engine"] = data["tts_engine"]
                    if "tts_pitch" in data: config_state["tts_pitch"] = data["tts_pitch"]
                    if "groq_model" in data: config_state["groq_model"] = data["groq_model"]
                    if "system_prompt" in data: config_state["system_prompt"] = data["system_prompt"]
                    if "tts_speed" in data: config_state["tts_speed"] = float(data["tts_speed"])
                    if "auto_adapt_voice" in data: config_state["auto_adapt_voice"] = bool(data["auto_adapt_voice"])
                    if "stt_engine" in data: config_state["stt_engine"] = data["stt_engine"]
                    if "stt_model" in data:
                        new_m = data["stt_model"]
                        config_state["stt_model"] = new_m
                        if new_m and str(new_m).startswith("sherpa-"):
                            try:
                                sherpa = get_sherpa_engine()
                                if new_m != sherpa.active_model_id:
                                    sherpa.switch_model(new_m)
                            except Exception as sm_err:
                                logger.warning(f"Failed to switch Sherpa model to {new_m}: {sm_err}")
                        elif new_m:
                            try:
                                stt = get_stt_engine()
                                stt.select_model(new_m)
                            except Exception as wm_err:
                                logger.warning(f"Failed to switch Whisper model to {new_m}: {wm_err}")
                    if "stt_language" in data: config_state["stt_language"] = data["stt_language"]
                    logger.info(f"[{session_id}] Updated config: {config_state}")

                elif msg_type == "set_auto_adapt_voice":
                    config_state["auto_adapt_voice"] = bool(data.get("enabled", True))
                    await websocket.send_json({
                        "type": "auto_adapt_voice_changed",
                        "enabled": config_state["auto_adapt_voice"]
                    })

                elif msg_type == "interrupt":
                    interrupt_start = time.perf_counter()
                    time_since_ai_start = interrupt_start - ai_turn_start_time
                    if is_ai_speaking or (current_ai_task and not current_ai_task.done()):
                        await cancel_active_ai_response("client_interrupt_message")
                        interrupt_latency_ms = round((time.perf_counter() - interrupt_start) * 1000, 2)
                        await websocket.send_json({
                            "type": "interrupted",
                            "turn_id": active_turn_id,
                            "reason": "client_barge_in",
                            "interruption_latency_ms": interrupt_latency_ms
                        })
                    else:
                        await websocket.send_json({
                            "type": "interrupted",
                            "turn_id": active_turn_id,
                            "reason": "already_idle",
                            "interruption_latency_ms": 0.0
                        })

                elif msg_type == "clear_history":
                    history.clear()
                    await websocket.send_json({"type": "history_cleared"})

                elif msg_type == "ping":
                    await websocket.send_json({"type": "pong"})

            elif message.get("type") == "websocket.disconnect":
                break

    except WebSocketDisconnect:
        logger.info(f"[{session_id}] Client disconnected.")
    except Exception as e:
        logger.error(f"[{session_id}] WebSocket error: {e}")
    finally:
        await cancel_active_ai_response("session_closed")
        sherpa.close_stream(session_id)
        accent_detector.clear_session(session_id)
        logger.info(f"[{session_id}] Cleaned up session stream and accent session.")

