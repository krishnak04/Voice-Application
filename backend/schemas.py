from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

class LatencyMetrics(BaseModel):
    audio_duration: float = Field(..., description="Duration of audio in seconds")
    stt_latency: float = Field(..., description="Faster-Whisper STT elapsed time in seconds")
    rtf: float = Field(..., description="Real-time factor: stt_latency / audio_duration")
    speed_factor: float = Field(..., description="Speed factor: audio_duration / stt_latency")
    groq_latency: Optional[float] = Field(0.0, description="Groq LLM response elapsed time in seconds")
    tts_latency: Optional[float] = Field(0.0, description="TTS synthesis elapsed time in seconds")
    total_server_time: Optional[float] = Field(0.0, description="Total pipeline latency on server")

class AccentDetectionResponse(BaseModel):
    accent: str = Field(..., description="Estimated English accent / variety name")
    code: str = Field(..., description="Locale / variety code (e.g. en-IN, en-US, en-GB, en-AU, en-CA, en-NZ, en-IE, en-ZA, en-OTHER, unknown)")
    flag: Optional[str] = Field("🌐", description="Flag emoji for variety")
    confidence: float = Field(..., description="Classification confidence (0.0 to 1.0)")
    status: str = Field(..., description="'detected', 'detecting', 'uncertain', 'insufficient_audio', or 'error'")
    sample_duration: float = Field(..., description="Duration of audio analyzed in seconds")
    model: str = Field(..., description="Audio classifier model used")
    license: Optional[str] = Field("MIT", description="Model open-source license")
    device: str = Field("cpu", description="Inference device (cpu or cuda)")
    latency_ms: float = Field(..., description="Elapsed accent detection latency in ms")
    inference_latency_ms: Optional[float] = Field(None, description="Pure model forward pass latency in ms")
    tts_voice: Optional[str] = Field(None, description="Mapped native Edge-TTS voice ID for this variety")
    details: Optional[str] = Field(None, description="Detailed explanation or status context")
    is_smoothed: Optional[bool] = Field(False, description="Whether result was stabilized using session smoothing")

class AccentInfoResponse(BaseModel):
    model_name: str
    license: str
    device: str
    classes: List[str]
    min_audio_seconds: float
    confidence_threshold: float
    status: str

class TranscriptionResponse(BaseModel):
    transcript: str = Field(..., description="Faster-Whisper speech transcript")
    language: Optional[str] = Field("en", description="Detected or requested language code")
    language_probability: Optional[float] = Field(None, description="Language detection confidence (0.0 to 1.0)")
    duration: float = Field(..., description="Duration of audio in seconds")
    stt_latency: float = Field(..., description="Whisper STT latency in seconds")
    rtf: float = Field(..., description="Real-time factor: stt_latency / duration")
    speed_factor: float = Field(..., description="Processing speed factor relative to real-time")
    device: str = Field(..., description="Inference device (cuda or cpu)")
    model_name: str = Field(..., description="Whisper model name used")
    accent_analysis: Optional[AccentDetectionResponse] = Field(None, description="Acoustic English accent variety analysis")

class ChatMessage(BaseModel):
    role: str = Field(..., description="Message role: 'user', 'assistant', or 'system'")
    content: str = Field(..., description="Message text content")

class ChatRequest(BaseModel):
    message: str = Field(..., description="User message/transcript to send to Groq LLM")
    history: Optional[List[ChatMessage]] = Field(default=None, description="Conversation history for phone call context")
    system_prompt: Optional[str] = Field(default=None, description="Optional custom system prompt")
    groq_api_key: Optional[str] = Field(None, description="Optional custom Groq API key")
    groq_model: Optional[str] = Field(None, description="Optional custom Groq model name")

class ChatResponse(BaseModel):
    reply: str = Field(..., description="Groq LLM generated text reply")
    groq_latency: float = Field(..., description="Groq API elapsed time in seconds")
    model: str = Field(..., description="Groq model name used")

class TTSRequest(BaseModel):
    text: str = Field(..., description="Text to synthesize to speech")
    voice: Optional[str] = Field(None, description="Voice ID (e.g., 'en-US-JennyNeural', 'en-IN-NeerjaNeural')")
    speed: Optional[float] = Field(1.0, description="Speech rate speed factor")
    pitch: Optional[str] = Field(None, description="Pitch adjustment (e.g., '+0Hz', '+15Hz', '-15Hz')")
    engine: Optional[str] = Field(None, description="TTS engine ('edge_tts', 'kokoro', 'piper')")

class TTSResponse(BaseModel):
    audio_base64: str = Field(..., description="Base64-encoded audio payload")
    format: str = Field(..., description="Audio format ('wav' or 'mp3')")
    media_type: str = Field(..., description="MIME media type")
    tts_latency: float = Field(..., description="TTS synthesis latency in seconds")
    duration: float = Field(..., description="Estimated audio duration in seconds")
    engine: str = Field(..., description="TTS engine used")
    voice: str = Field(..., description="Voice ID used")
    pitch: Optional[str] = Field("+0Hz", description="Pitch adjustment used")

class HumanVoiceModel(BaseModel):
    id: str = Field(..., description="Provider Voice ShortName ID")
    name: str = Field(..., description="Human display name")
    full_name: str = Field(..., description="Provider friendly name")
    gender: str = Field(..., description="Female or Male")
    locale: str = Field(..., description="Locale code (e.g. en-US, en-IN, hi-IN)")
    locale_name: str = Field(..., description="Human-readable country/language")
    accent: str = Field(..., description="Short regional accent code (US, UK, IN, CA, AU, etc.)")
    category: str = Field(..., description="Demographic category: Adult Female, Adult Male, Child / Youth")
    age_group: str = Field(..., description="Age demographic: Youth / Child, Adult")
    style: str = Field(..., description="Voice delivery personality: Conversational, Warm, Cheerful, Professional, etc.")
    engine: str = Field("edge_tts", description="TTS Provider Engine")
    pitch_supported: bool = Field(True, description="Whether pitch modulation is supported")
    sample_text: str = Field(..., description="Natural human preview sentence")

class HumanVoicesResponse(BaseModel):
    total: int
    categories: List[str]
    accents: List[str]
    voices: List[HumanVoiceModel]
    elderly_limitation_notice: Optional[str] = None

class VoiceResponse(BaseModel):
    transcript: str = Field(..., description="Faster-Whisper speech transcript")
    reply: str = Field(..., description="Groq LLM generated text reply")
    latency: LatencyMetrics = Field(..., description="Detailed latency metrics")
    device: str = Field(..., description="STT inference device (cuda or cpu)")
    model_name: str = Field(..., description="Whisper model name")
    accent_analysis: Optional[AccentDetectionResponse] = Field(None, description="Acoustic English accent variety analysis")

class CallTurnResponse(BaseModel):
    transcript: str = Field(..., description="User transcribed speech")
    reply: str = Field(..., description="AI conversational reply")
    audio_base64: str = Field(..., description="Base64-encoded audio of AI speech")
    audio_format: str = Field(..., description="Audio format ('wav' or 'mp3')")
    latency: LatencyMetrics = Field(..., description="End-to-end breakdown (STT, LLM, TTS, Total)")
    device: str = Field(..., description="STT inference device")
    whisper_model: str = Field(..., description="Whisper model name used")
    tts_engine: str = Field(..., description="TTS engine used")
    tts_voice: str = Field(..., description="Voice used for caller")
    accent_analysis: Optional[AccentDetectionResponse] = Field(None, description="Acoustic English accent variety analysis")

class HealthResponse(BaseModel):
    status: str
    whisper_model: str
    device: str
    compute_type: str
    loaded_models: Optional[list[str]] = Field(default_factory=list)
    groq_configured: bool
    groq_model: str
    tts_engine: Optional[str] = "none"
    tts_device: Optional[str] = "cpu"
    tts_voice: Optional[str] = "default"
    sherpa_available: Optional[bool] = False
    sherpa_model: Optional[str] = None
    sherpa_active_model: Optional[str] = None
    sherpa_type: Optional[str] = "streaming"
    sherpa_category: Optional[str] = "Standard"

class SherpaModelItem(BaseModel):
    id: str
    name: str
    language: str
    lang_code: str
    type: str
    category: str
    params: str
    dataset: str
    streaming_supported: bool
    offline_supported: bool
    is_downloaded: bool
    is_active: bool
    disk_size_mb: float
    license: str
    description: str

class SherpaModelsResponse(BaseModel):
    models: List[SherpaModelItem]
    active_model: str
    active_type: str
    active_category: str

class SherpaSelectRequest(BaseModel):
    model_id: str

class SherpaDownloadRequest(BaseModel):
    model_id: str

class WhisperModelItem(BaseModel):
    id: str
    name: str
    description: str
    params: str
    languages: str
    type: str
    category: str
    is_english_only: bool
    is_downloaded: bool
    downloaded: bool
    is_loaded: bool
    is_active: bool
    device: str
    compute_type: str

class WhisperSelectRequest(BaseModel):
    model_id: str

class STTSelectRequest(BaseModel):
    engine: str = Field(..., description="'whisper' or 'sherpa'")
    model_id: str = Field(..., description="Target model ID for the selected engine")

class STTModelsResponse(BaseModel):
    engines: List[str]
    active_engine: str
    whisper: Dict[str, Any]
    sherpa: Dict[str, Any]
