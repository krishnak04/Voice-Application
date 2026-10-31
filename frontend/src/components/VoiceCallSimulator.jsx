import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import {
  Phone,
  PhoneOff,
  Mic,
  MicOff,
  Volume2,
  VolumeX,
  Play,
  Pause,
  Square,
  RotateCcw,
  Sparkles,
  User,
  Bot,
  Activity,
  Sliders,
  Settings2,
  Radio,
  Globe,
  Cpu,
  Layers,
  Copy,
  Check,
  Download,
  UploadCloud,
  FileAudio,
  Wand2,
  Clock,
  Zap,
  FileText
} from 'lucide-react';
import {
  sendVoiceCallTurn,
  playAudioFromBase64,
  stopAudioPlayback,
  synthesizeSpeech,
  transcribeAudio,
  getWebSocketUrl,
  getFallbackWebSocketUrl,
  getHumanVoices
} from '../services/api';
import {
  MODELS as STT_MODELS,
  LANGUAGES as STT_LANGUAGES,
  WHISPER_MODELS,
  SHERPA_MODELS
} from './AudioRecorder';

export const EXTENDED_STT_MODELS = [
  ...SHERPA_MODELS.map(m => ({ id: m.id, label: `${m.type === 'streaming' ? '⚡' : '📦'} [Sherpa] ${m.label}` })),
  ...WHISPER_MODELS.map(m => ({ id: m.id, label: `🎙️ [Whisper] ${m.label}` }))
];

const DEEPGRAM_MODELS = [{ id: 'nova-3', label: 'Nova-3 Real-Time', status: 'Cloud' }];

export const TTS_MODELS = [
  {
    id: 'auto',
    name: 'Auto-Select (Smart Engine)',
    desc: 'Automatically chooses engine based on selected voice',
    badge: 'Smart'
  },
  {
    id: 'edge_tts',
    name: 'Edge-TTS Neural (Zero VRAM)',
    desc: 'Microsoft Cloud Neural with rich Indian & English accents (0 VRAM)',
    badge: 'Cloud Neural'
  },
  {
    id: 'kokoro',
    name: 'Kokoro-82M (Local Neural)',
    desc: 'On-device ~300MB VRAM, ultra-natural US & UK voices',
    badge: 'Local GPU/CPU'
  },
  {
    id: 'piper',
    name: 'Piper TTS (Fast Telephony)',
    desc: 'Ultra-fast local CPU telephony synthesis (0 GPU)',
    badge: 'Offline CPU'
  }
];

export const PITCH_OPTIONS = [
  { label: 'Deep (-20Hz)', value: '-20Hz' },
  { label: 'Low (-10Hz)', value: '-10Hz' },
  { label: 'Natural (+0Hz)', value: '+0Hz' },
  { label: 'High (+10Hz)', value: '+10Hz' },
  { label: 'Bright (+20Hz)', value: '+20Hz' }
];

export const VOICE_CATEGORIES = [
  'All',
  'Adult Female',
  'Adult Male',
  'Child / Youth'
];

export const VOICE_ACCENTS = [
  { id: 'All', label: 'All Accents' },
  { id: 'US', label: 'United States 🇺🇸' },
  { id: 'UK', label: 'United Kingdom 🇬🇧' },
  { id: 'IN', label: 'India (English) 🇮🇳' },
  { id: 'IN-Hindi', label: 'India (Hindi) 🇮🇳' },
  { id: 'CA', label: 'Canada 🇨🇦' },
  { id: 'AU', label: 'Australia 🇦🇺' },
  { id: 'IE', label: 'Ireland 🇮🇪' }
];

// Curated 33 Real Human Voices from Edge-TTS (Zero synthetic/animal/robot voices)
export const CALL_VOICES = [
  // --- CHILD / YOUTH HUMAN VOICES ---
  {
    id: 'en-US-AnaNeural',
    name: 'Ana',
    gender: 'Female',
    locale: 'en-US',
    accent: 'US',
    category: 'Child / Youth',
    age_group: 'Child / Youth',
    style: 'Youthful, lively, friendly American girl tone',
    engine: 'edge_tts',
    pitch_supported: true,
    sample_text: "Hi there! My name is Ana. I'm super happy to talk to you today!"
  },
  {
    id: 'en-GB-MaisieNeural',
    name: 'Maisie',
    gender: 'Female',
    locale: 'en-GB',
    accent: 'UK',
    category: 'Child / Youth',
    age_group: 'Child / Youth',
    style: 'Bright, cheerful, polite British youth voice',
    engine: 'edge_tts',
    pitch_supported: true,
    sample_text: "Hello! I am Maisie. I really love learning new things and helping out!"
  },

  // --- ADULT FEMALE HUMAN VOICES ---
  {
    id: 'en-US-JennyNeural',
    name: 'Jenny',
    gender: 'Female',
    locale: 'en-US',
    accent: 'US',
    category: 'Adult Female',
    age_group: 'Adult',
    style: 'Natural, friendly, conversational US assistant',
    engine: 'edge_tts',
    pitch_supported: true,
    sample_text: "Hello! I'm Jenny. How can I assist you with your call today?"
  },
  {
    id: 'en-US-AriaNeural',
    name: 'Aria',
    gender: 'Female',
    locale: 'en-US',
    accent: 'US',
    category: 'Adult Female',
    age_group: 'Adult',
    style: 'Expressive, dynamic, confident customer service voice',
    engine: 'edge_tts',
    pitch_supported: true,
    sample_text: "Welcome to our support line! My name is Aria. What can I help you find?"
  },
  {
    id: 'en-US-MichelleNeural',
    name: 'Michelle',
    gender: 'Female',
    locale: 'en-US',
    accent: 'US',
    category: 'Adult Female',
    age_group: 'Adult',
    style: 'Warm, empathetic, professional female narrator',
    engine: 'edge_tts',
    pitch_supported: true,
    sample_text: "Hi, this is Michelle. I am here to make sure everything runs smoothly for you."
  },
  {
    id: 'en-GB-SoniaNeural',
    name: 'Sonia',
    gender: 'Female',
    locale: 'en-GB',
    accent: 'UK',
    category: 'Adult Female',
    age_group: 'Adult',
    style: 'Polished, clear, neutral British English executive',
    engine: 'edge_tts',
    pitch_supported: true,
    sample_text: "Good day. Sonia speaking. I am pleased to assist you with your request."
  },
  {
    id: 'en-GB-LibbyNeural',
    name: 'Libby',
    gender: 'Female',
    locale: 'en-GB',
    accent: 'UK',
    category: 'Adult Female',
    age_group: 'Adult',
    style: 'Approachable, warm, conversational British tone',
    engine: 'edge_tts',
    pitch_supported: true,
    sample_text: "Hi there! I am Libby. Let me know what you need and we can sort it out."
  },
  {
    id: 'en-IN-NeerjaNeural',
    name: 'Neerja',
    gender: 'Female',
    locale: 'en-IN',
    accent: 'IN',
    category: 'Adult Female',
    age_group: 'Adult',
    style: 'Warm, expressive, authentic Indian English female',
    engine: 'edge_tts',
    pitch_supported: true,
    sample_text: "Namaste! Thank you for calling customer care. How may I help you today?"
  },
  {
    id: 'en-IN-KavyaNeural',
    name: 'Kavya',
    gender: 'Female',
    locale: 'en-IN',
    accent: 'IN',
    category: 'Adult Female',
    age_group: 'Adult',
    style: 'Articulate, courteous, customer-centric Indian voice',
    engine: 'edge_tts',
    pitch_supported: true,
    sample_text: "Hello! My name is Kavya. I will be glad to resolve your query right away."
  },
  {
    id: 'en-IN-AnanyaNeural',
    name: 'Ananya',
    gender: 'Female',
    locale: 'en-IN',
    accent: 'IN',
    category: 'Adult Female',
    age_group: 'Adult',
    style: 'Friendly, modern, clear urban Indian English',
    engine: 'edge_tts',
    pitch_supported: true,
    sample_text: "Hi! Ananya here. I am ready to guide you step by step through your account."
  },
  {
    id: 'en-CA-ClaraNeural',
    name: 'Clara',
    gender: 'Female',
    locale: 'en-CA',
    accent: 'CA',
    category: 'Adult Female',
    age_group: 'Adult',
    style: 'Calm, crisp, authentic Canadian English female',
    engine: 'edge_tts',
    pitch_supported: true,
    sample_text: "Hello, Clara here! I hope you are having a wonderful day. How can I help?"
  },
  {
    id: 'en-AU-NatashaNeural',
    name: 'Natasha',
    gender: 'Female',
    locale: 'en-AU',
    accent: 'AU',
    category: 'Adult Female',
    age_group: 'Adult',
    style: 'Natural Australian female, upbeat and approachable',
    engine: 'edge_tts',
    pitch_supported: true,
    sample_text: "G'day! Natasha here. What can I help you out with today?"
  },
  {
    id: 'en-IE-EmilyNeural',
    name: 'Emily',
    gender: 'Female',
    locale: 'en-IE',
    accent: 'IE',
    category: 'Adult Female',
    age_group: 'Adult',
    style: 'Friendly, melodic, professional Irish female',
    engine: 'edge_tts',
    pitch_supported: true,
    sample_text: "Welcome! My name is Emily. I will be delighted to help you out today."
  },
  {
    id: 'hi-IN-SwaraNeural',
    name: 'Swara',
    gender: 'Female',
    locale: 'hi-IN',
    accent: 'IN-Hindi',
    category: 'Adult Female',
    age_group: 'Adult',
    style: 'Clear, natural, professional Hindi female',
    engine: 'edge_tts',
    pitch_supported: true,
    sample_text: "नमस्ते! हमारी सेवा में आपका स्वागत है। आज मैं आपकी क्या सहायता कर सकती हूँ?"
  },

  // --- ADULT MALE HUMAN VOICES ---
  {
    id: 'en-US-GuyNeural',
    name: 'Guy',
    gender: 'Male',
    locale: 'en-US',
    accent: 'US',
    category: 'Adult Male',
    age_group: 'Adult',
    style: 'Natural, dependable, casual American male',
    engine: 'edge_tts',
    pitch_supported: true,
    sample_text: "Hello! Guy here. What can I look up or help you with today?"
  },
  {
    id: 'en-US-BrianNeural',
    name: 'Brian',
    gender: 'Male',
    locale: 'en-US',
    accent: 'US',
    category: 'Adult Male',
    age_group: 'Adult',
    style: 'Clear, professional, corporate American male',
    engine: 'edge_tts',
    pitch_supported: true,
    sample_text: "Welcome to client services. My name is Brian. How can I be of assistance?"
  },
  {
    id: 'en-US-AndrewNeural',
    name: 'Andrew',
    gender: 'Male',
    locale: 'en-US',
    accent: 'US',
    category: 'Adult Male',
    age_group: 'Adult',
    style: 'Warm, steady, conversational American male',
    engine: 'edge_tts',
    pitch_supported: true,
    sample_text: "Hi, this is Andrew. I am standing by to help with any questions you have."
  },
  {
    id: 'en-IN-PrabhatNeural',
    name: 'Prabhat',
    gender: 'Male',
    locale: 'en-IN',
    accent: 'IN',
    category: 'Adult Male',
    age_group: 'Adult',
    style: 'Direct, confident, authentic Indian English male',
    engine: 'edge_tts',
    pitch_supported: true,
    sample_text: "Good day! Prabhat here. Please let me know your account number or query."
  },
  {
    id: 'en-GB-RyanNeural',
    name: 'Ryan',
    gender: 'Male',
    locale: 'en-GB',
    accent: 'UK',
    category: 'Adult Male',
    age_group: 'Adult',
    style: 'Friendly, contemporary, modern British male',
    engine: 'edge_tts',
    pitch_supported: true,
    sample_text: "Cheers for calling! I am Ryan. What can I do for you today?"
  },
  {
    id: 'en-GB-ThomasNeural',
    name: 'Thomas',
    gender: 'Male',
    locale: 'en-GB',
    accent: 'UK',
    category: 'Adult Male',
    age_group: 'Adult',
    style: 'Dignified, precise, formal British English',
    engine: 'edge_tts',
    pitch_supported: true,
    sample_text: "Good day. My name is Thomas. I will be pleased to assist with your requirements."
  },
  {
    id: 'en-CA-LiamNeural',
    name: 'Liam',
    gender: 'Male',
    locale: 'en-CA',
    accent: 'CA',
    category: 'Adult Male',
    age_group: 'Adult',
    style: 'Energetic, clear Canadian English male',
    engine: 'edge_tts',
    pitch_supported: true,
    sample_text: "Hey there! Liam here. How's it going? Let me know how I can help."
  },
  {
    id: 'en-AU-WilliamMultilingualNeural',
    name: 'William',
    gender: 'Male',
    locale: 'en-AU',
    accent: 'AU',
    category: 'Adult Male',
    age_group: 'Adult',
    style: 'Authentic Australian male, balanced and clear',
    engine: 'edge_tts',
    pitch_supported: true,
    sample_text: "Hello, William speaking. Ready to answer any questions you might have."
  },
  {
    id: 'en-IE-ConnorNeural',
    name: 'Connor',
    gender: 'Male',
    locale: 'en-IE',
    accent: 'IE',
    category: 'Adult Male',
    age_group: 'Adult',
    style: 'Articulate, friendly Irish male accent',
    engine: 'edge_tts',
    pitch_supported: true,
    sample_text: "Hello there! Connor here from customer service. How can I help you today?"
  },
  {
    id: 'hi-IN-MadhurNeural',
    name: 'Madhur',
    gender: 'Male',
    locale: 'hi-IN',
    accent: 'IN-Hindi',
    category: 'Adult Male',
    age_group: 'Adult',
    style: 'Calm, articulate, natural Hindi male',
    engine: 'edge_tts',
    pitch_supported: true,
  }
];

const KOKORO_VOICES = CALL_VOICES;

const SILENCE_OPTIONS = [
  { label: 'Responsive (450ms - Fast Queries)', value: 450 },
  { label: 'Balanced (600ms - Recommended Default)', value: 600 },
  { label: 'Conversational (800ms - Natural Pauses)', value: 800 },
  { label: 'Relaxed (1000ms - Extended Pauses)', value: 1000 },
];

export default function VoiceCallSimulator({
  language: initialLanguage = 'auto',
  onLanguageChange,
  model: initialModel = 'base.en',
  onModelChange,
  engine: initialEngine = 'whisper',
  onEngineChange,
  onCallResult,
  health,
  mode: propMode,
  onModeChange,
  onOpenSherpa
}) {
  // Call Lifecycle: 'idle' | 'calling' | 'active' | 'ended'
  const [callState, setCallState] = useState('idle');
  // Call Status: 'listening' | 'speaking' | 'processing' | 'ai_speaking' | 'muted'
  const [callStatus, setCallStatus] = useState('listening');
  const [conversationState, setConversationState] = useState('IDLE');
  const [recognitionStatus, setRecognitionStatus] = useState('Local recognition ready');
  const [callDuration, setCallDuration] = useState(0);
  const [isMuted, setIsMuted] = useState(false);

  // STT Engine, Model & Language State
  const [sttEngine, setSttEngine] = useState(() => {
    const saved = localStorage.getItem('stt_engine');
    return (saved && saved !== 'sherpa') ? saved : (initialEngine && initialEngine !== 'sherpa' ? initialEngine : 'sherpa');
  });
  const [sttModel, setSttModel] = useState(() => {
    const saved = localStorage.getItem('whisper_model');
    const savedEngine = localStorage.getItem('stt_engine');
    const usingSherpa = savedEngine === 'sherpa' || (
      !savedEngine && (!initialEngine || initialEngine === 'sherpa')
    );
    if (usingSherpa) {
      return (saved && saved.startsWith('sherpa-') && saved !== 'sherpa-streaming')
        ? saved
        : 'sherpa-onnx-streaming-zipformer-en-2023-06-26';
    }
    return (saved && saved !== 'sherpa-streaming')
      ? saved
      : (initialModel && initialModel !== 'sherpa-streaming' ? initialModel : 'base.en');
  });
  const [sttLanguage, setSttLanguage] = useState(() => initialLanguage || localStorage.getItem('whisper_language') || 'en');

  // Text-to-Voice (TTS) Model & Human Voice State
  const [ttsModel, setTtsModel] = useState(() => localStorage.getItem('call_tts_model') || 'auto');
  const [selectedVoice, setSelectedVoice] = useState(() => localStorage.getItem('call_selected_voice') || 'en-IN-NeerjaNeural');
  const [selectedPitch, setSelectedPitch] = useState(() => localStorage.getItem('call_selected_pitch') || '+0Hz');
  const [selectedVoiceCategory, setSelectedVoiceCategory] = useState('All');
  const [selectedVoiceAccent, setSelectedVoiceAccent] = useState('All');
  const [previewPlayingVoiceId, setPreviewPlayingVoiceId] = useState(null);
  const [previewLatency, setPreviewLatency] = useState(null);
  const [humanVoiceCatalog, setHumanVoiceCatalog] = useState(CALL_VOICES);

  // Refresh verified human voices from backend API on mount
  useEffect(() => {
    getHumanVoices()
      .then((data) => {
        if (data && data.voices && data.voices.length > 0) {
          setHumanVoiceCatalog(data.voices);
        }
      })
      .catch((e) => console.warn('Could not refresh human voice catalog:', e));
  }, []);

  // Real-Time Full-Duplex Streaming & Interruption States
  const [livePartial, setLivePartial] = useState('');
  const [liveAssistantText, setLiveAssistantText] = useState('');
  const [bargeInNotice, setBargeInNotice] = useState(null);
  const [fdMetrics, setFdMetrics] = useState({
    timeToFirstPartial: null,
    timeToFinal: null,
    timeToFirstToken: null,
    timeToFirstAudio: null,
    totalTurnLatency: null,
    interruptionLatency: null
  });

  // Automatic English Accent Detection States
  const [detectedAccent, setDetectedAccent] = useState(null);
  const [isDetectingAccent, setIsDetectingAccent] = useState(false);
  const [autoAdaptVoice, setAutoAdaptVoice] = useState(() => {
    const saved = localStorage.getItem('call_auto_adapt_voice');
    return saved !== null ? saved === 'true' : true;
  });
  const [voiceAdaptationNotice, setVoiceAdaptationNotice] = useState(null);
  const voiceAdaptTimerRef = useRef(null);

  const handleToggleAutoAdaptVoice = (enabled) => {
    setAutoAdaptVoice(enabled);
    localStorage.setItem('call_auto_adapt_voice', enabled ? 'true' : 'false');
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'set_auto_adapt_voice',
        enabled: enabled
      }));
    }
  };

  // Full-Duplex Refs
  const wsRef = useRef(null);
  const audioQueueRef = useRef([]);
  const isPlayingAudioRef = useRef(false);
  const currentAudioRef = useRef(null);
  const currentTurnIdRef = useRef(0);
  const scriptProcessorRef = useRef(null);
  const bargeInTimerRef = useRef(null);
  const tEndpointClientRef = useRef(null);
  const firstAudibleMeasuredRef = useRef(null);
  const lastFinalUserTextRef = useRef('');
  const lastFinalMetricsRef = useRef(null);
  const processedEventsRef = useRef(new Set());
  const lastSeqByTurnRef = useRef({});

  // Sync with prop changes if parent updates
  useEffect(() => {
    if (initialEngine && initialEngine !== sttEngine) {
      setSttEngine(initialEngine);
    }
  }, [initialEngine]);

  useEffect(() => {
    if (initialModel && initialModel !== sttModel) {
      setSttModel(initialModel);
    }
  }, [initialModel]);

  useEffect(() => {
    if (initialLanguage && initialLanguage !== sttLanguage) {
      setSttLanguage(initialLanguage);
    }
  }, [initialLanguage]);

  const [silenceTimeoutMs, setSilenceTimeoutMs] = useState(() => {
    const saved = localStorage.getItem('call_silence_timeout');
    return saved ? Math.max(300, Number(saved)) : 600;
  });
  const [chatHistory, setChatHistory] = useState([]);
  const [audioLevel, setAudioLevel] = useState(0); // 0 to 100 for visualizer
  const [showCallSettings, setShowCallSettings] = useState(false);

  // Simulator Mode: 'call' | 'stt' | 'tts'

  const [simulatorMode, setSimulatorMode] = useState(() => propMode || localStorage.getItem('sim_active_mode') || 'call');

  // Sync with propMode changes from header dropdown
  useEffect(() => {
    if (propMode && propMode !== simulatorMode) {
      if (callState === 'active') {
        cleanupCallAudio();
        setCallState('idle');
      }
      stopAudioPlayback();
      setSimulatorMode(propMode);
    }
  }, [propMode]);

  // Dedicated STT Transcriber State
  const [sttIsRecording, setSttIsRecording] = useState(false);
  const [sttRecordDuration, setSttRecordDuration] = useState(0);
  const [sttIsProcessing, setSttIsProcessing] = useState(false);
  const [sttTranscriptResult, setSttTranscriptResult] = useState(null);
  const [sttError, setSttError] = useState(null);
  const [sttCopied, setSttCopied] = useState(false);

  const sttRecorderRef = useRef(null);
  const sttStreamRef = useRef(null);
  const sttChunksRef = useRef([]);
  const sttTimerRef = useRef(null);
  const sttCanvasRef = useRef(null);
  const sttAnalyserRef = useRef(null);
  const sttAnimRef = useRef(null);
  const sttAudioContextRef = useRef(null);

  // Dedicated TTS Speech Generator State
  const [ttsInputText, setTtsInputText] = useState(
    'Hello! Thank you for calling customer service. How may I assist you with your inquiry today?'
  );
  const [ttsSpeed, setTtsSpeed] = useState(1.0);
  const [ttsIsGenerating, setTtsIsGenerating] = useState(false);
  const [ttsResult, setTtsResult] = useState(null);
  const [ttsAudioUrl, setTtsAudioUrl] = useState(null);
  const [ttsError, setTtsError] = useState(null);

  // Standalone TTS Quick Tester State
  const [testText, setTestText] = useState('Hello! I can hear you loud and clear. How can I help you today?');
  const [isTestingTTS, setIsTestingTTS] = useState(false);

  // Audio & Hardware Refs (kept in refs for high-frequency 60FPS loop access)
  const streamRef = useRef(null);
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const vadAnimationRef = useRef(null);
  const callTimerRef = useRef(null);
  const transcriptBottomRef = useRef(null);
  const canvasRef = useRef(null);

  // VAD state refs
  const callActiveRef = useRef(false);
  const isSpeakingRef = useRef(false);
  const speechStartTimeRef = useRef(0);
  const lastSpeechTimeRef = useRef(0);
  const isProcessingRef = useRef(false);
  const isAiSpeakingRef = useRef(false);
  const isMutedRef = useRef(false);
  const reconnectAttemptsRef = useRef(0);
  const reconnectTimerRef = useRef(null);
  const chatHistoryRef = useRef(chatHistory);
  chatHistoryRef.current = chatHistory;

  // Sync mute ref
  useEffect(() => {
    isMutedRef.current = isMuted;
  }, [isMuted]);

  // Duration Timer for active call
  useEffect(() => {
    if (callState === 'active') {
      callTimerRef.current = setInterval(() => {
        setCallDuration((prev) => prev + 1);
      }, 1000);
    } else {
      if (callTimerRef.current) clearInterval(callTimerRef.current);
    }
    return () => {
      if (callTimerRef.current) clearInterval(callTimerRef.current);
    };
  }, [callState]);

  // Auto-scroll chat transcript to latest message
  useEffect(() => {
    transcriptBottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatHistory]);

  const formatCallTime = (secs) => {
    const mins = Math.floor(secs / 60);
    const s = secs % 60;
    return `${mins < 10 ? '0' : ''}${mins}:${s < 10 ? '0' : ''}${s}`;
  };

  const handleSilenceChange = (val) => {
    setSilenceTimeoutMs(val);
    localStorage.setItem('call_silence_timeout', String(val));
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(
        JSON.stringify({
          type: 'config',
          silence_timeout_sec: val / 1000
        })
      );
    }
  };

  const handleSttEngineChange = (newEngine) => {
    setSttEngine(newEngine);
    localStorage.setItem('stt_engine', newEngine);
    if (onEngineChange) {
      onEngineChange(newEngine);
    }
    const defaultModel = newEngine === 'sherpa'
      ? 'sherpa-onnx-streaming-zipformer-en-2023-06-26'
      : newEngine === 'deepgram' ? 'nova-3' : 'base.en';
    handleSttModelChange(defaultModel);
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'config',
        stt_engine: newEngine,
        stt_model: defaultModel,
        stt_language: sttLanguage
      }));
    }
  };

  const handleSttModelChange = (newModel) => {
    setSttModel(newModel);
    localStorage.setItem('whisper_model', newModel);
    if (onModelChange) {
      onModelChange(newModel);
    }
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'config',
        stt_engine: sttEngine,
        stt_model: newModel,
        stt_language: sttLanguage
      }));
    }
  };

  const handleSttLanguageChange = (newLang) => {
    setSttLanguage(newLang);
    localStorage.setItem('whisper_language', newLang);
    if (onLanguageChange) {
      onLanguageChange(newLang);
    }
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'config',
        stt_engine: sttEngine,
        stt_model: sttModel,
        stt_language: newLang
      }));
    }
  };

  const handleTtsModelChange = (newEngine) => {
    setTtsModel(newEngine);
    localStorage.setItem('call_tts_model', newEngine);

    if (newEngine !== 'auto') {
      const compatibleVoices = humanVoiceCatalog.filter((v) => v.engine === newEngine);
      const isCurrentVoiceCompatible = compatibleVoices.some((v) => v.id === selectedVoice);
      if (!isCurrentVoiceCompatible && compatibleVoices.length > 0) {
        const fallbackVoice = compatibleVoices[0].id;
        setSelectedVoice(fallbackVoice);
        localStorage.setItem('call_selected_voice', fallbackVoice);
      }
    }
  };

  const handleVoiceChange = (voiceId) => {
    setSelectedVoice(voiceId);
    localStorage.setItem('call_selected_voice', voiceId);
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(
        JSON.stringify({
          type: 'config',
          tts_voice: voiceId,
          tts_pitch: selectedPitch,
          tts_speed: ttsSpeed
        })
      );
    }
    const voiceObj = humanVoiceCatalog.find((v) => v.id === voiceId);
    if (voiceObj && ttsModel !== 'auto' && voiceObj.engine !== ttsModel) {
      setTtsModel(voiceObj.engine);
      localStorage.setItem('call_tts_model', voiceObj.engine);
    }
  };

  const handlePitchChange = (pitchVal) => {
    setSelectedPitch(pitchVal);
    localStorage.setItem('call_selected_pitch', pitchVal);
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(
        JSON.stringify({
          type: 'config',
          tts_voice: selectedVoice,
          tts_pitch: pitchVal,
          tts_speed: ttsSpeed
        })
      );
    }
  };

  const handleSpeedChange = (speedVal) => {
    setTtsSpeed(speedVal);
    localStorage.setItem('call_selected_speed', String(speedVal));
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(
        JSON.stringify({
          type: 'config',
          tts_voice: selectedVoice,
          tts_pitch: selectedPitch,
          tts_speed: speedVal
        })
      );
    }
  };

  const handlePlayVoiceSample = async (voice) => {
    if (previewPlayingVoiceId === voice.id) {
      stopAudioPlayback();
      setPreviewPlayingVoiceId(null);
      return;
    }
    stopAudioPlayback();
    setPreviewPlayingVoiceId(voice.id);
    setPreviewLatency(null);
    const start = performance.now();
    try {
      const sampleText = voice.sample_text || `Hello, this is a sample of ${voice.name}'s natural human voice.`;
      const res = await synthesizeSpeech(sampleText, {
        voice: voice.id,
        pitch: selectedPitch,
        speed: ttsSpeed,
        engine: 'edge_tts'
      });
      const latencyMs = Math.round(performance.now() - start);
      setPreviewLatency({ voiceId: voice.id, latencyMs });
      playAudioFromBase64(res.audio_base64, res.format, () => {
        setPreviewPlayingVoiceId(null);
      });
    } catch (err) {
      console.error('Voice sample preview failed:', err);
      setPreviewPlayingVoiceId(null);
    }
  };

  // Filtered voice options based on chosen Category, Accent, and Engine
  const filteredVoices = useMemo(() => {
    let list = humanVoiceCatalog;
    if (ttsModel !== 'auto') {
      list = list.filter((v) => v.engine === ttsModel);
    }
    if (selectedVoiceCategory !== 'All') {
      list = list.filter((v) => v.category === selectedVoiceCategory);
    }
    if (selectedVoiceAccent !== 'All') {
      list = list.filter((v) => v.accent === selectedVoiceAccent);
    }
    return list;
  }, [humanVoiceCatalog, ttsModel, selectedVoiceCategory, selectedVoiceAccent]);

  const selectedVoiceObj = useMemo(() => {
    return humanVoiceCatalog.find((v) => v.id === selectedVoice) || humanVoiceCatalog[0] || CALL_VOICES[0];
  }, [selectedVoice, humanVoiceCatalog]);

  const activeTtsEngine = useMemo(() => {
    if (ttsModel !== 'auto') return ttsModel;
    return selectedVoiceObj?.engine || 'edge_tts';
  }, [ttsModel, selectedVoiceObj]);

  const resamplePhaseRef = useRef(0);

  /**
   * Resample Float32 audio to 16kHz mono PCM16 Int16Array for Sherpa-ONNX streaming.
   * Tracks phase continuity across buffer chunks to prevent boundary clicking.
   * Uses area-weighted averaging / linear interpolation to eliminate high-frequency aliasing.
   */
  const resampleAndEncodePCM16 = useCallback((audioBuffer, fromSampleRate, toSampleRate = 16000) => {
    if (fromSampleRate === toSampleRate) {
      const pcm16 = new Int16Array(audioBuffer.length);
      for (let i = 0; i < audioBuffer.length; i++) {
        const s = Math.max(-1, Math.min(1, audioBuffer[i]));
        pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
      }
      return pcm16;
    }

    const ratio = fromSampleRate / toSampleRate;
    let phase = resamplePhaseRef.current || 0;
    const outSamples = [];

    if (ratio > 1.0) {
      // Downsampling (e.g. 48kHz / 44.1kHz -> 16kHz): box / area averaging acts as low-pass anti-aliasing filter
      while (phase < audioBuffer.length) {
        const start = Math.floor(phase);
        const end = Math.min(audioBuffer.length, Math.floor(phase + ratio));
        let sum = 0;
        let count = 0;
        for (let j = start; j < end; j++) {
          sum += audioBuffer[j];
          count++;
        }
        const s = count > 0 ? sum / count : (audioBuffer[start] || 0);
        const clamped = Math.max(-1, Math.min(1, s));
        outSamples.push(clamped < 0 ? clamped * 0x8000 : clamped * 0x7FFF);
        phase += ratio;
      }
    } else {
      // Upsampling: linear interpolation
      while (phase < audioBuffer.length - 1) {
        const i = Math.floor(phase);
        const frac = phase - i;
        const s = audioBuffer[i] * (1 - frac) + audioBuffer[i + 1] * frac;
        const clamped = Math.max(-1, Math.min(1, s));
        outSamples.push(clamped < 0 ? clamped * 0x8000 : clamped * 0x7FFF);
        phase += ratio;
      }
    }

    // Save fractional phase offset for seamless continuation on the next chunk
    resamplePhaseRef.current = Math.max(0, phase - audioBuffer.length);

    const pcm16 = new Int16Array(outSamples.length);
    for (let k = 0; k < outSamples.length; k++) {
      pcm16[k] = outSamples[k];
    }
    return pcm16;
  }, []);

  /**
   * Stop currently playing assistant audio and flush queued chunks (used on barge-in)
   */
  const stopActiveAudio = useCallback(() => {
    if (currentAudioRef.current) {
      try {
        currentAudioRef.current.pause();
        currentAudioRef.current.currentTime = 0;
      } catch (e) { }
      currentAudioRef.current = null;
    }
    audioQueueRef.current = [];
    isPlayingAudioRef.current = false;
    isAiSpeakingRef.current = false;
  }, []);

  /**
   * Play received incremental audio chunks sequentially
   */
  const playNextAudioChunk = useCallback(() => {
    if (audioQueueRef.current.length === 0) {
      isPlayingAudioRef.current = false;
      isAiSpeakingRef.current = false;
      if (callActiveRef.current) {
        setCallStatus('listening');
      }
      return;
    }

    const chunk = audioQueueRef.current.shift();
    if (chunk.turnId !== currentTurnIdRef.current) {
      // Chunk belongs to an interrupted or stale turn; discard it
      playNextAudioChunk();
      return;
    }

    isPlayingAudioRef.current = true;
    isAiSpeakingRef.current = true;
    setCallStatus('ai_speaking');

    const mime = chunk.format === 'mp3' ? 'audio/mpeg' : 'audio/wav';
    const audio = new Audio(`data:${mime};base64,${chunk.base64}`);
    currentAudioRef.current = audio;

    if (!firstAudibleMeasuredRef.current && tEndpointClientRef.current) {
      firstAudibleMeasuredRef.current = Math.round(performance.now() - tEndpointClientRef.current);
    }

    audio.onended = () => {
      currentAudioRef.current = null;
      playNextAudioChunk();
    };
    audio.onerror = () => {
      currentAudioRef.current = null;
      playNextAudioChunk();
    };
    audio.play().catch(() => {
      playNextAudioChunk();
    });
  }, []);

  /**
   * Complete Audio Clean Up on Call Termination or Unmount
   */
  const cleanupCallAudio = useCallback(() => {
    callActiveRef.current = false;
    isSpeakingRef.current = false;
    isProcessingRef.current = false;
    isAiSpeakingRef.current = false;

    if (wsRef.current) {
      try {
        wsRef.current.close();
      } catch (e) { }
      wsRef.current = null;
    }

    if (scriptProcessorRef.current) {
      try {
        scriptProcessorRef.current.disconnect();
      } catch (e) { }
      scriptProcessorRef.current = null;
    }

    if (bargeInTimerRef.current) {
      clearTimeout(bargeInTimerRef.current);
      bargeInTimerRef.current = null;
    }
    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }

    stopActiveAudio();
    stopAudioPlayback();

    if (vadAnimationRef.current) {
      cancelAnimationFrame(vadAnimationRef.current);
      vadAnimationRef.current = null;
    }

    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      try {
        mediaRecorderRef.current.onstop = null;
        mediaRecorderRef.current.stop();
      } catch (e) { }
      mediaRecorderRef.current = null;
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
      audioContextRef.current.close().catch(() => { });
      audioContextRef.current = null;
    }

    audioChunksRef.current = [];
    resamplePhaseRef.current = 0;
    setAudioLevel(0);
    setLivePartial('');
    setLiveAssistantText('');
  }, [stopActiveAudio]);


  useEffect(() => {
    return () => cleanupCallAudio();
  }, [cleanupCallAudio]);

  /**
   * Start a new recording turn on the continuous stream
   */
  const startListeningRecorder = useCallback(() => {
    if (!callActiveRef.current || !streamRef.current) return;
    if (isProcessingRef.current || isAiSpeakingRef.current) return;

    audioChunksRef.current = [];
    isSpeakingRef.current = false;
    speechStartTimeRef.current = 0;
    lastSpeechTimeRef.current = 0;

    if (isMutedRef.current) {
      setCallStatus('muted');
      return;
    }

    try {
      const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
        ? 'audio/webm;codecs=opus'
        : 'audio/webm';

      const recorder = new MediaRecorder(streamRef.current, { mimeType });
      mediaRecorderRef.current = recorder;

      recorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      recorder.start(100);
      setCallStatus('listening');
    } catch (err) {
      console.error('Failed to start MediaRecorder:', err);
    }
  }, []);

  /**
   * Recycle recorder if idle in silence for > 5 seconds to keep buffer fresh
   */
  const recycleIdleRecorder = useCallback(() => {
    const recorder = mediaRecorderRef.current;
    if (recorder && recorder.state === 'recording' && !isSpeakingRef.current) {
      recorder.onstop = () => {
        audioChunksRef.current = [];
        if (callActiveRef.current && !isProcessingRef.current && !isAiSpeakingRef.current && !isMutedRef.current) {
          try {
            const mimeType = recorder.mimeType || 'audio/webm';
            const freshRecorder = new MediaRecorder(streamRef.current, { mimeType });
            mediaRecorderRef.current = freshRecorder;
            freshRecorder.ondataavailable = (event) => {
              if (event.data && event.data.size > 0) {
                audioChunksRef.current.push(event.data);
              }
            };
            freshRecorder.start(100);
          } catch (e) { }
        }
      };
      recorder.stop();
    }
  }, []);

  /**
   * Complete utterance turn and dispatch to AI
   */
  const handleCompleteUtterance = useCallback(() => {
    if (isProcessingRef.current) return;
    isProcessingRef.current = true;
    setCallStatus('processing');

    const recorder = mediaRecorderRef.current;
    if (!recorder || recorder.state === 'inactive') {
      isProcessingRef.current = false;
      startListeningRecorder();
      return;
    }

    recorder.onstop = async () => {
      const mimeType = recorder.mimeType || 'audio/webm';
      const chunks = audioChunksRef.current;
      const audioBlob = new Blob(chunks, { type: mimeType });
      audioChunksRef.current = [];

      if (!audioBlob || audioBlob.size === 0) {
        isProcessingRef.current = false;
        startListeningRecorder();
        return;
      }

      try {
        const groqApiKey = localStorage.getItem('groq_api_key') || undefined;
        const groqModel = localStorage.getItem('groq_model') || 'qwen/qwen3.8-27b';

        // Filter clean history for telephony context
        const cleanHistory = chatHistoryRef.current
          .filter((m) => m.role === 'user' || m.role === 'assistant')
          .map((m) => ({ role: m.role, content: m.content }));

        const result = await sendVoiceCallTurn(audioBlob, cleanHistory, {
          language: sttLanguage,
          whisperModel: sttModel,
          groqApiKey,
          groqModel,
          ttsVoice: selectedVoice,
          ttsEngine: activeTtsEngine,
          engine: sttEngine,
          stt_engine: sttEngine
        });

        const userText = (result.transcript || '').trim();
        const aiReply = (result.reply || '').trim();

        if (userText && aiReply) {
          // Add turn to chat history
          setChatHistory((prev) => [
            ...prev,
            { role: 'user', content: userText },
            { role: 'assistant', content: aiReply }
          ]);

          if (result.accent_analysis) {
            setDetectedAccent({
              accent: result.accent_analysis.accent,
              code: result.accent_analysis.code,
              flag: result.accent_analysis.flag || '🌐',
              confidence: result.accent_analysis.confidence,
              status: result.accent_analysis.status,
              sampleDuration: result.accent_analysis.sample_duration,
              latencyMs: result.accent_analysis.latency_ms,
              suggestedTtsVoice: result.accent_analysis.tts_voice,
              details: result.accent_analysis.details,
              isSmoothed: false
            });
            if (autoAdaptVoice && result.accent_analysis.tts_voice) {
              setSelectedVoice(result.accent_analysis.tts_voice);
              localStorage.setItem('call_selected_voice', result.accent_analysis.tts_voice);
              setVoiceAdaptationNotice(`AI voice adapted to ${result.accent_analysis.accent} (${result.accent_analysis.tts_voice})`);
              if (voiceAdaptTimerRef.current) clearTimeout(voiceAdaptTimerRef.current);
              voiceAdaptTimerRef.current = setTimeout(() => setVoiceAdaptationNotice(null), 4000);
            }
          }

          if (onCallResult) {
            onCallResult(result);
          }

          // If AI synthesized speech audio, play it immediately
          if (result.audio_base64 && callActiveRef.current) {
            isAiSpeakingRef.current = true;
            setCallStatus('ai_speaking');

            playAudioFromBase64(result.audio_base64, result.audio_format, () => {
              isAiSpeakingRef.current = false;
              isProcessingRef.current = false;
              // Resume continuous listening automatically!
              startListeningRecorder();
            });
            return;
          }
        }
      } catch (err) {
        console.error('Call turn error:', err);
        setChatHistory((prev) => [
          ...prev,
          { role: 'system_notification', content: `Notice: ${err.message}` }
        ]);
      }

      // If silence or empty transcript, smoothly resume listening
      isProcessingRef.current = false;
      startListeningRecorder();
    };

    recorder.stop();
  }, [sttLanguage, sttModel, activeTtsEngine, selectedVoice, onCallResult, startListeningRecorder]);

  /**
   * Barge-in interruption: called when user starts speaking while AI is talking
   */
  const handleBargeIn = useCallback(() => {
    stopActiveAudio();
    stopAudioPlayback();
    isAiSpeakingRef.current = false;
    isProcessingRef.current = false;
    isSpeakingRef.current = true;
    speechStartTimeRef.current = Date.now();
    lastSpeechTimeRef.current = Date.now();
    setCallStatus('speaking');

    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'interrupt' }));
    } else {
      startListeningRecorder();
    }
  }, [stopActiveAudio, startListeningRecorder]);


  /**
   * Real-time Voice Activity Detection (VAD) & Waveform Loop
   */
  const startVADLoop = useCallback(() => {
    if (!analyserRef.current) return;
    const analyser = analyserRef.current;
    const bufferLength = analyser.frequencyBinCount;
    const freqData = new Uint8Array(bufferLength);

    const SPEECH_THRESHOLD = 14; // Voice energy threshold (out of 255)
    const BARGE_IN_THRESHOLD = 26; // Slightly higher threshold to avoid speaker echo

    const checkAudioFrame = () => {
      vadAnimationRef.current = requestAnimationFrame(checkAudioFrame);
      if (!callActiveRef.current) return;

      analyser.getByteFrequencyData(freqData);

      // Compute average volume in human speech frequency bands
      let sum = 0;
      for (let i = 2; i < Math.min(bufferLength, 64); i++) {
        sum += freqData[i];
      }
      const voiceEnergy = sum / (Math.min(bufferLength, 64) - 2);
      const normalizedLevel = Math.min(100, Math.round((voiceEnergy / 60) * 100));
      setAudioLevel(normalizedLevel);

      // Draw dynamic visualizer canvas
      if (canvasRef.current) {
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        const barCount = 28;
        const barWidth = canvas.width / barCount - 2;
        for (let i = 0; i < barCount; i++) {
          const val = freqData[i * 2] || 0;
          const barHeight = Math.max(3, (val / 255) * canvas.height);
          const x = i * (barWidth + 2);

          if (isAiSpeakingRef.current) {
            ctx.fillStyle = `rgba(245, 158, 11, ${0.4 + (val / 255) * 0.6})`;
          } else if (isSpeakingRef.current) {
            ctx.fillStyle = `rgba(56, 189, 248, ${0.5 + (val / 255) * 0.5})`;
          } else {
            ctx.fillStyle = `rgba(100, 116, 139, 0.25)`;
          }

          ctx.beginPath();
          ctx.roundRect(x, (canvas.height - barHeight) / 2, barWidth, barHeight, 2);
          ctx.fill();
        }
      }

      // Check Mute State
      if (isMutedRef.current) {
        return;
      }

      // Check Barge-In while AI is speaking
      if (isAiSpeakingRef.current) {
        return;
      }

      // If AI is currently generating response on server, skip mic VAD
      if (isProcessingRef.current) {
        return;
      }

      // Active Call Listening VAD
      const now = Date.now();
      const isVoiceActive = voiceEnergy >= SPEECH_THRESHOLD;

      if (isVoiceActive) {
        if (!isSpeakingRef.current) {
          // User started speaking!
          isSpeakingRef.current = true;
          speechStartTimeRef.current = now;
          setCallStatus('speaking');
        }
        lastSpeechTimeRef.current = now;
      } else {
        if (isSpeakingRef.current) {
          // User was speaking, now paused / silent
          const silenceDuration = now - lastSpeechTimeRef.current;
          const totalSpeechDuration = now - speechStartTimeRef.current;

          if (silenceDuration >= silenceTimeoutMs) {
            if (totalSpeechDuration >= 220) {
              // Valid utterance turn completed!
              isSpeakingRef.current = false;
              // If real-time WebSocket is open, server handles turn streaming.
              // ONLY dispatch to REST handleCompleteUtterance if WebSocket is NOT open!
              if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
                handleCompleteUtterance();
              }
            } else {
              // Too short (minor click/cough), return to listening
              isSpeakingRef.current = false;
              setCallStatus('listening');
            }
          }
        } else {
          // Silent and not speaking: periodically recycle recorder chunks
          if (audioChunksRef.current.length > 55) {
            recycleIdleRecorder();
          }
        }
      }
    };

    checkAudioFrame();
  }, [silenceTimeoutMs, handleBargeIn, handleCompleteUtterance, recycleIdleRecorder]);

  /**
   * Initiate Call: Request mic, connect audio context, play AI greeting, and start continuous loop
   */
  const handleStartCall = async ({ reconnecting = false } = {}) => {
    cleanupCallAudio();
    stopAudioPlayback();
    setCallState('calling');
    setConversationState(reconnecting ? 'RECONNECTING' : 'CONNECTING');
    if (!reconnecting) {
      reconnectAttemptsRef.current = 0;
      setCallDuration(0);
    }
    setIsMuted(false);
    isMutedRef.current = false;
    setLivePartial('');
    setLiveAssistantText('');
    setBargeInNotice(null);

    const isStreamingSherpa = sttEngine === 'sherpa' && (!sttModel || sttModel === 'sherpa-streaming' || sttModel.includes('streaming'));

    if (!reconnecting) setChatHistory([]);

    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error(
          'Microphone requires a secure connection (HTTPS or http://localhost). Remote HTTP IP connections block microphone access by default.'
        );
      }

      // 1. Obtain continuous microphone stream with echo cancellation
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          channelCount: 1,
          sampleRate: 16000
        }
      });
      streamRef.current = stream;

      // 2. Set up AudioContext and AnalyserNode
      const audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 256;
      analyser.smoothingTimeConstant = 0.3;
      const source = audioCtx.createMediaStreamSource(stream);
      source.connect(analyser);

      audioContextRef.current = audioCtx;
      analyserRef.current = analyser;

      if (true) {
        // --- REAL-TIME FULL-DUPLEX WEBSOCKET PIPELINE ---
        const primaryWsUrl = getWebSocketUrl('/ws/call');
        const fallbackWsUrl = getFallbackWebSocketUrl('/ws/call');

        // Clean up any existing connection before opening a new one
        if (wsRef.current) {
          try {
            wsRef.current.onopen = null;
            wsRef.current.onmessage = null;
            wsRef.current.onerror = null;
            wsRef.current.onclose = null;
            wsRef.current.close();
          } catch (_) {}
          wsRef.current = null;
        }
        processedEventsRef.current.clear();
        lastSeqByTurnRef.current = {};

        let activeWsUrl = primaryWsUrl;
        const ws = new WebSocket(primaryWsUrl);
        wsRef.current = ws;

        ws.onopen = () => {
          setCallState('active');
          setConversationState('LISTENING');
          callActiveRef.current = true;
          startVADLoop();

          // Send current configuration
          const groqModel = localStorage.getItem('groq_model') || 'qwen/qwen3.8-27b';
          ws.send(
            JSON.stringify({
              type: 'config',
              tts_voice: selectedVoice,
              tts_pitch: selectedPitch,
              tts_speed: ttsSpeed,
              tts_engine: activeTtsEngine,
              groq_model: groqModel,
              is_phone_call: true,
              auto_adapt_voice: autoAdaptVoice,
              stt_engine: sttEngine,
              stt_model: sttModel,
              stt_language: 'en',
              silence_timeout_sec: silenceTimeoutMs / 1000,
              conversation_history: reconnecting
                ? chatHistoryRef.current
                  .filter((message) => (message.role === 'user' || message.role === 'assistant') && !message.isPartial)
                  .map(({ role, content }) => ({ role, content }))
                : []
            })
          );

          // Greet only new calls. A reconnect restores prior context instead.
          if (!reconnecting) {
          // Initial greeting
          const isHindi = selectedVoice.startsWith('hi-') || sttLanguage === 'hi';
          const greeting = isHindi
            ? 'नमस्ते! कॉल करने के लिए धन्यवाद। मैं आज आपकी क्या सहायता कर सकता हूँ?'
            : 'Hello! Thanks for calling. How can I help you today?';

          setChatHistory([
            { id: 'asst_greeting_0', role: 'assistant', content: greeting, timestamp: Date.now() }
          ]);

          isAiSpeakingRef.current = true;
          setCallStatus('ai_speaking');

          const voiceObj = humanVoiceCatalog.find((v) => v.id === selectedVoice) || CALL_VOICES.find((v) => v.id === selectedVoice);
          const ttsEngine = voiceObj?.engine || (selectedVoice.includes('Neural') ? 'edge_tts' : 'kokoro');

          synthesizeSpeech(greeting, {
            voice: selectedVoice,
            pitch: selectedPitch,
            speed: ttsSpeed,
            engine: ttsEngine
          })
            .then((res) => {
              if (callActiveRef.current) {
                playAudioFromBase64(res.audio_base64, res.format, () => {
                  isAiSpeakingRef.current = false;
                  setCallStatus('listening');
                });
              }
            })
            .catch(() => {
              isAiSpeakingRef.current = false;
              setCallStatus('listening');
            });
          }
        };

        ws.onmessage = (event) => {
          if (!callActiveRef.current) return;
          try {
            const data = JSON.parse(event.data);

            if (data.type === 'conversation_state') {
              const nextState = data.state || 'ERROR';
              setConversationState(nextState);
              const statusByState = {
                LISTENING: 'listening',
                USER_SPEAKING: 'speaking',
                PROCESSING: 'processing',
                AI_THINKING: 'processing',
                AI_SPEAKING: 'ai_speaking',
                INTERRUPTED: 'speaking'
              };
              if (statusByState[nextState]) setCallStatus(statusByState[nextState]);
              if (nextState === 'INTERRUPTED') {
                stopActiveAudio();
                audioQueueRef.current = [];
              }
            } else if (data.type === 'partial_transcript') {
              const turnId = data.turn_id ?? currentTurnIdRef.current;
              const userMsgId = `user_${turnId}`;
              const seq = data.seq ?? 0;

              // Stale event rejection: if final transcript was already processed, ignore late partials
              if (processedEventsRef.current.has(`${turnId}_final`)) {
                return;
              }
              if (lastSeqByTurnRef.current[turnId] && seq < lastSeqByTurnRef.current[turnId]) {
                return;
              }
              lastSeqByTurnRef.current[turnId] = seq;

              setLivePartial(data.text);
              setCallStatus('speaking');
              if (!detectedAccent || detectedAccent.status !== 'detected') {
                setIsDetectingAccent(true);
              }
              if (data.first_partial_latency_ms) {
                setFdMetrics((m) => ({ ...m, timeToFirstPartial: data.first_partial_latency_ms }));
              }

              // Update the single user utterance bubble in-place
              if (data.text && data.text.trim()) {
                setChatHistory((prev) => {
                  const existingIndex = prev.findIndex((m) => m.id === userMsgId);
                  if (existingIndex >= 0) {
                    const updated = [...prev];
                    updated[existingIndex] = {
                      ...updated[existingIndex],
                      content: data.text.trim(),
                      isPartial: true,
                      seq
                    };
                    return updated;
                  } else {
                    return [
                      ...prev,
                      {
                        id: userMsgId,
                        role: 'user',
                        content: data.text.trim(),
                        turnId,
                        isPartial: true,
                        seq,
                        timestamp: Date.now()
                      }
                    ];
                  }
                });
              }
            } else if (data.type === 'recognition_status') {
              setRecognitionStatus(
                data.status === 'connected'
                  ? 'Deepgram Nova-3 connected'
                  : data.status === 'fallback_local'
                    ? `Deepgram unavailable — ${data.provider} fallback active`
                    : 'Recognition status updated'
              );
            } else if (data.type === 'accent_detected') {
              setDetectedAccent({
                accent: data.accent,
                code: data.code,
                flag: data.flag || '🌐',
                confidence: data.confidence,
                status: data.status,
                sampleDuration: data.sample_duration,
                latencyMs: data.latency_ms,
                suggestedTtsVoice: data.suggested_tts_voice,
                details: data.details,
                isSmoothed: data.is_smoothed
              });
              setIsDetectingAccent(false);
            } else if (data.type === 'voice_adapted') {
              setSelectedVoice(data.voice);
              localStorage.setItem('call_selected_voice', data.voice);
              setVoiceAdaptationNotice(`AI voice adapted to ${data.accent} (${data.voice})`);
              if (voiceAdaptTimerRef.current) clearTimeout(voiceAdaptTimerRef.current);
              voiceAdaptTimerRef.current = setTimeout(() => setVoiceAdaptationNotice(null), 4000);
            } else if (data.type === 'final_transcript') {
              const turnId = data.turn_id ?? currentTurnIdRef.current;
              const userMsgId = `user_${turnId}`;
              const finalKey = `${turnId}_final`;

              // Idempotency: ensure each finalized utterance is processed exactly once
              if (processedEventsRef.current.has(finalKey)) {
                return;
              }
              processedEventsRef.current.add(finalKey);

              setLivePartial('');
              currentTurnIdRef.current = turnId;
              tEndpointClientRef.current = performance.now();
              firstAudibleMeasuredRef.current = null;
              lastFinalUserTextRef.current = data.text || '';
              lastFinalMetricsRef.current = {
                timeToFinal: data.endpoint_latency_ms,
                audioDuration: data.audio_duration_sec,
                rtf: data.rtf
              };

              const text = (data.text || '').trim();
              if (text) {
                setChatHistory((prev) => {
                  const existingIndex = prev.findIndex((m) => m.id === userMsgId);
                  if (existingIndex >= 0) {
                    // Update the SAME message in-place. Never create another bubble!
                    const updated = [...prev];
                    updated[existingIndex] = {
                      ...updated[existingIndex],
                      content: text,
                      isPartial: false
                    };
                    return updated;
                  } else {
                    return [
                      ...prev,
                      {
                        id: userMsgId,
                        role: 'user',
                        content: text,
                        turnId,
                        isPartial: false,
                        timestamp: Date.now()
                      }
                    ];
                  }
                });
              }
              setCallStatus('processing');
              setFdMetrics((m) => ({
                ...m,
                timeToFinal: data.endpoint_latency_ms,
                timeToFirstToken: null,
                timeToFirstAudio: null,
                totalTurnLatency: null
              }));
            } else if (data.type === 'assistant_text_delta') {
              const turnId = data.turn_id ?? currentTurnIdRef.current;
              const asstMsgId = `asst_${turnId}`;

              setLiveAssistantText((prev) => prev + data.delta);
              if (data.time_to_first_token_ms) {
                setFdMetrics((m) => ({ ...m, timeToFirstToken: data.time_to_first_token_ms }));
              }

              // Update assistant reply incrementally in chatHistory
              setChatHistory((prev) => {
                const existingIndex = prev.findIndex((m) => m.id === asstMsgId);
                if (existingIndex >= 0) {
                  const updated = [...prev];
                  updated[existingIndex] = {
                    ...updated[existingIndex],
                    content: updated[existingIndex].content + data.delta,
                    isPartial: true
                  };
                  return updated;
                } else {
                  return [
                    ...prev,
                    {
                      id: asstMsgId,
                      role: 'assistant',
                      content: data.delta,
                      turnId,
                      isPartial: true,
                      timestamp: Date.now()
                    }
                  ];
                }
              });
            } else if (data.type === 'audio_chunk') {
              if (data.metrics?.time_to_first_audio_ms) {
                setFdMetrics((m) => ({ ...m, timeToFirstAudio: data.metrics.time_to_first_audio_ms }));
              }
              // Enqueue audio chunk for sequential playback
              audioQueueRef.current.push({
                base64: data.audio_base64,
                format: data.format,
                turnId: data.turn_id
              });
              if (!isPlayingAudioRef.current) {
                playNextAudioChunk();
              }
            } else if (data.type === 'turn_complete') {
              const turnId = data.turn_id ?? currentTurnIdRef.current;
              const asstMsgId = `asst_${turnId}`;
              setLiveAssistantText('');

              if (data.full_reply && data.full_reply.trim()) {
                const fullReply = data.full_reply.trim();
                setChatHistory((prev) => {
                  const existingIndex = prev.findIndex((m) => m.id === asstMsgId);
                  if (existingIndex >= 0) {
                    const updated = [...prev];
                    updated[existingIndex] = {
                      ...updated[existingIndex],
                      content: fullReply,
                      isPartial: false
                    };
                    return updated;
                  } else {
                    return [
                      ...prev,
                      {
                        id: asstMsgId,
                        role: 'assistant',
                        content: fullReply,
                        turnId,
                        isPartial: false,
                        timestamp: Date.now()
                      }
                    ];
                  }
                });
              }
              const metrics = data.metrics || {};
              const finalMetrics = lastFinalMetricsRef.current || {};
              if (metrics.total_turn_latency_ms) {
                setFdMetrics((m) => ({ ...m, totalTurnLatency: metrics.total_turn_latency_ms }));
              }
              if (onCallResult) {
                const sttSec = metrics.endpoint_latency_ms ? +(metrics.endpoint_latency_ms / 1000).toFixed(3) : (finalMetrics.timeToFinal ? +(finalMetrics.timeToFinal / 1000).toFixed(3) : null);
                const audioDur = metrics.audio_duration_sec || finalMetrics.audioDuration || null;
                const rtf = metrics.rtf || finalMetrics.rtf || (audioDur && sttSec ? +(sttSec / audioDur).toFixed(3) : null);
                const speedFactor = rtf && rtf > 0 ? +(1 / rtf).toFixed(1) : null;
                const groqFirstTokenMs = metrics.time_to_first_token_ms || null;
                const groqSec = metrics.groq_completion_ms ? +(metrics.groq_completion_ms / 1000).toFixed(3) : (groqFirstTokenMs ? +(groqFirstTokenMs / 1000).toFixed(3) : null);
                const ttsFirstChunkMs = metrics.tts_first_chunk_ms || metrics.time_to_first_audio_ms || null;
                const totalTurnMs = metrics.total_turn_latency_ms || null;
                const ttsSec = metrics.tts_total_ms ? +(metrics.tts_total_ms / 1000).toFixed(3) : (ttsFirstChunkMs ? +(ttsFirstChunkMs / 1000).toFixed(3) : null);
                const totalServerSec = totalTurnMs ? +(totalTurnMs / 1000).toFixed(3) : (sttSec && groqSec ? +(sttSec + groqSec).toFixed(3) : null);
                const firstAudibleMs = firstAudibleMeasuredRef.current || metrics.time_to_first_audio_ms || ttsFirstChunkMs || null;
                const firstPartialMs = metrics.first_partial_latency_ms || fdMetrics.timeToFirstPartial || null;
                const clientRttSec = firstAudibleMs ? +(firstAudibleMs / 1000).toFixed(3) : (metrics.voice_to_voice_rtt_ms ? +(metrics.voice_to_voice_rtt_ms / 1000).toFixed(3) : totalServerSec);

                onCallResult({
                  transcript: lastFinalUserTextRef.current || '',
                  reply: data.full_reply || '',
                  latency: {
                    audio_duration: audioDur,
                    stt_latency: sttSec,
                    first_partial_latency_ms: firstPartialMs,
                    endpoint_latency_ms: metrics.endpoint_latency_ms || null,
                    groq_first_token_ms: groqFirstTokenMs,
                    groq_latency: groqSec,
                    tts_first_chunk_ms: ttsFirstChunkMs,
                    tts_latency: ttsSec,
                    first_audible_ms: firstAudibleMs,
                    total_server_time: totalServerSec,
                    client_rtt: clientRttSec,
                    rtf: rtf,
                    speed_factor: speedFactor
                  }
                });
              }
            } else if (data.type === 'error') {
              setLiveAssistantText('');
              setCallStatus('listening');
              if (data.message) {
                setChatHistory((prev) => [
                  ...prev,
                  { id: `sys_err_${Date.now()}`, role: 'system_notification', content: `⚠️ ${data.message}` }
                ]);
              }
            } else if (data.type === 'interrupted') {
              const turnId = data.turn_id ?? currentTurnIdRef.current;
              const asstMsgId = `asst_${turnId}`;
              // Real-Time Barge-In Interruption: halt active audio, clear queue
              const tStopStart = performance.now();
              stopActiveAudio();
              const clientStopLatency = Math.round((performance.now() - tStopStart) * 100) / 100;
              const serverLatency = typeof data.interruption_latency_ms === 'number' ? data.interruption_latency_ms : 0.5;
              const combinedLatency = Math.round((serverLatency + clientStopLatency) * 10) / 10;
              setBargeInNotice(`⚡ User barge-in: Speech interrupted (detect: ${serverLatency}ms, stop: ${clientStopLatency}ms)`);
              setFdMetrics((m) => ({ ...m, interruptionLatency: combinedLatency }));
              setCallStatus('speaking');
              setLiveAssistantText('');

              // Finalize partial assistant message on interruption
              setChatHistory((prev) => {
                const existingIndex = prev.findIndex((m) => m.id === asstMsgId);
                if (existingIndex >= 0 && prev[existingIndex].isPartial) {
                  const updated = [...prev];
                  updated[existingIndex] = {
                    ...updated[existingIndex],
                    isPartial: false
                  };
                  return updated;
                }
                return prev;
              });

              if (bargeInTimerRef.current) clearTimeout(bargeInTimerRef.current);
              bargeInTimerRef.current = setTimeout(() => setBargeInNotice(null), 3500);
            }
          } catch (e) {
            console.error('Error parsing WebSocket message:', e);
          }
        };

        ws.onerror = (err) => {
          console.error(`[VoiceCall] WebSocket call error on ${activeWsUrl}:`, err);
          if (!callActiveRef.current && activeWsUrl !== fallbackWsUrl) {
            console.warn(`[VoiceCall] Primary WebSocket ${activeWsUrl} failed. Falling back to proxied ${fallbackWsUrl}...`);
            const prevOnOpen = ws.onopen;
            const prevOnMessage = ws.onmessage;
            const prevOnClose = ws.onclose;
            activeWsUrl = fallbackWsUrl;
            try {
              ws.onopen = null;
              ws.onmessage = null;
              ws.onerror = null;
              ws.onclose = null;
              ws.close();
            } catch (_) {}
            const fallbackWs = new WebSocket(fallbackWsUrl);
            wsRef.current = fallbackWs;
            fallbackWs.onopen = prevOnOpen;
            fallbackWs.onmessage = prevOnMessage;
            fallbackWs.onerror = (fErr) => {
              console.error(`[VoiceCall] Fallback WebSocket also failed on ${fallbackWsUrl}:`, fErr);
              setChatHistory((prev) => [
                ...prev,
                { role: 'system_notification', content: 'WebSocket connection error. Check server status.' }
              ]);
            };
            fallbackWs.onclose = prevOnClose;
            return;
          }

          setChatHistory((prev) => [
            ...prev,
            { role: 'system_notification', content: 'WebSocket connection error. Check server status.' }
          ]);
        };

        ws.onclose = (event) => {
          console.log(`[VoiceCall] WebSocket closed (code=${event.code}, reason='${event.reason}')`);
          if (callActiveRef.current) {
            const canReconnect = reconnectAttemptsRef.current < 3 && event.code !== 1000;
            if (canReconnect) {
              reconnectAttemptsRef.current += 1;
              const delayMs = 500 * reconnectAttemptsRef.current;
              setConversationState('RECONNECTING');
              setCallStatus('processing');
              setChatHistory((prev) => [
                ...prev,
                { id: `sys_reconnect_${Date.now()}`, role: 'system_notification', content: `Connection lost. Reconnecting (${reconnectAttemptsRef.current}/3)…` }
              ]);
              reconnectTimerRef.current = setTimeout(() => {
                if (callActiveRef.current) handleStartCall({ reconnecting: true });
              }, delayMs);
            } else {
              setConversationState('ERROR');
              setCallStatus('listening');
              setChatHistory((prev) => [...prev, { id: `sys_ws_error_${Date.now()}`, role: 'system_notification', content: 'Connection could not be restored. Please start a new call.' }]);
              cleanupCallAudio();
              setCallState('idle');
            }
          }
        };

        // Continuous microphone processor. 1024 samples keeps transport/decoder
        // buffering near 21 ms at 48 kHz without changing the audio model.
        // Stays permanently active even during assistant speech (true full-duplex)
        const processor = audioCtx.createScriptProcessor(1024, 1, 1);
        scriptProcessorRef.current = processor;

        processor.onaudioprocess = (e) => {
          if (!callActiveRef.current || isMutedRef.current) return;
          if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;

          const inputData = e.inputBuffer.getChannelData(0);
          const pcm16 = resampleAndEncodePCM16(inputData, audioCtx.sampleRate, 16000);
          if (pcm16.length > 0) {
            wsRef.current.send(pcm16.buffer);
          }
        };

        // Connect through a zero-gain node to keep onaudioprocess running
        // without routing raw microphone audio back out of the speaker (prevents acoustic echo feedback loop)
        const muteGain = audioCtx.createGain();
        muteGain.gain.value = 0.0;
        source.connect(processor);
        processor.connect(muteGain);
        muteGain.connect(audioCtx.destination);

      } else {
        // --- CLASSIC TURN-BASED FASTER-WHISPER PIPELINE ---
        setTimeout(() => {
          setCallState('active');
          callActiveRef.current = true;

          const isHindi = selectedVoice.startsWith('hi-') || sttLanguage === 'hi';
          const greeting = isHindi
            ? 'नमस्ते! कॉल करने के लिए धन्यवाद। मैं आज आपकी क्या सहायता कर सकता हूँ?'
            : 'Hello! Thanks for calling. How can I help you today?';

          setChatHistory((prev) => [
            ...prev,
            { role: 'assistant', content: greeting }
          ]);

          startVADLoop();

          isAiSpeakingRef.current = true;
          setCallStatus('ai_speaking');

          const voiceObj = CALL_VOICES.find((v) => v.id === selectedVoice);
          const ttsEngine = voiceObj?.engine || (selectedVoice.includes('Neural') ? 'edge_tts' : 'kokoro');

          synthesizeSpeech(greeting, { voice: selectedVoice, engine: ttsEngine })
            .then((res) => {
              playAudioFromBase64(res.audio_base64, res.format, () => {
                isAiSpeakingRef.current = false;
                startListeningRecorder();
              });
            })
            .catch(() => {
              isAiSpeakingRef.current = false;
              startListeningRecorder();
            });
        }, 1000);
      }
    } catch (err) {
      alert(`Call failed: ${err.message}`);
      setCallState('idle');
      setConversationState('ERROR');
      cleanupCallAudio();
    }
  };


  /**
   * End Call: Hang up and release all audio hardware
   */
  const handleEndCall = () => {
    cleanupCallAudio();
    setCallState('ended');
    setConversationState('ENDED');
    setTimeout(() => {
      setCallState('idle');
    }, 2000);
  };

  /**
   * Toggle Microphone Mute during Call
   */
  const handleToggleMute = () => {
    const nextMuted = !isMuted;
    setIsMuted(nextMuted);
    isMutedRef.current = nextMuted;

    if (streamRef.current) {
      streamRef.current.getAudioTracks().forEach((track) => {
        track.enabled = !nextMuted;
      });
    }

    if (nextMuted) {
      setCallStatus('muted');
    } else {
      setCallStatus('listening');
      startListeningRecorder();
    }
  };

  /**
   * Mode switcher: 'call' | 'stt' | 'tts'
   */
  const handleModeChange = (mode) => {
    if (callState === 'active') {
      cleanupCallAudio();
      setCallState('idle');
    }
    stopAudioPlayback();
    setSimulatorMode(mode);
    localStorage.setItem('sim_active_mode', mode);
    if (onModeChange) {
      onModeChange(mode);
    }
  };

  /**
   * Dedicated STT Transcriber: Start Microphone Recording
   */
  const handleStartSttRecording = async () => {
    setSttError(null);
    setSttTranscriptResult(null);
    sttChunksRef.current = [];

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: 16000,
          echoCancellation: true,
          noiseSuppression: true
        }
      });
      sttStreamRef.current = stream;

      const audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
      sttAudioContextRef.current = audioCtx;
      const source = audioCtx.createMediaStreamSource(stream);
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 256;
      source.connect(analyser);
      sttAnalyserRef.current = analyser;

      const canvas = sttCanvasRef.current;
      if (canvas) {
        const ctx = canvas.getContext('2d');
        const bufferLen = analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLen);

        const draw = () => {
          sttAnimRef.current = requestAnimationFrame(draw);
          analyser.getByteFrequencyData(dataArray);
          ctx.clearRect(0, 0, canvas.width, canvas.height);
          const barWidth = (canvas.width / bufferLen) * 2;
          let x = 0;
          for (let i = 0; i < bufferLen; i++) {
            const barHeight = (dataArray[i] / 255) * canvas.height;
            ctx.fillStyle = `rgba(56, 189, 248, ${0.4 + (dataArray[i] / 255) * 0.6})`;
            ctx.fillRect(x, canvas.height - barHeight, barWidth - 1, barHeight);
            x += barWidth;
          }
        };
        draw();
      }

      const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
        ? 'audio/webm;codecs=opus'
        : 'audio/webm';
      const recorder = new MediaRecorder(stream, { mimeType });
      sttRecorderRef.current = recorder;

      recorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) {
          sttChunksRef.current.push(e.data);
        }
      };

      recorder.start(100);
      setSttIsRecording(true);
      setSttRecordDuration(0);

      sttTimerRef.current = setInterval(() => {
        setSttRecordDuration((d) => d + 1);
      }, 1000);
    } catch (err) {
      setSttError(err.message || 'Microphone access denied');
    }
  };

  /**
   * Dedicated STT Transcriber: Stop Microphone Recording & Transcribe
   */
  const handleStopSttRecording = async () => {
    if (!sttRecorderRef.current || sttRecorderRef.current.state === 'inactive') return;

    if (sttTimerRef.current) {
      clearInterval(sttTimerRef.current);
      sttTimerRef.current = null;
    }
    if (sttAnimRef.current) {
      cancelAnimationFrame(sttAnimRef.current);
      sttAnimRef.current = null;
    }

    setSttIsRecording(false);
    setSttIsProcessing(true);

    const recorder = sttRecorderRef.current;
    recorder.onstop = async () => {
      if (sttStreamRef.current) {
        sttStreamRef.current.getTracks().forEach((t) => t.stop());
        sttStreamRef.current = null;
      }
      if (sttAudioContextRef.current && sttAudioContextRef.current.state !== 'closed') {
        sttAudioContextRef.current.close().catch(() => { });
        sttAudioContextRef.current = null;
      }

      const blob = new Blob(sttChunksRef.current, { type: recorder.mimeType || 'audio/webm' });
      sttChunksRef.current = [];

      if (!blob || blob.size === 0) {
        setSttIsProcessing(false);
        setSttError('No audio recorded');
        return;
      }

      try {
        const data = await transcribeAudio(blob, {
          language: sttLanguage,
          model: sttModel,
          engine: sttEngine
        });
        setSttTranscriptResult(data);
      } catch (err) {
        setSttError(err.message || 'Transcription failed');
      } finally {
        setSttIsProcessing(false);
      }
    };

    recorder.stop();
  };

  const handleSttFileUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setSttError(null);
    setSttTranscriptResult(null);
    setSttIsProcessing(true);
    try {
      const data = await transcribeAudio(file, {
        language: sttLanguage,
        model: sttModel,
        engine: sttEngine
      });
      setSttTranscriptResult(data);
    } catch (err) {
      setSttError(err.message || 'Failed to transcribe audio file');
    } finally {
      setSttIsProcessing(false);
      e.target.value = '';
    }
  };

  const handleCopySttTranscript = () => {
    if (!sttTranscriptResult?.transcript) return;
    navigator.clipboard.writeText(sttTranscriptResult.transcript);
    setSttCopied(true);
    setTimeout(() => setSttCopied(false), 2000);
  };

  /**
   * Dedicated TTS Generator: Synthesize speech from text
   */
  const handleGenerateTtsSpeech = async () => {
    if (!ttsInputText.trim() || ttsIsGenerating) return;
    setTtsIsGenerating(true);
    setTtsError(null);
    stopAudioPlayback();

    try {
      const data = await synthesizeSpeech(ttsInputText.trim(), {
        voice: selectedVoice,
        pitch: selectedPitch,
        speed: ttsSpeed,
        engine: activeTtsEngine
      });

      setTtsResult(data);

      if (data.audio_base64) {
        const byteCharacters = atob(data.audio_base64);
        const byteNumbers = new Array(byteCharacters.length);
        for (let i = 0; i < byteCharacters.length; i++) {
          byteNumbers[i] = byteCharacters.charCodeAt(i);
        }
        const byteArray = new Uint8Array(byteNumbers);
        const mimeType = data.media_type || (data.format === 'mp3' ? 'audio/mpeg' : 'audio/wav');
        const blob = new Blob([byteArray], { type: mimeType });
        if (ttsAudioUrl) URL.revokeObjectURL(ttsAudioUrl);
        const url = URL.createObjectURL(blob);
        setTtsAudioUrl(url);

        playAudioFromBase64(data.audio_base64, data.format);
      }
    } catch (err) {
      setTtsError(err.message || 'Speech synthesis failed');
    } finally {
      setTtsIsGenerating(false);
    }
  };

  const TTS_PRESETS = [
    { label: '📞 Helpline Greeting', text: 'Hello! Thank you for calling customer service. How may I assist you with your inquiry today?' },
    { label: '🇮🇳 Hindi Welcome', text: 'नमस्ते! हमारी ग्राहक सहायता सेवा में आपका स्वागत है। आज हम आपकी क्या मदद कर सकते हैं?' },
    { label: '🔐 OTP Code', text: 'Your one-time phone verification code is 8 4 1 9 2 0. Please do not share this PIN with anyone.' },
    { label: '✈️ Boarding Call', text: 'Attention passengers on Flight 6E 402 to Mumbai, now boarding at Gate Number 12.' },
    { label: '📦 Delivery Update', text: 'Your package with tracking number 9 8 3 4 5 has arrived at your local hub and will be delivered today.' }
  ];

  /**
   * Quick Standalone TTS Tester
   */
  const handleTestTTS = async () => {
    if (!testText.trim()) return;
    setIsTestingTTS(true);
    stopAudioPlayback();
    try {
      const data = await synthesizeSpeech(testText, {
        voice: selectedVoice,
        pitch: selectedPitch,
        speed: ttsSpeed,
        engine: activeTtsEngine
      });
      playAudioFromBase64(data.audio_base64, data.format, () => {
        setIsTestingTTS(false);
      });
    } catch (err) {
      alert(`TTS test failed: ${err.message}`);
      setIsTestingTTS(false);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
      {/* Active Studio Status Banner */}
      <div
        className="glass-panel"
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          flexWrap: 'wrap',
          gap: '0.85rem',
          padding: '0.7rem 1.25rem',
          background: 'var(--bg-card)',
          borderRadius: '12px',
          border:
            simulatorMode === 'stt'
              ? '1px solid var(--accent-stt-border)'
              : simulatorMode === 'tts'
                ? '1px solid var(--accent-tts-border)'
                : '1px solid var(--accent-call-border)',
          boxShadow: 'var(--shadow-sm)'
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem' }}>
          {simulatorMode === 'stt' ? (
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem', color: 'var(--accent-stt)', fontWeight: 700, fontSize: '0.92rem' }}>
              <Mic size={18} />
              <span>Voice to Text (STT) Studio</span>
            </div>
          ) : simulatorMode === 'tts' ? (
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem', color: 'var(--accent-tts)', fontWeight: 700, fontSize: '0.92rem' }}>
              <Volume2 size={18} />
              <span>Text to Voice (TTS) Speech Studio</span>
            </div>
          ) : (
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem', color: 'var(--accent-call)', fontWeight: 700, fontSize: '0.92rem' }}>
              <Phone size={18} />
              <span>Interactive Phone Call Agent</span>
            </div>
          )}
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.76rem', color: 'var(--text-muted)' }}>
          <span style={{ display: 'inline-block', width: '7px', height: '7px', borderRadius: '50%', background: '#10b981' }} />
          <span>Backend Online ({health?.device ? health.device.toUpperCase() : 'Ready'})</span>
        </div>
      </div>

      {/* Phone Call Simulator Model Selection Deck */}
      <section
        className="glass-panel"
        style={{
          padding: '1.1rem 1.35rem',
          display: 'flex',
          flexDirection: 'column',
          gap: '0.9rem',
          background: 'var(--bg-card)',
          border: '1px solid var(--border)',
          boxShadow: 'var(--shadow-md)'
        }}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.6rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.55rem' }}>
            <Radio size={16} color="var(--accent-call)" />
            <span style={{ fontSize: '0.92rem', fontWeight: 700, color: 'var(--text-main)', letterSpacing: '0.3px' }}>
              Call Pipeline Model Configuration
            </span>
          </div>

          {/* Active Flow Badges */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.74rem', fontFamily: 'var(--font-mono)', flexWrap: 'wrap' }}>
            <span className="badge badge-green" title="Active Voice to Text Model">
              <Mic size={11} /> STT: {sttModel}
            </span>
            <span style={{ color: 'var(--text-muted)' }}>➔</span>
            <span className="badge badge-purple" title="LLM Agent">
              <Bot size={11} /> Groq
            </span>
            <span style={{ color: 'var(--text-muted)' }}>➔</span>
            <span className="badge badge-blue" title="Active Text to Voice Model">
              <Volume2 size={11} /> TTS: {activeTtsEngine}
            </span>
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(290px, 1fr))', gap: '1rem' }}>
          {/* 1. Voice-to-Text (STT) Dual Engine & Model Control Card */}
          <div
            style={{
              display: 'flex',
              flexDirection: 'column',
              gap: '0.65rem',
              padding: '0.85rem 1rem',
              borderRadius: '12px',
              background: 'var(--bg-subtle)',
              border: '1px solid var(--border)'
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.4rem' }}>
              <label style={{ fontSize: '0.82rem', fontWeight: 600, color: 'var(--accent-call)', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                <Mic size={15} /> Voice to Text (STT) Engine & Model
              </label>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                  {health?.device ? `Device: ${health.device.toUpperCase()}` : 'CPU/GPU'}
                </span>
                {onOpenSherpa && (
                  <button
                    type="button"
                    onClick={onOpenSherpa}
                    style={{
                      background: 'transparent',
                      border: 'none',
                      color: 'var(--accent-call)',
                      fontSize: '0.7rem',
                      fontWeight: 600,
                      cursor: 'pointer',
                      textDecoration: 'underline'
                    }}
                    title="Configure models, download weights, or view benchmarks"
                  >
                    Configure
                  </button>
                )}
              </div>
            </div>

            {/* STT Engine Selector */}
            <div style={{ display: 'flex', gap: '0.35rem', background: 'var(--bg-card)', padding: '0.2rem', borderRadius: '8px', border: '1px solid var(--border)' }}>
              <button
                type="button"
                onClick={() => handleSttEngineChange('sherpa')}
                style={{
                  flex: 1,
                  padding: '0.35rem 0.5rem',
                  borderRadius: '6px',
                  border: 'none',
                  background: sttEngine === 'sherpa' ? 'rgba(168, 85, 247, 0.16)' : 'transparent',
                  color: sttEngine === 'sherpa' ? '#9333ea' : 'var(--text-muted)',
                  fontWeight: sttEngine === 'sherpa' ? 700 : 500,
                  fontSize: '0.75rem',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: '0.3rem',
                  transition: 'all 0.2s ease'
                }}
              >
                <span>⚡ Sherpa-ONNX</span>
              </button>
              <button
                type="button"
                onClick={() => handleSttEngineChange('whisper')}
                style={{
                  flex: 1,
                  padding: '0.35rem 0.5rem',
                  borderRadius: '6px',
                  border: 'none',
                  background: sttEngine === 'whisper' ? 'rgba(2, 132, 199, 0.16)' : 'transparent',
                  color: sttEngine === 'whisper' ? '#0284c7' : 'var(--text-muted)',
                  fontWeight: sttEngine === 'whisper' ? 700 : 500,
                  fontSize: '0.75rem',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: '0.3rem',
                  transition: 'all 0.2s ease'
                }}
              >
                <span>🎙️ Faster-Whisper</span>
              </button>
            </div>

              <button type="button" onClick={() => handleSttEngineChange('deepgram')} style={{ flex: 1, padding: '0.35rem 0.5rem', borderRadius: '6px', border: 'none', background: sttEngine === 'deepgram' ? 'rgba(16, 185, 129, 0.16)' : 'transparent', color: sttEngine === 'deepgram' ? '#059669' : 'var(--text-muted)', fontWeight: sttEngine === 'deepgram' ? 700 : 500, fontSize: '0.75rem', cursor: 'pointer' }}>
                <span>Deepgram Nova-3</span>
              </button>
            {/* Model & Language Dropdowns */}
            <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
              {/* STT Model Dropdown (Filtered by Engine) */}
              <select
                value={sttModel}
                onChange={(e) => handleSttModelChange(e.target.value)}
                title={`Select ${sttEngine === 'sherpa' ? 'Sherpa-ONNX' : 'Whisper'} Model`}
                style={{
                  flex: 1.4,
                  minWidth: '160px',
                  padding: '0.52rem 0.65rem',
                  borderRadius: '7px',
                  background: 'var(--bg-input)',
                  border: '1px solid var(--border)',
                  color: 'var(--text-main)',
                  fontSize: '0.82rem',
                  fontFamily: 'var(--font-mono)',
                  outline: 'none',
                  cursor: 'pointer'
                }}
              >
                {sttEngine === 'sherpa' ? (
                  SHERPA_MODELS.map((m) => (
                    <option key={m.id} value={m.id}>
                      {m.type === 'streaming' ? '⚡' : '📦'} {m.label} • {m.status || 'Ready'}
                    </option>
                  ))
                ) : sttEngine === 'deepgram' ? (
                  DEEPGRAM_MODELS.map((m) => <option key={m.id} value={m.id}>Deepgram {m.label} • {m.status}</option>)
                ) : (
                  WHISPER_MODELS.map((m) => (
                    <option key={m.id} value={m.id}>
                      🎙️ {m.label} • {m.status || 'Ready'}
                    </option>
                  ))
                )}
              </select>

              {/* STT Language Dropdown */}
              <select
                value={sttLanguage}
                onChange={(e) => handleSttLanguageChange(e.target.value)}
                title="Select Caller Language"
                style={{
                  flex: 1,
                  minWidth: '110px',
                  padding: '0.52rem 0.65rem',
                  borderRadius: '7px',
                  background: 'var(--bg-input)',
                  border: '1px solid var(--border)',
                  color: 'var(--text-main)',
                  fontSize: '0.82rem',
                  outline: 'none',
                  cursor: 'pointer'
                }}
              >
                {STT_LANGUAGES.filter((l) => sttEngine !== 'deepgram' || l.code === 'en').map((l) => (
                  <option key={l.code} value={l.code}>
                    {l.label}
                  </option>
                ))}
              </select>
            </div>

            {/* Dynamic Model Type & Compatibility Info */}
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: '0.7rem' }}>
              <span style={{
                color: (sttEngine === 'sherpa' && (!sttModel || sttModel.includes('streaming'))) ? '#10b981' : '#f59e0b',
                fontWeight: 600,
                display: 'flex',
                alignItems: 'center',
                gap: '0.3rem'
              }}>
                {sttEngine === 'deepgram'
                  ? 'Deepgram Nova-3 Streaming (English)'
                  : (sttEngine === 'sherpa' && (!sttModel || sttModel.includes('streaming')))
                  ? '⚡ Streaming Full-Duplex (WebSocket sub-100ms)'
                  : '📦 Turn-Based Offline Pipeline'}
              </span>
              {sttEngine === 'sherpa' && sttLanguage !== 'en' && sttLanguage !== 'auto' && (
                <span style={{ color: '#ef4444', fontWeight: 600 }}>
                  ⚠️ Sherpa is English-only. Switch to Whisper for Indic.
                </span>
              )}
            </div>
            <div style={{ fontSize: '0.68rem', color: sttEngine === 'deepgram' ? '#059669' : 'var(--text-muted)' }}>
              Recognition: {sttEngine === 'deepgram' ? recognitionStatus : 'Local recognition ready'}
            </div>
          </div>

          {/* 2. Simplified Natural Human Voice Selector Card */}
          <div
            style={{
              display: 'flex',
              flexDirection: 'column',
              gap: '0.75rem',
              padding: '0.95rem 1.15rem',
              borderRadius: '12px',
              background: 'var(--bg-subtle)',
              border: '1px solid var(--accent-tts-border)',
              boxShadow: 'var(--shadow-sm)'
            }}
          >
            {/* Header: Title & Play Sample button */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.5rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem' }}>
                <Volume2 size={17} color="var(--accent-tts)" />
                <span style={{ fontSize: '0.9rem', fontWeight: 700, color: 'var(--text-main)' }}>
                  Voice Persona
                </span>
                <span className="badge badge-purple" style={{ fontSize: '0.7rem' }}>
                  {selectedVoiceObj?.category || 'Adult'}
                </span>
                <span className="badge badge-blue" style={{ fontSize: '0.7rem' }}>
                  {selectedVoiceObj?.accent}
                </span>
              </div>

              {/* Play Sample Button */}
              <button
                type="button"
                onClick={() => handlePlayVoiceSample(selectedVoiceObj)}
                title="Listen to audio preview sample"
                style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: '0.35rem',
                  padding: '0.38rem 0.85rem',
                  borderRadius: '7px',
                  background: previewPlayingVoiceId === selectedVoiceObj?.id ? '#ef4444' : 'var(--accent-tts)',
                  color: '#ffffff',
                  border: 'none',
                  fontSize: '0.78rem',
                  fontWeight: 600,
                  cursor: 'pointer',
                  transition: 'all 0.18s ease'
                }}
              >
                {previewPlayingVoiceId === selectedVoiceObj?.id ? (
                  <>
                    <Square size={12} />
                    <span>Stop</span>
                  </>
                ) : (
                  <>
                    <Play size={12} />
                    <span>Play Sample</span>
                  </>
                )}
              </button>
            </div>

            {/* Selector Row: Category Filter + Voice Selector */}
            <div style={{ display: 'grid', gridTemplateColumns: 'minmax(130px, 1fr) minmax(200px, 2fr)', gap: '0.55rem' }}>
              {/* Category Dropdown */}
              <select
                value={selectedVoiceCategory}
                onChange={(e) => {
                  const cat = e.target.value;
                  setSelectedVoiceCategory(cat);
                  const inCat = humanVoiceCatalog.filter((v) => cat === 'All' || v.category === cat);
                  if (inCat.length > 0 && !inCat.some((v) => v.id === selectedVoice)) {
                    handleVoiceChange(inCat[0].id);
                  }
                }}
                title="Filter by demographic category"
                style={{
                  padding: '0.52rem 0.65rem',
                  borderRadius: '7px',
                  background: 'var(--bg-input)',
                  border: '1px solid var(--border)',
                  color: 'var(--text-main)',
                  fontSize: '0.82rem',
                  fontWeight: 600,
                  outline: 'none',
                  cursor: 'pointer'
                }}
              >
                {VOICE_CATEGORIES.map((cat) => (
                  <option key={cat} value={cat}>
                    {cat === 'All' ? `All Categories (${humanVoiceCatalog.length})` : `${cat} (${humanVoiceCatalog.filter((v) => v.category === cat).length})`}
                  </option>
                ))}
              </select>

              {/* Voice Persona Dropdown */}
              <select
                value={selectedVoice}
                onChange={(e) => handleVoiceChange(e.target.value)}
                title="Select Voice"
                style={{
                  padding: '0.52rem 0.65rem',
                  borderRadius: '7px',
                  background: 'var(--bg-input)',
                  border: '1px solid var(--accent-tts-border)',
                  color: 'var(--text-main)',
                  fontSize: '0.82rem',
                  fontWeight: 600,
                  outline: 'none',
                  cursor: 'pointer'
                }}
              >
                {filteredVoices.map((v) => (
                  <option key={v.id} value={v.id}>
                    {v.name} ({v.gender}, {v.category}) — {v.accent}
                  </option>
                ))}
              </select>
            </div>

            {/* Voice Style Summary & Latency Tag */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '0.74rem', color: 'var(--text-muted)' }}>
              <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: '75%' }}>
                💬 <strong>{selectedVoiceObj?.name}</strong>: {selectedVoiceObj?.style}
              </span>
              {previewLatency && previewLatency.voiceId === selectedVoice && (
                <span style={{ color: '#10b981', fontFamily: 'var(--font-mono)', fontWeight: 600, flexShrink: 0 }}>
                  ⚡ {previewLatency.latencyMs}ms TTFA
                </span>
              )}
            </div>

            {/* Pitch & Speed Control Row */}
            <div style={{ display: 'grid', gridTemplateColumns: 'minmax(140px, 1fr) minmax(180px, 1.2fr)', gap: '0.75rem', alignItems: 'center', paddingTop: '0.35rem', borderTop: '1px dashed var(--border)' }}>
              {/* Pitch Selector Dropdown */}
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                <span style={{ fontSize: '0.73rem', color: 'var(--text-muted)', whiteSpace: 'nowrap' }}>
                  Pitch:
                </span>
                <select
                  value={selectedPitch}
                  onChange={(e) => handlePitchChange(e.target.value)}
                  style={{
                    flex: 1,
                    padding: '0.36rem 0.5rem',
                    borderRadius: '6px',
                    background: 'var(--bg-input)',
                    border: '1px solid var(--border)',
                    color: 'var(--text-main)',
                    fontSize: '0.76rem',
                    cursor: 'pointer'
                  }}
                >
                  {PITCH_OPTIONS.map((po) => (
                    <option key={po.value} value={po.value}>
                      {po.label}
                    </option>
                  ))}
                </select>
              </div>

              {/* Speed Slider */}
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem' }}>
                <span style={{ fontSize: '0.73rem', color: 'var(--text-muted)', whiteSpace: 'nowrap' }}>
                  Speed:
                </span>
                <input
                  type="range"
                  min="0.8"
                  max="1.5"
                  step="0.05"
                  value={ttsSpeed}
                  onChange={(e) => handleSpeedChange(parseFloat(e.target.value))}
                  style={{ flex: 1, cursor: 'pointer' }}
                />
                <span style={{ fontSize: '0.76rem', fontWeight: 700, color: 'var(--accent-tts)', fontFamily: 'var(--font-mono)', minWidth: '34px' }}>
                  {ttsSpeed.toFixed(2)}x
                </span>
                <button
                  type="button"
                  onClick={() => handleSpeedChange(1.0)}
                  title="Reset to 1.0x"
                  style={{
                    background: 'transparent',
                    border: 'none',
                    color: 'var(--text-muted)',
                    fontSize: '0.68rem',
                    cursor: 'pointer',
                    textDecoration: 'underline',
                    padding: 0
                  }}
                >
                  Reset
                </button>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ======================================================== */}
      {/* MODE 1: PHONE CALL AGENT (FULL DUPLEX)                    */}
      {/* ======================================================== */}
      {simulatorMode === 'call' && (
        <>
          <section
            className="glass-panel"
            style={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              padding: '2.2rem 1.5rem',
              position: 'relative',
              background:
                callState === 'active'
                  ? callStatus === 'ai_speaking'
                    ? 'radial-gradient(circle at 50% 20%, rgba(245, 158, 11, 0.12), var(--bg-card-solid))'
                    : 'radial-gradient(circle at 50% 20%, var(--accent-call-bg), var(--bg-card-solid))'
                  : 'var(--bg-card)',
              borderRadius: '16px',
              border:
                callState === 'active'
                  ? callStatus === 'ai_speaking'
                    ? '1px solid rgba(245, 158, 11, 0.4)'
                    : '1px solid var(--accent-call-border)'
                  : '1px solid var(--border)',
              boxShadow:
                callState === 'active'
                  ? callStatus === 'ai_speaking'
                    ? '0 0 35px rgba(245, 158, 11, 0.2)'
                    : '0 0 35px var(--accent-call-glow)'
                  : 'var(--shadow-md)',
              transition: 'all 0.4s ease'
            }}
          >
            {/* Call Status Badge */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', marginBottom: '0.5rem' }}>
              <span
                style={{
                  width: '10px',
                  height: '10px',
                  borderRadius: '50%',
                  background:
                    callState === 'active'
                      ? isMuted
                        ? '#ef4444'
                        : '#34d399'
                      : callState === 'calling'
                        ? '#fbbf24'
                        : '#64748b',
                  boxShadow: callState === 'active' ? '0 0 10px #34d399' : undefined
                }}
              />
              <span
                style={{
                  fontFamily: 'var(--font-mono)',
                  fontSize: '0.9rem',
                  fontWeight: 600,
                  color: callState === 'active' ? '#34d399' : 'var(--text-muted)',
                  textTransform: 'uppercase',
                  letterSpacing: '1px'
                }}
              >
                {callState === 'idle' && 'Call Simulator Ready'}
                {callState === 'calling' && 'Connecting Call...'}
                {callState === 'active' && `Call in Progress • ${formatCallTime(callDuration)}`}
                {callState === 'ended' && 'Call Ended'}
              </span>
              {callState === 'active' && (
                <span style={{ fontSize: '0.68rem', color: 'var(--text-muted)', marginLeft: '0.45rem' }}>
                  {conversationState.replaceAll('_', ' ')}
                </span>
              )}
            </div>

            {/* Center Telephony Avatar with Dynamic Audio Rings */}
            <div
              style={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                margin: '1.2rem 0 1.2rem',
                position: 'relative'
              }}
            >
              {callState === 'active' && (
                <div
                  style={{
                    position: 'absolute',
                    top: '-10px',
                    width: '110px',
                    height: '110px',
                    borderRadius: '50%',
                    border: `2px solid ${callStatus === 'ai_speaking'
                        ? 'rgba(245, 158, 11, 0.4)'
                        : callStatus === 'speaking'
                          ? 'rgba(56, 189, 248, 0.5)'
                          : 'rgba(52, 211, 153, 0.25)'
                      }`,
                    transform: `scale(${1 + audioLevel * 0.003})`,
                    transition: 'transform 0.1s ease',
                    pointerEvents: 'none'
                  }}
                />
              )}

              <div
                onClick={callStatus === 'ai_speaking' ? handleBargeIn : undefined}
                title={callStatus === 'ai_speaking' ? 'Click to interrupt AI' : undefined}
                style={{
                  width: '92px',
                  height: '92px',
                  borderRadius: '50%',
                  background:
                    callStatus === 'ai_speaking'
                      ? 'linear-gradient(135deg, #f59e0b, #ef4444)'
                      : callStatus === 'speaking'
                        ? 'linear-gradient(135deg, #38bdf8, #6366f1)'
                        : callStatus === 'processing'
                          ? 'linear-gradient(135deg, #a855f7, #6366f1)'
                          : isMuted
                            ? 'linear-gradient(135deg, #ef4444, #7f1d1d)'
                            : callState === 'active'
                              ? 'linear-gradient(135deg, #0284c7, #0f766e)'
                              : 'linear-gradient(135deg, #1e293b, #334155)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  boxShadow:
                    callStatus === 'ai_speaking'
                      ? '0 0 35px rgba(245, 158, 11, 0.6)'
                      : callStatus === 'speaking'
                        ? '0 0 35px rgba(56, 189, 248, 0.6)'
                        : callStatus === 'processing'
                          ? '0 0 30px rgba(168, 85, 247, 0.5)'
                          : '0 8px 24px rgba(0,0,0,0.4)',
                  transition: 'all 0.3s ease',
                  transform:
                    callStatus === 'ai_speaking' || callStatus === 'speaking' ? 'scale(1.06)' : 'scale(1)',
                  cursor: callStatus === 'ai_speaking' ? 'pointer' : 'default'
                }}
              >
                {callStatus === 'ai_speaking' ? (
                  <Volume2 size={44} color="#ffffff" className="pulse-icon" />
                ) : callStatus === 'speaking' ? (
                  <Activity size={44} color="#ffffff" />
                ) : callStatus === 'processing' ? (
                  <Sparkles size={42} color="#ffffff" />
                ) : isMuted ? (
                  <MicOff size={42} color="#ffffff" />
                ) : callState === 'active' ? (
                  <Mic size={42} color="#ffffff" />
                ) : (
                  <Bot size={44} color="#94a3b8" />
                )}
              </div>

              <div style={{ marginTop: '0.85rem', textAlign: 'center' }}>
                <h4 style={{ margin: 0, fontSize: '1.15rem', color: 'var(--text-main)' }}>AI Telephony Agent</h4>
                <p
                  style={{
                    margin: '0.35rem 0 0',
                    fontSize: '0.88rem',
                    fontWeight: 500,
                    color:
                      callStatus === 'ai_speaking'
                        ? '#fbbf24'
                        : callStatus === 'speaking'
                          ? '#38bdf8'
                          : callStatus === 'processing'
                            ? '#c084fc'
                            : isMuted
                              ? '#f87171'
                              : 'var(--text-muted)'
                  }}
                >
                  {callState === 'idle' && 'Click Start Voice Call to connect'}
                  {callState === 'calling' && 'Dialing agent...'}
                  {callState === 'ended' && 'Call completed'}
                  {callState === 'active' && (
                    <>
                      {isMuted && '🔇 Microphone Muted'}
                      {!isMuted && callStatus === 'ai_speaking' && '🔊 Speaking... (Talk to barge in)'}
                      {!isMuted && callStatus === 'speaking' && '🗣️ Listening to you speak...'}
                      {!isMuted && callStatus === 'processing' && '⏳ AI Thinking & Synthesizing...'}
                      {!isMuted && callStatus === 'listening' && '🎙️ Listening... (Speak naturally)'}
                    </>
                  )}
                </p>
              </div>

              {callState === 'active' && (
                <div style={{ marginTop: '0.75rem', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                  <canvas
                    ref={canvasRef}
                    width={220}
                    height={28}
                    style={{
                      display: 'block',
                      borderRadius: '6px',
                      background: 'var(--bg-subtle)'
                    }}
                  />
                </div>
              )}

              {/* Barge-In Notification Banner */}
              {bargeInNotice && (
                <div
                  style={{
                    marginTop: '0.75rem',
                    padding: '0.45rem 1rem',
                    borderRadius: '8px',
                    background: 'rgba(239, 68, 68, 0.12)',
                    border: '1px solid rgba(239, 68, 68, 0.35)',
                    color: '#ef4444',
                    fontSize: '0.82rem',
                    fontWeight: 600,
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.45rem'
                  }}
                >
                  <Zap size={15} />
                  <span>{bargeInNotice}</span>
                </div>
              )}

              {/* Automatic English Accent Variety Status Badge */}
              {callState === 'active' && (
                <div style={{ marginTop: '0.85rem', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.45rem' }}>
                  <div
                    style={{
                      padding: '0.45rem 1rem',
                      borderRadius: '9999px',
                      background: detectedAccent
                        ? (detectedAccent.status === 'detected' ? 'rgba(16, 185, 129, 0.14)' : 'rgba(245, 158, 11, 0.14)')
                        : 'rgba(56, 189, 248, 0.12)',
                      border: `1px solid ${detectedAccent
                        ? (detectedAccent.status === 'detected' ? 'rgba(16, 185, 129, 0.35)' : 'rgba(245, 158, 11, 0.35)')
                        : 'rgba(56, 189, 248, 0.3)'}`,
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: '0.55rem',
                      fontSize: '0.84rem',
                      fontWeight: 600,
                      color: detectedAccent
                        ? (detectedAccent.status === 'detected' ? '#059669' : '#d97706')
                        : '#0284c7',
                      boxShadow: 'var(--shadow-sm)',
                      transition: 'all 0.3s ease'
                    }}
                  >
                    <Globe size={15} />
                    <span>
                      {detectedAccent ? (
                        detectedAccent.status === 'detected' ? (
                          <>
                            <span style={{ marginRight: '0.3rem' }}>{detectedAccent.flag}</span>
                            Detected speech variety: <strong>{detectedAccent.accent}</strong>
                            <span style={{ fontSize: '0.74rem', opacity: 0.85, marginLeft: '0.35rem' }}>
                              ({Math.round(detectedAccent.confidence * 100)}% confidence • {detectedAccent.latencyMs}ms)
                            </span>
                          </>
                        ) : detectedAccent.status === 'insufficient_audio' ? (
                          'Detecting English accent... (speak a full phrase)'
                        ) : (
                          `Accent: ${detectedAccent.accent}`
                        )
                      ) : isDetectingAccent ? (
                        'Detecting English accent...'
                      ) : (
                        'Automatic English accent detection active'
                      )}
                    </span>
                  </div>

                  {/* Auto-adapt AI Voice Toggle */}
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem', fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                    <label style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', cursor: 'pointer', userSelect: 'none' }}>
                      <input
                        type="checkbox"
                        checked={autoAdaptVoice}
                        onChange={(e) => handleToggleAutoAdaptVoice(e.target.checked)}
                        style={{ cursor: 'pointer', accentColor: '#0284c7' }}
                      />
                      <span>Auto-adapt AI response voice to caller accent</span>
                    </label>
                    {selectedVoice && (
                      <span style={{ fontSize: '0.73rem', background: 'var(--bg-subtle)', border: '1px solid var(--border)', padding: '0.15rem 0.5rem', borderRadius: '4px' }}>
                        Voice: <strong>{selectedVoice}</strong>
                      </span>
                    )}
                  </div>
                </div>
              )}

              {/* Voice Adaptation Banner */}
              {voiceAdaptationNotice && (
                <div
                  style={{
                    marginTop: '0.55rem',
                    padding: '0.4rem 0.95rem',
                    borderRadius: '8px',
                    background: 'rgba(59, 130, 246, 0.12)',
                    border: '1px solid rgba(59, 130, 246, 0.35)',
                    color: '#2563eb',
                    fontSize: '0.8rem',
                    fontWeight: 600,
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.45rem'
                  }}
                >
                  <Sparkles size={14} />
                  <span>{voiceAdaptationNotice}</span>
                </div>
              )}

              {/* Live Full-Duplex Real-Time Metrics Panel */}
              {sttModel === 'sherpa-streaming' && callState === 'active' && (
                <div
                  style={{
                    width: '100%',
                    maxWidth: '640px',
                    margin: '0.85rem auto 0',
                    padding: '0.75rem 1rem',
                    borderRadius: '10px',
                    background: 'var(--bg-card-solid)',
                    border: '1px solid var(--border)',
                    boxShadow: 'var(--shadow-sm)'
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.55rem' }}>
                    <span style={{ fontSize: '0.76rem', fontWeight: 700, color: '#9333ea', display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                      <Zap size={13} /> Sherpa-ONNX Full-Duplex Observed Latency
                    </span>
                    <span style={{ fontSize: '0.68rem', color: 'var(--text-muted)' }}>
                      Zipformer INT8 • Trailing Silence VAD
                    </span>
                  </div>

                  <div
                    style={{
                      display: 'grid',
                      gridTemplateColumns: 'repeat(auto-fit, minmax(95px, 1fr))',
                      gap: '0.5rem',
                      textAlign: 'center'
                    }}
                  >
                    <div style={{ padding: '0.4rem 0.25rem', borderRadius: '7px', background: 'var(--bg-subtle)' }}>
                      <div style={{ fontSize: '0.67rem', color: 'var(--text-muted)' }}>1st Partial</div>
                      <div style={{ fontSize: '0.88rem', fontWeight: 700, color: '#38bdf8' }}>
                        {fdMetrics.timeToFirstPartial ? `${fdMetrics.timeToFirstPartial}ms` : '—'}
                      </div>
                    </div>

                    <div style={{ padding: '0.4rem 0.25rem', borderRadius: '7px', background: 'var(--bg-subtle)' }}>
                      <div style={{ fontSize: '0.67rem', color: 'var(--text-muted)' }}>Final STT</div>
                      <div style={{ fontSize: '0.88rem', fontWeight: 700, color: '#818cf8' }}>
                        {fdMetrics.timeToFinal ? `${fdMetrics.timeToFinal}ms` : '—'}
                      </div>
                    </div>

                    <div style={{ padding: '0.4rem 0.25rem', borderRadius: '7px', background: 'var(--bg-subtle)' }}>
                      <div style={{ fontSize: '0.67rem', color: 'var(--text-muted)' }}>1st Token (LLM)</div>
                      <div style={{ fontSize: '0.88rem', fontWeight: 700, color: '#a855f7' }}>
                        {fdMetrics.timeToFirstToken ? `${fdMetrics.timeToFirstToken}ms` : '—'}
                      </div>
                    </div>

                    <div style={{ padding: '0.4rem 0.25rem', borderRadius: '7px', background: 'var(--bg-subtle)' }}>
                      <div style={{ fontSize: '0.67rem', color: 'var(--text-muted)' }}>1st Audio (TTS)</div>
                      <div style={{ fontSize: '0.88rem', fontWeight: 700, color: '#f59e0b' }}>
                        {fdMetrics.timeToFirstAudio ? `${fdMetrics.timeToFirstAudio}ms` : '—'}
                      </div>
                    </div>

                    <div style={{ padding: '0.4rem 0.25rem', borderRadius: '7px', background: 'var(--bg-subtle)' }}>
                      <div style={{ fontSize: '0.67rem', color: 'var(--text-muted)' }}>Turn Total</div>
                      <div style={{ fontSize: '0.88rem', fontWeight: 700, color: '#10b981' }}>
                        {fdMetrics.totalTurnLatency ? `${fdMetrics.totalTurnLatency}ms` : '—'}
                      </div>
                    </div>

                    <div style={{ padding: '0.4rem 0.25rem', borderRadius: '7px', background: 'var(--bg-subtle)' }}>
                      <div style={{ fontSize: '0.67rem', color: 'var(--text-muted)' }}>Interruption</div>
                      <div style={{ fontSize: '0.88rem', fontWeight: 700, color: '#ef4444' }}>
                        {fdMetrics.interruptionLatency !== null ? `${fdMetrics.interruptionLatency}ms` : '0ms'}
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>


            {/* Call Controls */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '1.25rem', marginTop: '0.5rem' }}>
              {callState === 'idle' ? (
                <button
                  onClick={handleStartCall}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.65rem',
                    padding: '0.9rem 2rem',
                    borderRadius: '999px',
                    background: '#10b981',
                    color: '#ffffff',
                    border: 'none',
                    fontWeight: 600,
                    fontSize: '1rem',
                    cursor: 'pointer',
                    boxShadow: '0 0 25px rgba(16, 185, 129, 0.45)',
                    transition: 'all 0.2s ease'
                  }}
                >
                  <Phone size={20} />
                  <span>Start Voice Call</span>
                </button>
              ) : (
                <>
                  <button
                    onClick={handleToggleMute}
                    title={isMuted ? 'Unmute Microphone' : 'Mute Microphone'}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '0.5rem',
                      padding: '0.8rem 1.4rem',
                      borderRadius: '999px',
                      background: isMuted ? 'rgba(239, 68, 68, 0.15)' : 'var(--bg-subtle)',
                      color: isMuted ? '#ef4444' : 'var(--text-main)',
                      border: isMuted ? '1px solid #ef4444' : '1px solid var(--border)',
                      fontWeight: 600,
                      fontSize: '0.92rem',
                      cursor: 'pointer',
                      transition: 'all 0.2s ease'
                    }}
                  >
                    {isMuted ? <MicOff size={18} color="#ef4444" /> : <Mic size={18} />}
                    <span>{isMuted ? 'Unmute' : 'Mute'}</span>
                  </button>

                  <button
                    onClick={handleEndCall}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '0.5rem',
                      padding: '0.8rem 1.6rem',
                      borderRadius: '999px',
                      background: '#ef4444',
                      color: '#ffffff',
                      border: 'none',
                      fontWeight: 600,
                      fontSize: '0.95rem',
                      cursor: 'pointer',
                      boxShadow: '0 0 22px rgba(239, 68, 68, 0.45)',
                      transition: 'all 0.2s ease'
                    }}
                  >
                    <PhoneOff size={18} />
                    <span>Hang Up</span>
                  </button>
                </>
              )}
            </div>



            {/* Settings Toggle Bar */}
            <div style={{ marginTop: '1.25rem', display: 'flex', alignItems: 'center', gap: '1rem' }}>
              <button
                onClick={() => setShowCallSettings((v) => !v)}
                style={{
                  background: 'transparent',
                  border: 'none',
                  color: 'var(--text-muted)',
                  fontSize: '0.78rem',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.35rem'
                }}
              >
                <Sliders size={14} />
                <span>{showCallSettings ? 'Hide Detailed Call Options' : 'Detailed Call & Model Settings'}</span>
              </button>
            </div>

            {showCallSettings && (
              <div
                style={{
                  marginTop: '0.75rem',
                  padding: '1rem 1.2rem',
                  borderRadius: '12px',
                  background: 'var(--bg-card-solid)',
                  border: '1px solid var(--border)',
                  boxShadow: 'var(--shadow-md)',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '1.1rem',
                  fontSize: '0.82rem',
                  width: '100%',
                  maxWidth: '620px'
                }}
              >
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '1rem' }}>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.45rem' }}>
                    <label style={{ color: 'var(--accent-call)', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                      <Mic size={14} /> Voice-to-Text (STT) Engine & Model:
                    </label>

                    {/* Dual Engine Selector Buttons */}
                    <div style={{ display: 'flex', gap: '0.3rem', background: 'var(--bg-input)', padding: '0.2rem', borderRadius: '6px', border: '1px solid var(--border)' }}>
                      <button
                        type="button"
                        onClick={() => handleSttEngineChange('sherpa')}
                        style={{
                          flex: 1,
                          padding: '0.3rem',
                          borderRadius: '4px',
                          border: 'none',
                          background: sttEngine === 'sherpa' ? 'rgba(168, 85, 247, 0.2)' : 'transparent',
                          color: sttEngine === 'sherpa' ? '#9333ea' : 'var(--text-muted)',
                          fontWeight: sttEngine === 'sherpa' ? 700 : 500,
                          fontSize: '0.72rem',
                          cursor: 'pointer'
                        }}
                      >
                        ⚡ Sherpa-ONNX
                      </button>
                      <button
                        type="button"
                        onClick={() => handleSttEngineChange('whisper')}
                        style={{
                          flex: 1,
                          padding: '0.3rem',
                          borderRadius: '4px',
                          border: 'none',
                          background: sttEngine === 'whisper' ? 'rgba(2, 132, 199, 0.2)' : 'transparent',
                          color: sttEngine === 'whisper' ? '#0284c7' : 'var(--text-muted)',
                          fontWeight: sttEngine === 'whisper' ? 700 : 500,
                          fontSize: '0.72rem',
                          cursor: 'pointer'
                        }}
                      >
                        🎙️ Faster-Whisper
                      </button>
                    </div>

                    <select
                      value={sttModel}
                      onChange={(e) => handleSttModelChange(e.target.value)}
                      style={{
                        padding: '0.5rem 0.65rem',
                        borderRadius: '6px',
                        background: 'var(--bg-input)',
                        border: '1px solid var(--border)',
                        color: 'var(--text-main)',
                        fontSize: '0.82rem'
                      }}
                    >
                      {sttEngine === 'sherpa' ? (
                        SHERPA_MODELS.map((m) => (
                          <option key={m.id} value={m.id}>
                            {m.type === 'streaming' ? '⚡' : '📦'} {m.label} • {m.status || 'Ready'}
                          </option>
                        ))
                      ) : (
                        WHISPER_MODELS.map((m) => (
                          <option key={m.id} value={m.id}>
                            🎙️ {m.label} • {m.status || 'Ready'}
                          </option>
                        ))
                      )}
                    </select>

                    {onOpenSherpa && (
                      <button
                        type="button"
                        onClick={onOpenSherpa}
                        style={{
                          marginTop: '0.2rem',
                          padding: '0.35rem 0.6rem',
                          borderRadius: '6px',
                          background: 'rgba(168, 85, 247, 0.12)',
                          border: '1px solid rgba(168, 85, 247, 0.35)',
                          color: '#9333ea',
                          fontSize: '0.74rem',
                          fontWeight: 600,
                          cursor: 'pointer',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'space-between',
                          gap: '0.4rem',
                          transition: 'all 0.2s ease'
                        }}
                        title="Open STT model catalog & empirical benchmark report"
                      >
                        <span>⚙️ Manage & Benchmark All Models</span>
                        <span style={{ textDecoration: 'underline', fontSize: '0.7rem' }}>Catalog</span>
                      </button>
                    )}
                  </div>

                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
                    <label style={{ color: 'var(--text-muted)', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                      <Globe size={14} /> Caller Audio Language:
                    </label>
                    <select
                      value={sttLanguage}
                      onChange={(e) => handleSttLanguageChange(e.target.value)}
                      style={{
                        padding: '0.5rem 0.65rem',
                        borderRadius: '6px',
                        background: 'var(--bg-input)',
                        border: '1px solid var(--border)',
                        color: 'var(--text-main)',
                        fontSize: '0.82rem'
                      }}
                    >
                      {STT_LANGUAGES.map((l) => (
                        <option key={l.code} value={l.code}>
                          {l.label}
                        </option>
                      ))}
                    </select>
                    {sttModel === 'sherpa-streaming' && sttLanguage !== 'en' && sttLanguage !== 'auto' && (
                      <span style={{ fontSize: '0.72rem', color: '#f59e0b', marginTop: '0.15rem' }}>
                        ⚠️ Sherpa is English-only. For Indic/Hindi speech, select Whisper Base.
                      </span>
                    )}
                  </div>

                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
                    <label style={{ color: 'var(--accent-tts)', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                      <Volume2 size={14} /> Text-to-Voice (TTS) Model:
                    </label>
                    <select
                      value={ttsModel}
                      onChange={(e) => handleTtsModelChange(e.target.value)}
                      style={{
                        padding: '0.5rem 0.65rem',
                        borderRadius: '6px',
                        background: 'var(--bg-input)',
                        border: '1px solid var(--border)',
                        color: 'var(--text-main)',
                        fontSize: '0.82rem'
                      }}
                    >
                      {TTS_MODELS.map((tm) => (
                        <option key={tm.id} value={tm.id}>
                          {tm.name}
                        </option>
                      ))}
                    </select>
                  </div>

                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
                    <label style={{ color: 'var(--text-muted)', fontWeight: 600 }}>Agent Voice Persona:</label>
                    <select
                      value={selectedVoice}
                      onChange={(e) => handleVoiceChange(e.target.value)}
                      style={{
                        padding: '0.5rem 0.65rem',
                        borderRadius: '6px',
                        background: 'var(--bg-input)',
                        border: '1px solid var(--border)',
                        color: 'var(--text-main)',
                        fontSize: '0.82rem'
                      }}
                    >
                      {filteredVoices.map((v) => (
                        <option key={v.id} value={v.id}>
                          {v.name} [{v.accent}] ({v.engine})
                        </option>
                      ))}
                    </select>
                  </div>
                </div>

                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.35rem', paddingTop: '0.5rem', borderTop: '1px solid var(--border)' }}>
                  <label style={{ color: 'var(--text-muted)', fontWeight: 600 }}>Silence Detection Pause (VAD Trigger):</label>
                  <select
                    value={silenceTimeoutMs}
                    onChange={(e) => handleSilenceChange(Number(e.target.value))}
                    style={{
                      padding: '0.45rem 0.6rem',
                      borderRadius: '6px',
                      background: 'var(--bg-input)',
                      border: '1px solid var(--border)',
                      color: 'var(--text-main)',
                      fontSize: '0.82rem'
                    }}
                  >
                    {SILENCE_OPTIONS.map((opt) => (
                      <option key={opt.value} value={opt.value}>
                        {opt.label}
                      </option>
                    ))}
                  </select>
                </div>
              </div>
            )}
          </section>

          {/* Live Conversation Transcript Log */}
          {(chatHistory.length > 0 || livePartial || liveAssistantText) && (
            <section
              className="glass-panel"
              style={{
                display: 'flex',
                flexDirection: 'column',
                gap: '0.75rem',
                maxHeight: '280px',
                overflowY: 'auto'
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-muted)' }}>
                  Call Transcript Log
                </span>
                <button
                  onClick={() => setChatHistory([])}
                  style={{
                    background: 'transparent',
                    border: 'none',
                    color: 'var(--text-muted)',
                    cursor: 'pointer',
                    fontSize: '0.75rem',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.25rem'
                  }}
                >
                  <RotateCcw size={12} /> Clear Log
                </button>
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem' }}>
                {chatHistory.map((msg, idx) => {
                  if (!msg || typeof msg.content !== 'string' || !msg.content.trim()) {
                    return null;
                  }

                  const stableKey = msg.id || `msg_${idx}`;

                  if (msg.role === 'system_notification') {
                    return (
                      <div
                        key={stableKey}
                        style={{
                          textAlign: 'center',
                          fontSize: '0.75rem',
                          color: 'var(--text-muted)',
                          fontStyle: 'italic',
                          padding: '0.2rem 0'
                        }}
                      >
                        {msg.content}
                      </div>
                    );
                  }

                  const isUser = msg.role === 'user';
                  return (
                    <div
                      key={stableKey}
                      style={{
                        display: 'flex',
                        alignSelf: isUser ? 'flex-end' : 'flex-start',
                        maxWidth: '82%',
                        gap: '0.5rem',
                        flexDirection: isUser ? 'row-reverse' : 'row'
                      }}
                    >
                      <div
                        style={{
                          width: '26px',
                          height: '26px',
                          borderRadius: '50%',
                          background: isUser ? 'rgba(56, 189, 248, 0.2)' : 'rgba(245, 158, 11, 0.2)',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          flexShrink: 0
                        }}
                      >
                        {isUser ? <User size={14} color="#38bdf8" /> : <Bot size={14} color="#f59e0b" />}
                      </div>
                      <div
                        style={{
                          padding: '0.55rem 0.85rem',
                          borderRadius: isUser ? '14px 14px 2px 14px' : '14px 14px 14px 2px',
                          background: isUser
                            ? (msg.isPartial ? 'rgba(2, 132, 199, 0.08)' : 'var(--accent-call-bg)')
                            : 'var(--bg-subtle)',
                          border: isUser
                            ? (msg.isPartial ? '1px dashed #0284c7' : '1px solid var(--accent-call-border)')
                            : '1px solid var(--border)',
                          color: 'var(--text-main)',
                          fontSize: '0.88rem',
                          lineHeight: '1.4'
                        }}
                      >
                        {msg.isPartial && isUser && (
                          <span
                            style={{
                              width: '8px',
                              height: '8px',
                              borderRadius: '50%',
                              background: '#0284c7',
                              display: 'inline-block',
                              boxShadow: '0 0 6px #0284c7',
                              marginRight: '6px'
                            }}
                          />
                        )}
                        <span>{msg.content}</span>
                        {msg.isPartial && !isUser && (
                          <span
                            style={{
                              display: 'inline-block',
                              width: '2px',
                              height: '13px',
                              background: '#f59e0b',
                              marginLeft: '4px',
                              verticalAlign: 'middle'
                            }}
                          />
                        )}
                      </div>
                    </div>
                  );
                })}

                <div ref={transcriptBottomRef} />
              </div>
            </section>
          )}

          {/* Quick Standalone TTS Tester Bar */}
          <section
            className="glass-panel"
            style={{
              display: 'flex',
              flexDirection: 'column',
              gap: '0.85rem'
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <h4
                style={{
                  margin: 0,
                  fontSize: '0.9rem',
                  color: 'var(--text-main)',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.5rem'
                }}
              >
                <Volume2 size={16} color="#f59e0b" />
                <span>Quick Text-to-Voice Tester</span>
              </h4>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                TTS Model: <strong style={{ color: '#fbbf24' }}>{activeTtsEngine}</strong> • Voice:{' '}
                <strong style={{ color: '#38bdf8' }}>{selectedVoiceObj?.name?.split(' ')[0] || selectedVoice}</strong>
              </span>
            </div>

            <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap' }}>
              <input
                type="text"
                value={testText}
                onChange={(e) => setTestText(e.target.value)}
                placeholder="Type any sentence to hear text-to-speech..."
                style={{
                  flex: 1,
                  minWidth: '240px',
                  padding: '0.55rem 0.85rem',
                  borderRadius: '8px',
                  background: 'var(--bg-input)',
                  border: '1px solid var(--border)',
                  color: 'var(--text-main)',
                  fontSize: '0.85rem'
                }}
              />

              <button
                onClick={handleTestTTS}
                disabled={isTestingTTS || !testText.trim()}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.4rem',
                  padding: '0.55rem 1.1rem',
                  borderRadius: '8px',
                  background: 'var(--accent-call)',
                  color: '#ffffff',
                  border: 'none',
                  fontWeight: 600,
                  fontSize: '0.85rem',
                  cursor: isTestingTTS ? 'not-allowed' : 'pointer',
                  transition: 'all 0.2s ease'
                }}
              >
                <Play size={15} />
                <span>{isTestingTTS ? 'Synthesizing...' : 'Speak Text'}</span>
              </button>
            </div>
          </section>
        </>
      )}

      {/* ======================================================== */}
      {/* MODE 2: STANDALONE VOICE TO TEXT (STT TRANSCRIBER)        */}
      {/* ======================================================== */}
      {simulatorMode === 'stt' && (
        <section
          className="glass-panel"
          style={{
            display: 'flex',
            flexDirection: 'column',
            gap: '1.5rem',
            padding: '2rem 1.5rem',
            background: 'radial-gradient(circle at 50% 10%, rgba(5, 150, 105, 0.08), var(--bg-card-solid))'
          }}
        >
          {/* STT Header */}
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.75rem' }}>
            <div>
              <h3 style={{ margin: 0, fontSize: '1.25rem', color: 'var(--text-main)', display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                <Mic size={22} color="var(--accent-stt)" />
                <span>Voice to Text (STT Transcriber)</span>
              </h3>
              <p style={{ margin: '0.3rem 0 0', fontSize: '0.85rem', color: 'var(--text-muted)' }}>
                Directly transcribe spoken voice into text using {sttModel === 'sherpa-streaming' ? 'Sherpa-ONNX Zipformer' : `Faster-Whisper (${sttModel})`}
              </p>
            </div>

            <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
              <span className="badge badge-green">Model: {sttModel}</span>
              <span className="badge badge-blue">
                Lang: English-only (en)
              </span>
            </div>
          </div>

          {/* Central Recording Console */}
          <div
            style={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              padding: '2rem 1rem',
              borderRadius: '16px',
              background: 'var(--bg-subtle)',
              border: sttIsRecording ? '1px solid rgba(239, 68, 68, 0.5)' : '1px solid var(--accent-stt-border)',
              boxShadow: sttIsRecording ? '0 0 35px rgba(239, 68, 68, 0.25)' : 'none',
              transition: 'all 0.3s ease'
            }}
          >
            {/* Record / Stop Button */}
            <div style={{ position: 'relative', marginBottom: '1.25rem' }}>
              {sttIsRecording && (
                <div
                  style={{
                    position: 'absolute',
                    inset: '-10px',
                    borderRadius: '50%',
                    border: '2px solid rgba(239, 68, 68, 0.6)',
                    animation: 'pulse 1.5s infinite',
                    pointerEvents: 'none'
                  }}
                />
              )}

              <button
                onClick={sttIsRecording ? handleStopSttRecording : handleStartSttRecording}
                disabled={sttIsProcessing}
                style={{
                  width: '90px',
                  height: '90px',
                  borderRadius: '50%',
                  background: sttIsRecording
                    ? 'linear-gradient(135deg, #ef4444, #b91c1c)'
                    : 'linear-gradient(135deg, #059669, #0284c7)',
                  color: '#ffffff',
                  border: 'none',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  cursor: sttIsProcessing ? 'not-allowed' : 'pointer',
                  boxShadow: sttIsRecording
                    ? '0 0 30px rgba(239, 68, 68, 0.6)'
                    : '0 0 30px var(--accent-stt-glow)',
                  transition: 'all 0.3s ease',
                  transform: sttIsRecording ? 'scale(1.08)' : 'scale(1)'
                }}
              >
                {sttIsRecording ? <Square size={36} /> : <Mic size={40} />}
              </button>
            </div>

            {/* Recording Status & Timer */}
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: '1.15rem', fontWeight: 700, color: sttIsRecording ? '#ef4444' : 'var(--text-main)' }}>
                {sttIsRecording ? formatCallTime(sttRecordDuration) : sttIsProcessing ? 'Transcribing...' : 'Ready to Record'}
              </div>
              <p style={{ margin: '0.35rem 0 0', fontSize: '0.85rem', color: 'var(--text-muted)' }}>
                {sttIsRecording
                  ? 'Speaking into microphone... Click square button when finished'
                  : sttIsProcessing
                    ? 'Whisper is converting your audio to text...'
                    : 'Click the microphone button to start recording speech'}
              </p>
            </div>

            {/* Waveform Canvas */}
            <div style={{ marginTop: '1rem', width: '100%', maxWidth: '380px' }}>
              <canvas
                ref={sttCanvasRef}
                width={380}
                height={34}
                style={{
                  display: 'block',
                  width: '100%',
                  height: '34px',
                  borderRadius: '8px',
                  background: 'var(--bg-card)'
                }}
              />
            </div>

            {/* File Upload Option */}
            <div style={{ marginTop: '1.25rem', display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
              <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>— OR upload an audio file —</span>
              <label
                style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: '0.4rem',
                  padding: '0.35rem 0.8rem',
                  borderRadius: '6px',
                  background: 'var(--bg-card)',
                  border: '1px solid var(--border)',
                  color: 'var(--text-main)',
                  fontSize: '0.78rem',
                  cursor: 'pointer',
                  boxShadow: 'var(--shadow-xs)',
                  transition: 'all 0.2s ease'
                }}
              >
                <UploadCloud size={14} />
                <span>Upload Audio File</span>
                <input
                  type="file"
                  accept="audio/*,.wav,.mp3,.webm,.ogg,.m4a"
                  onChange={handleSttFileUpload}
                  style={{ display: 'none' }}
                />
              </label>
            </div>

            {sttError && (
              <div style={{ marginTop: '0.85rem', color: '#f87171', fontSize: '0.82rem' }}>
                ⚠️ Error: {sttError}
              </div>
            )}
          </div>

          {/* Transcript Results Box */}
          {sttTranscriptResult && (
            <div
              style={{
                display: 'flex',
                flexDirection: 'column',
                gap: '1rem',
                padding: '1.35rem',
                borderRadius: '14px',
                background: 'var(--bg-card-solid)',
                border: '1px solid var(--accent-stt-border)',
                boxShadow: 'var(--shadow-md)'
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.5rem' }}>
                <span style={{ fontSize: '0.9rem', fontWeight: 700, color: 'var(--accent-stt)', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                  <FileText size={16} /> Recognized Voice Transcript
                </span>
                <div style={{ display: 'flex', gap: '0.5rem' }}>
                  <button
                    onClick={handleCopySttTranscript}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '0.35rem',
                      padding: '0.35rem 0.75rem',
                      borderRadius: '6px',
                      background: sttCopied ? 'rgba(16, 185, 129, 0.2)' : 'rgba(255, 255, 255, 0.08)',
                      border: sttCopied ? '1px solid #10b981' : '1px solid rgba(255, 255, 255, 0.15)',
                      color: sttCopied ? '#34d399' : '#e2e8f0',
                      fontSize: '0.78rem',
                      cursor: 'pointer',
                      transition: 'all 0.2s ease'
                    }}
                  >
                    {sttCopied ? <Check size={13} color="#34d399" /> : <Copy size={13} />}
                    <span>{sttCopied ? 'Copied!' : 'Copy Transcript'}</span>
                  </button>
                  <button
                    onClick={() => setSttTranscriptResult(null)}
                    style={{
                      background: 'transparent',
                      border: 'none',
                      color: 'var(--text-muted)',
                      fontSize: '0.78rem',
                      cursor: 'pointer'
                    }}
                  >
                    Clear
                  </button>
                </div>
              </div>

              {/* Transcript Text */}
              <div
                style={{
                  padding: '1rem',
                  borderRadius: '10px',
                  background: 'rgba(0, 0, 0, 0.35)',
                  border: '1px solid var(--border)',
                  color: '#f8fafc',
                  fontSize: '1rem',
                  lineHeight: '1.6',
                  minHeight: '60px'
                }}
              >
                {sttTranscriptResult.transcript || '(Empty speech or silence detected)'}
              </div>

              {/* Latency & Metrics Cards */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: '0.65rem' }}>
                <div style={{ padding: '0.65rem 0.85rem', borderRadius: '8px', background: 'rgba(30, 41, 59, 0.5)', border: '1px solid rgba(255, 255, 255, 0.08)' }}>
                  <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>STT Latency</div>
                  <div style={{ fontSize: '1rem', fontWeight: 700, color: '#38bdf8', fontFamily: 'var(--font-mono)', marginTop: '0.15rem' }}>
                    {sttTranscriptResult.stt_latency}s
                  </div>
                </div>

                <div style={{ padding: '0.65rem 0.85rem', borderRadius: '8px', background: 'var(--bg-subtle)', border: '1px solid var(--border)' }}>
                  <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Audio Length</div>
                  <div style={{ fontSize: '1rem', fontWeight: 700, color: 'var(--text-main)', fontFamily: 'var(--font-mono)', marginTop: '0.15rem' }}>
                    {sttTranscriptResult.duration}s
                  </div>
                </div>

                <div style={{ padding: '0.65rem 0.85rem', borderRadius: '8px', background: 'var(--bg-subtle)', border: '1px solid var(--border)' }}>
                  <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>RTF (Real-time Factor)</div>
                  <div style={{ fontSize: '1rem', fontWeight: 700, color: 'var(--accent-stt)', fontFamily: 'var(--font-mono)', marginTop: '0.15rem' }}>
                    {sttTranscriptResult.rtf}
                  </div>
                </div>

                <div style={{ padding: '0.65rem 0.85rem', borderRadius: '8px', background: 'var(--bg-subtle)', border: '1px solid var(--border)' }}>
                  <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Speed Factor</div>
                  <div style={{ fontSize: '1rem', fontWeight: 700, color: 'var(--accent-tts)', fontFamily: 'var(--font-mono)', marginTop: '0.15rem' }}>
                    {sttTranscriptResult.speed_factor}x
                  </div>
                </div>

                <div style={{ padding: '0.65rem 0.85rem', borderRadius: '8px', background: 'var(--bg-subtle)', border: '1px solid var(--border)' }}>
                  <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Language</div>
                  <div style={{ fontSize: '0.95rem', fontWeight: 700, color: 'var(--accent-call)', marginTop: '0.15rem' }}>
                    {sttTranscriptResult.language}
                  </div>
                </div>
              </div>

              {/* Automatic English Accent Variety Analysis Card */}
              {sttTranscriptResult.accent_analysis && (
                <div
                  style={{
                    padding: '0.85rem 1.15rem',
                    borderRadius: '10px',
                    background: sttTranscriptResult.accent_analysis.status === 'detected' ? 'rgba(16, 185, 129, 0.08)' : 'rgba(245, 158, 11, 0.08)',
                    border: `1px solid ${sttTranscriptResult.accent_analysis.status === 'detected' ? 'rgba(16, 185, 129, 0.25)' : 'rgba(245, 158, 11, 0.25)'}`,
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    flexWrap: 'wrap',
                    gap: '0.75rem'
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                    <span style={{ fontSize: '1.5rem' }}>{sttTranscriptResult.accent_analysis.flag || '🌐'}</span>
                    <div>
                      <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', fontWeight: 600 }}>
                        DETECTED ENGLISH SPEECH VARIETY (ACOUSTIC)
                      </div>
                      <div style={{ fontSize: '1.05rem', fontWeight: 700, color: sttTranscriptResult.accent_analysis.status === 'detected' ? '#10b981' : '#f59e0b' }}>
                        {sttTranscriptResult.accent_analysis.accent}
                      </div>
                    </div>
                  </div>

                  <div style={{ display: 'flex', gap: '1.15rem', alignItems: 'center', fontSize: '0.82rem', flexWrap: 'wrap' }}>
                    <div>
                      <span style={{ color: 'var(--text-muted)' }}>Confidence: </span>
                      <strong style={{ color: sttTranscriptResult.accent_analysis.status === 'detected' ? '#10b981' : '#f59e0b' }}>
                        {Math.round(sttTranscriptResult.accent_analysis.confidence * 100)}%
                      </strong>
                    </div>
                    <div>
                      <span style={{ color: 'var(--text-muted)' }}>Accent Latency: </span>
                      <strong style={{ fontFamily: 'var(--font-mono)', color: '#38bdf8' }}>
                        {sttTranscriptResult.accent_analysis.latency_ms}ms
                      </strong>
                    </div>
                    <div>
                      <span style={{ color: 'var(--text-muted)' }}>Matched Edge-TTS Voice: </span>
                      <strong style={{ background: 'var(--bg-subtle)', padding: '0.15rem 0.45rem', borderRadius: '4px', fontSize: '0.76rem' }}>
                        {sttTranscriptResult.accent_analysis.tts_voice || 'en-US-JennyNeural'}
                      </strong>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </section>
      )}

      {/* ======================================================== */}
      {/* MODE 3: STANDALONE TEXT TO VOICE (TTS SPEECH GENERATOR)   */}
      {/* ======================================================== */}
      {simulatorMode === 'tts' && (
        <section
          className="glass-panel"
          style={{
            display: 'flex',
            flexDirection: 'column',
            gap: '1.5rem',
            padding: '2rem 1.5rem',
            background: 'radial-gradient(circle at 50% 10%, rgba(124, 58, 237, 0.08), var(--bg-card-solid))'
          }}
        >
          {/* TTS Header */}
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.75rem' }}>
            <div>
              <h3 style={{ margin: 0, fontSize: '1.25rem', color: 'var(--text-main)', display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                <Volume2 size={22} color="var(--accent-tts)" />
                <span>Text to Voice (TTS Speech Generator)</span>
              </h3>
              <p style={{ margin: '0.3rem 0 0', fontSize: '0.85rem', color: 'var(--text-muted)' }}>
                Convert written text into natural spoken audio using {activeTtsEngine}
              </p>
            </div>

            <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
              <span className="badge badge-purple">Model: {activeTtsEngine}</span>
              <span className="badge badge-blue">Voice: {selectedVoiceObj?.name?.split(' ')[0] || selectedVoice}</span>
            </div>
          </div>

          {/* Quick Telephony Presets */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
            <span style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
              Telephony Prompt Presets (1-Click Fill)
            </span>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.45rem' }}>
              {TTS_PRESETS.map((p, idx) => (
                <button
                  key={idx}
                  onClick={() => setTtsInputText(p.text)}
                  style={{
                    padding: '0.35rem 0.75rem',
                    borderRadius: '8px',
                    background: ttsInputText === p.text ? 'var(--accent-tts-bg)' : 'var(--bg-subtle)',
                    border: ttsInputText === p.text ? '1px solid var(--accent-tts)' : '1px solid var(--border)',
                    color: ttsInputText === p.text ? 'var(--accent-tts)' : 'var(--text-main)',
                    fontSize: '0.78rem',
                    cursor: 'pointer',
                    transition: 'all 0.2s ease'
                  }}
                >
                  {p.label}
                </button>
              ))}
            </div>
          </div>

          {/* Text Input Area */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.45rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <label style={{ fontSize: '0.82rem', fontWeight: 600, color: 'var(--text-muted)' }}>
                Text to Synthesize:
              </label>
              <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                {ttsInputText.length} characters
              </span>
            </div>

            <textarea
              rows={4}
              value={ttsInputText}
              onChange={(e) => setTtsInputText(e.target.value)}
              placeholder="Type any sentence or message to generate speech audio..."
              style={{
                width: '100%',
                padding: '0.85rem 1rem',
                borderRadius: '10px',
                background: 'var(--bg-input)',
                border: '1px solid var(--border)',
                color: 'var(--text-main)',
                fontSize: '0.92rem',
                lineHeight: '1.5',
                resize: 'vertical',
                outline: 'none',
                fontFamily: 'inherit'
              }}
            />
          </div>

          {/* Pitch & Speed & Action Controls */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.9rem', padding: '1rem', borderRadius: '10px', background: 'var(--bg-subtle)', border: '1px solid var(--border)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
              {/* Pitch Modulation Selector */}
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem' }}>
                <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                  <Sliders size={14} /> Pitch:
                </span>
                <select
                  value={selectedPitch}
                  onChange={(e) => handlePitchChange(e.target.value)}
                  style={{
                    padding: '0.35rem 0.55rem',
                    borderRadius: '6px',
                    background: 'var(--bg-input)',
                    border: '1px solid var(--border)',
                    color: 'var(--text-main)',
                    fontSize: '0.78rem',
                    cursor: 'pointer'
                  }}
                >
                  {PITCH_OPTIONS.map((po) => (
                    <option key={po.value} value={po.value}>
                      {po.label}
                    </option>
                  ))}
                </select>
              </div>

              {/* Speed Slider */}
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', minWidth: '220px' }}>
                <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                  <Clock size={14} /> Speed:
                </span>
                <input
                  type="range"
                  min="0.8"
                  max="1.5"
                  step="0.05"
                  value={ttsSpeed}
                  onChange={(e) => handleSpeedChange(parseFloat(e.target.value))}
                  style={{ flex: 1, cursor: 'pointer' }}
                />
                <span style={{ fontSize: '0.82rem', fontWeight: 700, color: '#c084fc', fontFamily: 'var(--font-mono)', minWidth: '40px' }}>
                  {ttsSpeed.toFixed(2)}x
                </span>
                <button
                  type="button"
                  onClick={() => handleSpeedChange(1.0)}
                  title="Reset to 1.0x"
                  style={{
                    background: 'transparent',
                    border: 'none',
                    color: 'var(--text-muted)',
                    fontSize: '0.7rem',
                    cursor: 'pointer'
                  }}
                >
                  Reset
                </button>
              </div>
            </div>

            {/* Synthesize Button */}
            <div style={{ display: 'flex', justifyContent: 'flex-end', alignItems: 'center', gap: '0.75rem', borderTop: '1px dashed var(--border)', paddingTop: '0.75rem' }}>
              <button
                type="button"
                onClick={() => handlePlayVoiceSample(selectedVoiceObj)}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.4rem',
                  padding: '0.55rem 1rem',
                  borderRadius: '8px',
                  background: previewPlayingVoiceId === selectedVoiceObj?.id ? '#ef4444' : 'var(--bg-card)',
                  color: previewPlayingVoiceId === selectedVoiceObj?.id ? '#ffffff' : 'var(--text-main)',
                  border: '1px solid var(--border)',
                  fontSize: '0.82rem',
                  fontWeight: 600,
                  cursor: 'pointer'
                }}
              >
                {previewPlayingVoiceId === selectedVoiceObj?.id ? (
                  <>
                    <Square size={14} />
                    <span>Stop Sample</span>
                  </>
                ) : (
                  <>
                    <Play size={14} />
                    <span>Play Sample ({selectedVoiceObj?.name})</span>
                  </>
                )}
              </button>

              <button
                onClick={handleGenerateTtsSpeech}
                disabled={ttsIsGenerating || !ttsInputText.trim()}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.6rem',
                  padding: '0.8rem 2rem',
                  borderRadius: '999px',
                  background: 'linear-gradient(135deg, #a855f7, #7c3aed)',
                  color: '#ffffff',
                  border: 'none',
                  fontWeight: 700,
                  fontSize: '0.95rem',
                  cursor: ttsIsGenerating ? 'not-allowed' : 'pointer',
                  boxShadow: '0 8px 25px rgba(168, 85, 247, 0.45)',
                  transition: 'all 0.2s ease'
                }}
              >
                <Play size={18} />
                <span>{ttsIsGenerating ? 'Synthesizing Speech...' : 'Generate Spoken Voice'}</span>
              </button>
            </div>
          </div>

          {ttsError && (
            <div style={{ color: '#f87171', fontSize: '0.82rem' }}>
              ⚠️ Synthesis Error: {ttsError}
            </div>
          )}

          {/* Generated Audio Player & Metrics */}
          {ttsResult && (
            <div
              style={{
                display: 'flex',
                flexDirection: 'column',
                gap: '1.1rem',
                padding: '1.35rem',
                borderRadius: '14px',
                background: 'var(--bg-card-solid)',
                border: '1px solid var(--accent-tts-border)',
                boxShadow: 'var(--shadow-md)'
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.5rem' }}>
                <span style={{ fontSize: '0.9rem', fontWeight: 700, color: 'var(--accent-tts)', display: 'flex', alignItems: 'center', gap: '0.45rem' }}>
                  <Volume2 size={16} /> Synthesized Audio Player
                </span>

                {ttsAudioUrl && (
                  <a
                    href={ttsAudioUrl}
                    download={`speech_${selectedVoice}_${Date.now()}.${ttsResult.format || 'wav'}`}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '0.4rem',
                      padding: '0.35rem 0.8rem',
                      borderRadius: '6px',
                      background: 'var(--bg-subtle)',
                      border: '1px solid var(--border)',
                      color: 'var(--text-main)',
                      fontSize: '0.78rem',
                      textDecoration: 'none',
                      cursor: 'pointer'
                    }}
                  >
                    <Download size={14} />
                    <span>Download Audio ({ttsResult.format?.toUpperCase() || 'WAV'})</span>
                  </a>
                )}
              </div>

              {/* Native Audio Controls Element */}
              {ttsAudioUrl && (
                <div style={{ width: '100%', marginTop: '0.25rem' }}>
                  <audio
                    controls
                    src={ttsAudioUrl}
                    style={{
                      width: '100%',
                      borderRadius: '8px',
                      outline: 'none'
                    }}
                  />
                </div>
              )}

              {/* TTS Metrics Grid */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: '0.65rem' }}>
                <div style={{ padding: '0.65rem 0.85rem', borderRadius: '8px', background: 'var(--bg-subtle)', border: '1px solid var(--border)' }}>
                  <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>TTS Latency</div>
                  <div style={{ fontSize: '1rem', fontWeight: 700, color: 'var(--accent-tts)', fontFamily: 'var(--font-mono)', marginTop: '0.15rem' }}>
                    {ttsResult.tts_latency}s
                  </div>
                </div>

                <div style={{ padding: '0.65rem 0.85rem', borderRadius: '8px', background: 'var(--bg-subtle)', border: '1px solid var(--border)' }}>
                  <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Audio Duration</div>
                  <div style={{ fontSize: '1rem', fontWeight: 700, color: 'var(--text-main)', fontFamily: 'var(--font-mono)', marginTop: '0.15rem' }}>
                    {ttsResult.duration}s
                  </div>
                </div>

                <div style={{ padding: '0.65rem 0.85rem', borderRadius: '8px', background: 'var(--bg-subtle)', border: '1px solid var(--border)' }}>
                  <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Synthesis Engine</div>
                  <div style={{ fontSize: '0.95rem', fontWeight: 700, color: 'var(--accent-call)', marginTop: '0.15rem' }}>
                    {ttsResult.engine}
                  </div>
                </div>

                <div style={{ padding: '0.65rem 0.85rem', borderRadius: '8px', background: 'var(--bg-subtle)', border: '1px solid var(--border)' }}>
                  <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Voice Persona</div>
                  <div style={{ fontSize: '0.95rem', fontWeight: 700, color: 'var(--accent-stt)', marginTop: '0.15rem' }}>
                    {ttsResult.voice}
                  </div>
                </div>
              </div>
            </div>
          )}
        </section>
      )}
    </div>
  );
}
