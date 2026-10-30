export function getDefaultServerUrl() {
  if (typeof window !== 'undefined' && window.location && window.location.hostname) {
    const protocol = window.location.protocol || 'http:';
    const hostname = window.location.hostname;
    return `${protocol}//${hostname}:8024`;
  }
  return 'http://localhost:8024';
}

export function getServerUrl() {
  const saved = localStorage.getItem('voice_benchmark_server_url');
  if (saved) {
    if (saved === 'http://localhost:8000' || saved === 'http://127.0.0.1:8000') {
      return getDefaultServerUrl();
    }
    if (saved.includes(':8000')) {
      const updated = saved.replace(':8000', ':8024');
      localStorage.setItem('voice_benchmark_server_url', updated);
      return updated;
    }
    return saved;
  }
  return getDefaultServerUrl();
}

export function setServerUrl(url) {
  if (!url || !url.trim() || url.trim() === getDefaultServerUrl()) {
    localStorage.removeItem('voice_benchmark_server_url');
    return;
  }
  const cleanUrl = url.trim().replace(/\/+$/, '');
  localStorage.setItem('voice_benchmark_server_url', cleanUrl);
}

export async function checkServerHealth() {
  const baseUrl = getServerUrl();
  try {
    const res = await fetch(`${baseUrl}/api/health`, { method: 'GET' });
    if (!res.ok) throw new Error(`Health check failed: ${res.statusText}`);
    return await res.json();
  } catch (err) {
    return { status: 'offline', error: err.message };
  }
}

/**
 * Pure Voice-to-Text STT API call
 */
export async function transcribeAudio(audioBlob, options = {}) {
  const baseUrl = getServerUrl();
  const formData = new FormData();
  formData.append('audio', audioBlob, 'mic_recording.webm');

  if (options.language && options.language !== 'auto') {
    formData.append('language', options.language);
  }
  if (options.model) {
    formData.append('model', options.model);
  }
  if (options.engine) {
    formData.append('engine', options.engine);
  }

  const response = await fetch(`${baseUrl}/api/transcribe`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || `Transcription failed with ${response.status}`);
  }

  return await response.json();
}

/**
 * Dedicated Groq LLM Chat API call
 */
export async function sendChatMessage(message, options = {}) {
  const baseUrl = getServerUrl();
  const payload = {
    message,
    groq_api_key: options.groqApiKey || undefined,
    groq_model: options.groqModel || undefined,
    history: options.history || undefined,
  };

  const response = await fetch(`${baseUrl}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || `Groq LLM chat failed with ${response.status}`);
  }

  return await response.json();
}

/**
 * Text-to-Speech (TTS) standalone call
 */
export async function synthesizeSpeech(text, options = {}) {
  const baseUrl = getServerUrl();
  const payload = {
    text,
    voice: options.voice || undefined,
    speed: options.speed || 1.0,
    pitch: options.pitch || undefined,
    engine: options.engine || undefined,
  };

  const response = await fetch(`${baseUrl}/api/tts`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || `TTS synthesis failed with ${response.status}`);
  }

  return await response.json();
}

/**
 * Retrieve curated real human voices with metadata, categories, accents, and preview samples
 */
export async function getHumanVoices() {
  const baseUrl = getServerUrl();
  try {
    const res = await fetch(`${baseUrl}/api/tts/voices`, { method: 'GET' });
    if (!res.ok) throw new Error('Failed to get human voices');
    return await res.json();
  } catch (err) {
    return {
      engine: 'edge_tts',
      total_human_voices: 0,
      categories: ['Adult Female', 'Adult Male', 'Child / Youth', 'Mature / Narrator'],
      voices: [],
      elderly_limitation_notice: 'Edge-TTS provides child/youth voices (Ana, Maisie) and diverse adult voices. Dedicated elderly/senior character models are not published by Microsoft Edge-TTS; authoritative mature human adult voices are provided instead.'
    };
  }
}

/**
 * Get active TTS engine info and available voices
 */
export async function getTTSInfo() {
  const baseUrl = getServerUrl();
  try {
    const res = await fetch(`${baseUrl}/api/tts/info`, { method: 'GET' });
    if (!res.ok) throw new Error('Failed to get TTS info');
    return await res.json();
  } catch (err) {
    return { active_engine: 'none', available_voices: [] };
  }
}

/**
 * Get Sherpa-ONNX streaming engine metadata
 */
export async function getSherpaInfo() {
  const baseUrl = getServerUrl();
  try {
    const res = await fetch(`${baseUrl}/api/sherpa/info`, { method: 'GET' });
    if (!res.ok) throw new Error('Failed to get Sherpa info');
    return await res.json();
  } catch (err) {
    return { available: false, error: err.message };
  }
}

/**
 * List all verified Sherpa-ONNX English models with download status
 */
export async function getSherpaModels() {
  const baseUrl = getServerUrl();
  try {
    const res = await fetch(`${baseUrl}/api/sherpa/models`, { method: 'GET' });
    if (!res.ok) throw new Error('Failed to fetch Sherpa models');
    return await res.json();
  } catch (err) {
    return { models: [], active_model: null, error: err.message };
  }
}

/**
 * Safely select/switch active Sherpa-ONNX model
 */
export async function selectSherpaModel(modelId) {
  const baseUrl = getServerUrl();
  const res = await fetch(`${baseUrl}/api/sherpa/models/select`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model_id: modelId })
  });
  const data = await res.json();
  if (!res.ok) {
    throw new Error(data.detail || data.message || 'Failed to select Sherpa model');
  }
  return data;
}

/**
 * Download a verified Sherpa-ONNX model
 */
export async function downloadSherpaModel(modelId) {
  const baseUrl = getServerUrl();
  const res = await fetch(`${baseUrl}/api/sherpa/models/download`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model_id: modelId })
  });
  const data = await res.json();
  if (!res.ok) {
    throw new Error(data.detail || data.message || 'Failed to download Sherpa model');
  }
  return data;
}

/**
 * Retrieve empirical benchmark comparison metrics
 */
export async function getSherpaBenchmarks() {
  const baseUrl = getServerUrl();
  try {
    const res = await fetch(`${baseUrl}/api/sherpa/benchmark`, { method: 'GET' });
    if (!res.ok) throw new Error('Failed to fetch Sherpa benchmarks');
    return await res.json();
  } catch (err) {
    return { status: 'error', error: err.message };
  }
}

/**
 * Resolve full-duplex WebSocket URL from current server URL
 */
export function getWebSocketUrl(path = '/ws/call', preferSameOrigin = false) {
  const cleanPath = '/' + path.replace(/^\/+/, '');

  if (typeof window !== 'undefined' && window.location) {
    const isHttps = window.location.protocol === 'https:';
    const wsProto = isHttps ? 'wss:' : 'ws:';

    // In HTTPS environments, browsers strictly block mixed-content ws:// connections
    // and require wss:// to the same origin or reverse proxy.
    if (isHttps || preferSameOrigin) {
      return `${wsProto}//${window.location.host}${cleanPath}`;
    }

    const saved = localStorage.getItem('voice_benchmark_server_url');
    if (saved && saved.trim()) {
      const cleanSaved = saved.trim().replace(/\/+$/, '');
      const savedHost = cleanSaved.replace(/^https?:\/\//, '');
      const savedProto = cleanSaved.startsWith('https:') ? 'wss:' : 'ws:';
      return `${savedProto}//${savedHost}${cleanPath}`;
    }

    const hostname = window.location.hostname || 'localhost';
    return `${wsProto}//${hostname}:8024${cleanPath}`;
  }

  return `ws://localhost:8024${cleanPath}`;
}

/**
 * Return secondary fallback WebSocket URL (proxied through Vite/reverse proxy on port 8023)
 */
export function getFallbackWebSocketUrl(path = '/ws/call') {
  const cleanPath = '/' + path.replace(/^\/+/, '');
  if (typeof window !== 'undefined' && window.location) {
    const isHttps = window.location.protocol === 'https:';
    const wsProto = isHttps ? 'wss:' : 'ws:';
    return `${wsProto}//${window.location.host}${cleanPath}`;
  }
  return `ws://localhost:8023${cleanPath}`;
}


/**
 * Simulated Phone Call Turn
 * Audio In -> Faster-Whisper -> Groq LLM with context -> TTS -> Audio Out
 */
export async function sendVoiceCallTurn(audioBlob, history = [], options = {}) {
  const baseUrl = getServerUrl();
  const clientStart = performance.now();

  const formData = new FormData();
  formData.append('audio', audioBlob, 'phone_turn.webm');

  if (history && history.length > 0) {
    formData.append('history', JSON.stringify(history));
  }
  if (options.language && options.language !== 'auto') {
    formData.append('language', options.language);
  }
  if (options.whisperModel) {
    formData.append('whisper_model', options.whisperModel);
  }
  if (options.sttEngine) {
    formData.append('stt_engine', options.sttEngine);
  }
  if (options.groqApiKey) {
    formData.append('groq_api_key', options.groqApiKey);
  }
  if (options.groqModel) {
    formData.append('groq_model', options.groqModel);
  }
  if (options.ttsVoice) {
    formData.append('tts_voice', options.ttsVoice);
  }
  if (options.ttsPitch) {
    formData.append('tts_pitch', options.ttsPitch);
  }
  if (options.ttsSpeed) {
    formData.append('tts_speed', String(options.ttsSpeed));
  }
  if (options.ttsEngine) {
    formData.append('tts_engine', options.ttsEngine);
  }

  const response = await fetch(`${baseUrl}/api/call/turn`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || `Call turn failed with ${response.status}`);
  }

  const data = await response.json();
  const clientEnd = performance.now();
  data.latency.client_rtt = +((clientEnd - clientStart) / 1000).toFixed(3);

  return data;
}

/**
 * Full Pipeline Benchmark (STT + Groq Text)
 */
export async function sendAudioForBenchmark(audioBlob, options = {}) {
  const clientStart = performance.now();

  const sttData = await transcribeAudio(audioBlob, {
    language: options.language,
    model: options.model,
    engine: options.engine,
  });

  const transcript = (sttData.transcript || '').trim();
  let chatData = null;

  if (transcript) {
    chatData = await sendChatMessage(transcript, {
      groqApiKey: options.groqApiKey,
      groqModel: options.groqModel,
    });
  } else {
    chatData = {
      reply: '[No audible speech detected. Please speak closer to the microphone]',
      groq_latency: 0.0,
      model: options.groqModel || 'default',
    };
  }

  const clientEnd = performance.now();
  const clientRtt = +((clientEnd - clientStart) / 1000).toFixed(3);
  const groqLatency = chatData ? chatData.groq_latency : 0.0;
  const totalServerTime = +(sttData.stt_latency + groqLatency).toFixed(3);

  return {
    transcript: sttData.transcript,
    reply: chatData ? chatData.reply : '',
    language: sttData.language,
    accent_analysis: sttData.accent_analysis || null,
    latency: {
      audio_duration: sttData.duration,
      stt_latency: sttData.stt_latency,
      accent_latency_ms: sttData.accent_analysis ? sttData.accent_analysis.latency_ms : null,
      accent_latency_sec: sttData.accent_analysis ? +(sttData.accent_analysis.latency_ms / 1000).toFixed(3) : null,
      rtf: sttData.rtf,
      speed_factor: sttData.speed_factor,
      groq_latency: groqLatency,
      tts_latency: null,
      total_server_time: totalServerTime,
      client_rtt: clientRtt,
    },
    device: sttData.device,
    model_name: sttData.model_name,
  };
}

/**
 * Helper to play base64 audio directly in browser
 */
let currentAudio = null;

export function playAudioFromBase64(base64Data, format = 'wav', onEnded = null) {
  stopAudioPlayback();
  const mimeType = format === 'mp3' ? 'audio/mpeg' : 'audio/wav';
  const audioSrc = `data:${mimeType};base64,${base64Data}`;
  const audio = new Audio(audioSrc);
  currentAudio = audio;

  if (onEnded) {
    audio.onended = () => {
      currentAudio = null;
      onEnded();
    };
  }

  audio.play().catch((err) => {
    console.warn('Audio playback error (user interaction needed):', err);
  });

  return audio;
}

export function stopAudioPlayback() {
  if (currentAudio) {
    try {
      currentAudio.pause();
      currentAudio.currentTime = 0;
    } catch (e) {
      // ignore
    }
    currentAudio = null;
  }
}

/**
 * STT Unified & Engine-Specific APIs (Whisper & Sherpa-ONNX)
 */
export async function getSttModels(engine = null) {
  const baseUrl = getServerUrl();
  const url = engine ? `${baseUrl}/api/stt/models?engine=${encodeURIComponent(engine)}` : `${baseUrl}/api/stt/models`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to fetch STT models: ${res.statusText}`);
  return await res.json();
}

export async function selectSttModel(engine, modelId) {
  const baseUrl = getServerUrl();
  const res = await fetch(`${baseUrl}/api/stt/select`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ engine, model_id: modelId })
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || err.message || `Failed to select ${engine} model`);
  }
  return await res.json();
}

export async function getWhisperModels() {
  const baseUrl = getServerUrl();
  const res = await fetch(`${baseUrl}/api/whisper/models`);
  if (!res.ok) throw new Error(`Failed to fetch Whisper models: ${res.statusText}`);
  return await res.json();
}

export async function selectWhisperModel(modelId) {
  const baseUrl = getServerUrl();
  const res = await fetch(`${baseUrl}/api/whisper/models/select`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model_id: modelId })
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || err.message || 'Failed to select Whisper model');
  }
  return await res.json();
}

/**
 * Automatic English Accent Detection APIs
 */
export async function getAccentInfo() {
  const baseUrl = getServerUrl();
  const res = await fetch(`${baseUrl}/api/accent/info`);
  if (!res.ok) throw new Error(`Failed to fetch accent detector info: ${res.statusText}`);
  return await res.json();
}

export async function detectAudioAccent(audioBlob, sessionId = null) {
  const baseUrl = getServerUrl();
  const formData = new FormData();
  formData.append('audio', audioBlob, 'accent_sample.webm');
  if (sessionId) {
    formData.append('session_id', sessionId);
  }

  const res = await fetch(`${baseUrl}/api/accent/detect`, {
    method: 'POST',
    body: formData
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Accent detection failed with ${res.status}`);
  }
  return await res.json();
}


