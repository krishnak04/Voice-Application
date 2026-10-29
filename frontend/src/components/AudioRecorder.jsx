import React, { useEffect, useRef } from 'react';
import { Mic, Square, Loader2, Globe, Cpu } from 'lucide-react';

export const GENERAL_LANGUAGES = [
  { code: 'auto', label: '🌐 Auto-Detect' },
  { code: 'en', label: '🇺🇸 English (en)' },
];

export const INDIAN_LANGUAGES = [
  { code: 'hi', label: '1. Hindi (hi)' },
  { code: 'bn', label: '2. Bengali (bn)' },
  { code: 'ta', label: '3. Tamil (ta)' },
  { code: 'te', label: '4. Telugu (te)' },
  { code: 'mr', label: '5. Marathi (mr)' },
  { code: 'gu', label: '6. Gujarati (gu)' },
  { code: 'kn', label: '7. Kannada (kn)' },
  { code: 'ml', label: '8. Malayalam (ml)' },
  { code: 'pa', label: '9. Punjabi (pa)' },
  { code: 'ur', label: '10. Urdu (ur)' },
  { code: 'as', label: '11. Assamese (as)' },
  { code: 'or', label: '12. Odia / Oriya (or)' },
  { code: 'ne', label: '13. Nepali (ne)' },
  { code: 'sa', label: '14. Sanskrit (sa)' },
  { code: 'sd', label: '15. Sindhi (sd)' },
  { code: 'ks', label: '16. Kashmiri (ks)' },
  { code: 'kok', label: '17. Konkani (kok)' },
  { code: 'mni', label: '18. Manipuri / Meitei (mni)' },
  { code: 'brx', label: '19. Bodo (brx)' },
  { code: 'doi', label: '20. Dogri (doi)' },
  { code: 'sat', label: '21. Santali (sat)' },
  { code: 'mai', label: '22. Maithili (mai)' },
];

export const OTHER_LANGUAGES = [
  { code: 'es', label: '🇪🇸 Spanish (es)' },
  { code: 'fr', label: '🇫🇷 French (fr)' },
  { code: 'de', label: '🇩🇪 German (de)' },
  { code: 'ja', label: '🇯🇵 Japanese (ja)' },
  { code: 'zh', label: '🇨🇳 Chinese (zh)' },
  { code: 'ru', label: '🇷🇺 Russian (ru)' },
  { code: 'ar', label: '🇸🇦 Arabic (ar)' },
  { code: 'pt', label: '🇧🇷 Portuguese (pt)' },
];

export const LANGUAGES = [...GENERAL_LANGUAGES, ...INDIAN_LANGUAGES, ...OTHER_LANGUAGES];

export const WHISPER_MODELS = [
  { id: 'tiny', label: 'tiny (~39M - Ultra-fast)', type: 'offline', category: 'Small' },
  { id: 'base', label: 'base (~74M - Fast Multi)', type: 'offline', category: 'Standard' },
  { id: 'small', label: 'small (~244M - Higher accuracy)', type: 'offline', category: 'Large' },
  { id: 'medium', label: 'medium (~769M - Production Indic)', type: 'offline', category: 'Large' },
  { id: 'large-v3-turbo', label: 'large-v3-turbo (~809M - SOTA)', type: 'offline', category: 'Large' },
  { id: 'base.en', label: 'base.en (~74M - English-only)', type: 'offline', category: 'Standard' },
];

export const SHERPA_MODELS = [
  { id: 'sherpa-onnx-streaming-zipformer-en-2023-06-26', label: '⚡ Streaming Standard (~65.6M)', type: 'streaming', category: 'Standard', status: 'Ready' },
  { id: 'sherpa-onnx-streaming-zipformer-en-20M-2023-02-17', label: '⚡ Streaming Small Lightweight (~20M)', type: 'streaming', category: 'Small', status: 'Ready' },
  { id: 'sherpa-onnx-streaming-zipformer-en-2023-06-21', label: '⚡ Streaming LibriSpeech (~65.6M)', type: 'streaming', category: 'Standard', status: 'Ready' },
  { id: 'sherpa-onnx-zipformer-small-en-2023-06-26', label: '📦 Offline Small (~30M)', type: 'offline', category: 'Small', status: 'Ready' },
  { id: 'sherpa-onnx-zipformer-large-en-2023-06-26', label: '📦 Offline Large (~148M)', type: 'offline', category: 'Large', status: 'Ready' },
  { id: 'sherpa-onnx-zipformer-gigaspeech-2023-12-12', label: '📦 Offline GigaSpeech (~65.6M)', type: 'offline', category: 'Standard', status: 'Ready' },
];

export const MODELS = WHISPER_MODELS;

export default function AudioRecorder({
  isRecording,
  recordingDuration,
  isProcessing,
  error,
  analyser,
  language,
  onLanguageChange,
  model,
  onModelChange,
  engine = 'whisper',
  onEngineChange,
  onStart,
  onStop,
  lastAccentAnalysis = null,
}) {
  const canvasRef = useRef(null);
  const animationFrameRef = useRef(null);

  useEffect(() => {
    if (!isRecording || !analyser || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    const draw = () => {
      animationFrameRef.current = requestAnimationFrame(draw);
      analyser.getByteFrequencyData(dataArray);

      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const barWidth = (canvas.width / bufferLength) * 1.5;
      let x = 0;

      for (let i = 0; i < bufferLength; i++) {
        const barHeight = (dataArray[i] / 255) * canvas.height;
        ctx.fillStyle = `rgba(56, 189, 248, ${0.4 + (dataArray[i] / 255) * 0.6})`;
        ctx.fillRect(x, canvas.height - barHeight, barWidth - 2, barHeight);
        x += barWidth;
      }
    };

    draw();
    return () => {
      if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
    };
  }, [isRecording, analyser]);

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = (seconds % 60).toFixed(1);
    return `${mins < 10 ? '0' : ''}${mins}:${secs < 10 ? '0' : ''}${secs}`;
  };

  return (
    <section className="glass-panel mic-section">
      {/* Controls row for Language and Model selection */}
      <div style={{ display: 'flex', gap: '0.75rem', marginBottom: '1.25rem', flexWrap: 'wrap', justifyContent: 'center' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
          <Globe size={15} color="#38bdf8" />
          <select
            value={language}
            onChange={(e) => onLanguageChange(e.target.value)}
            disabled={isRecording || isProcessing}
            style={{
              padding: '0.45rem 0.75rem',
              borderRadius: '8px',
              background: 'var(--bg-input)',
              border: '1px solid var(--border)',
              color: 'var(--text-main)',
              fontSize: '0.85rem',
              fontFamily: 'var(--font-sans)',
              cursor: 'pointer'
            }}
          >
            <optgroup label="General">
              {GENERAL_LANGUAGES.map((l) => (
                <option key={l.code} value={l.code}>
                  {l.label}
                </option>
              ))}
            </optgroup>
            <optgroup label="🇮🇳 22 Official Indian Languages">
              {INDIAN_LANGUAGES.map((l) => (
                <option key={l.code} value={l.code}>
                  {l.label}
                </option>
              ))}
            </optgroup>
            <optgroup label="Other Languages">
              {OTHER_LANGUAGES.map((l) => (
                <option key={l.code} value={l.code}>
                  {l.label}
                </option>
              ))}
            </optgroup>
          </select>
        </div>

        {/* STT Engine Selector */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
          <select
            value={engine}
            onChange={(e) => {
              const nextEngine = e.target.value;
              if (onEngineChange) onEngineChange(nextEngine);
              if (nextEngine === 'sherpa') {
                onModelChange('sherpa-onnx-streaming-zipformer-en-2023-06-26');
              } else {
                onModelChange('base');
              }
            }}
            disabled={isRecording || isProcessing}
            style={{
              padding: '0.45rem 0.65rem',
              borderRadius: '8px',
              background: 'var(--bg-input)',
              border: '1px solid var(--border)',
              color: 'var(--text-main)',
              fontSize: '0.85rem',
              fontFamily: 'var(--font-sans)',
              fontWeight: 600,
              cursor: 'pointer'
            }}
            title="Select Speech-to-Text Recognition Engine"
          >
            <option value="whisper">🎙️ Faster-Whisper</option>
            <option value="sherpa">⚡ Sherpa-ONNX</option>
          </select>
        </div>

        {/* Model Selector Filtered by Engine */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
          <Cpu size={15} color={engine === 'sherpa' ? '#9333ea' : 'var(--accent-benchmark)'} />
          <select
            value={model}
            onChange={(e) => onModelChange(e.target.value)}
            disabled={isRecording || isProcessing}
            style={{
              padding: '0.45rem 0.75rem',
              borderRadius: '8px',
              background: 'var(--bg-input)',
              border: `1px solid ${engine === 'sherpa' ? 'rgba(168, 85, 247, 0.4)' : 'var(--border)'}`,
              color: 'var(--text-main)',
              fontSize: '0.85rem',
              fontFamily: 'var(--font-sans)',
              cursor: 'pointer'
            }}
          >
            {(engine === 'sherpa' ? SHERPA_MODELS : WHISPER_MODELS).map((m) => (
              <option key={m.id} value={m.id}>
                {m.label} • {m.status || 'Ready'}
              </option>
            ))}
          </select>
        </div>

        {/* Model Type Badge */}
        <span
          style={{
            fontSize: '0.72rem',
            fontWeight: 600,
            padding: '0.2rem 0.5rem',
            borderRadius: '6px',
            background: (engine === 'sherpa' && model.includes('streaming')) ? 'rgba(5, 150, 105, 0.15)' : 'rgba(124, 58, 237, 0.15)',
            color: (engine === 'sherpa' && model.includes('streaming')) ? '#059669' : '#7c3aed',
            fontFamily: 'var(--font-mono)'
          }}
        >
          {engine === 'sherpa'
            ? (model.includes('streaming') ? '⚡ Streaming Engine' : '📦 Offline Transducer')
            : '📦 Offline Transducer'}
        </span>

        {engine === 'sherpa' && language !== 'en' && language !== 'auto' && (
          <span style={{ fontSize: '0.72rem', color: '#f59e0b', width: '100%', textAlign: 'center' }}>
            ⚠️ Sherpa models are English-only. For Indic/Hindi speech, switch engine to Faster-Whisper.
          </span>
        )}
      </div>

      {/* Automatic English Accent Status Banner in Benchmark Mode */}
      <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '1.15rem' }}>
        <div
          style={{
            padding: '0.38rem 1rem',
            borderRadius: '9999px',
            background: lastAccentAnalysis && lastAccentAnalysis.status === 'detected'
              ? 'rgba(16, 185, 129, 0.12)'
              : 'rgba(56, 189, 248, 0.10)',
            border: `1px solid ${
              lastAccentAnalysis && lastAccentAnalysis.status === 'detected'
                ? 'rgba(16, 185, 129, 0.35)'
                : 'rgba(56, 189, 248, 0.3)'
            }`,
            display: 'inline-flex',
            alignItems: 'center',
            gap: '0.5rem',
            fontSize: '0.82rem',
            fontWeight: 600,
            color: lastAccentAnalysis && lastAccentAnalysis.status === 'detected' ? '#059669' : '#0284c7',
            boxShadow: 'var(--shadow-xs)'
          }}
        >
          <span>{lastAccentAnalysis?.flag || '🌐'}</span>
          <span>
            {lastAccentAnalysis && lastAccentAnalysis.status === 'detected' ? (
              <>
                Detected Variety: <strong>{lastAccentAnalysis.accent}</strong>
                <span style={{ fontSize: '0.74rem', opacity: 0.85, marginLeft: '0.35rem' }}>
                  ({Math.round(lastAccentAnalysis.confidence * 100)}% • {lastAccentAnalysis.latency_ms}ms)
                </span>
              </>
            ) : isRecording ? (
              'Capturing mic audio for automatic accent & STT analysis...'
            ) : isProcessing ? (
              'Acoustic accent detection & speech transcription...'
            ) : (
              'Automatic English Accent Detection Active (ECAPA-TDNN)'
            )}
          </span>
        </div>
      </div>

      <div className="mic-button-wrapper">
        <button
          className={`mic-btn ${isRecording ? 'recording' : ''}`}
          onClick={isRecording ? onStop : onStart}
          disabled={isProcessing}
          id="btn-mic-toggle"
        >
          {isProcessing ? (
            <Loader2 size={36} className="spin" style={{ animation: 'spin 1s linear infinite' }} />
          ) : isRecording ? (
            <Square size={32} />
          ) : (
            <Mic size={36} />
          )}
        </button>
      </div>

      <div style={{ minHeight: '50px' }}>
        {isRecording && (
          <>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: '1.25rem', color: '#f43f5e', fontWeight: 600 }}>
              {formatTime(recordingDuration)}
            </div>
            <canvas ref={canvasRef} className="audio-visualizer-canvas" width={280} height={38} />
          </>
        )}

        {isProcessing && (
          <p style={{ color: 'var(--accent-cyan)', fontSize: '0.95rem' }}>
            Transcribing audio with {engine === 'sherpa' ? 'Sherpa-ONNX' : 'Faster-Whisper'} & analyzing English accent...
          </p>
        )}

        {!isRecording && !isProcessing && (
          <p style={{ color: 'var(--text-muted)', fontSize: '0.95rem' }}>
            Click the microphone to record speech
          </p>
        )}

        {error && (
          <p style={{ color: '#f43f5e', fontSize: '0.85rem', marginTop: '0.5rem' }}>
            {error}
          </p>
        )}
      </div>
    </section>
  );
}

