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

export const MODELS = [
  { id: 'base', label: 'base (Dev PC - Fast Multi)' },
  { id: 'small', label: 'small (Higher accuracy)' },
  { id: 'medium', label: 'medium (Production Indic)' },
  { id: 'large-v3-turbo', label: 'large-v3-turbo (SOTA)' },
  { id: 'tiny', label: 'tiny (Ultra-fast)' },
  { id: 'base.en', label: 'base.en (English-only)' },
];

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
  onStart,
  onStop,
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

        <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
          <Cpu size={15} color="var(--accent-benchmark)" />
          <select
            value={model}
            onChange={(e) => onModelChange(e.target.value)}
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
            {MODELS.map((m) => (
              <option key={m.id} value={m.id}>
                {m.label}
              </option>
            ))}
          </select>
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
            Transcribing audio with Faster-Whisper & requesting Groq reply...
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

