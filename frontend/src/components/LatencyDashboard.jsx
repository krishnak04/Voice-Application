import React from 'react';
import { Activity, Clock, Zap, Cpu, Volume2, Radio, PlayCircle, Layers } from 'lucide-react';

export default function LatencyDashboard({ latency }) {
  if (!latency) return null;

  const {
    audio_duration = null,
    stt_latency = null,
    first_partial_latency_ms = null,
    accent_latency_ms = null,
    rtf = null,
    speed_factor = null,
    groq_first_token_ms = null,
    groq_latency = null,
    tts_first_chunk_ms = null,
    tts_latency = null,
    first_audible_ms = null,
    total_server_time = null,
    client_rtt = null,
  } = latency;

  // Format helpers with rigorous monotonic checks - never report 0 when measurement is missing
  const formatSec = (val) => {
    if (val === null || val === undefined || isNaN(val) || val <= 0) return 'N/A';
    return `${Number(val).toFixed(3)}s`;
  };

  const formatMs = (val) => {
    if (val === null || val === undefined || isNaN(val) || val <= 0) return 'N/A';
    return `${Math.round(val)}ms`;
  };

  const hasTTS = tts_latency !== null && tts_latency > 0;
  const hasFirstAudible = first_audible_ms !== null && first_audible_ms > 0;
  const hasFirstPartial = first_partial_latency_ms !== null && first_partial_latency_ms > 0;
  const hasGroqTTFT = groq_first_token_ms !== null && groq_first_token_ms > 0;
  const hasAccent = accent_latency_ms !== null && accent_latency_ms > 0;

  const validServerTime = total_server_time && total_server_time > 0 ? total_server_time : 0;
  const validClientRtt = client_rtt && client_rtt > 0 ? client_rtt : validServerTime;
  const networkOverhead = Math.max(0, +(validClientRtt - validServerTime).toFixed(3));

  // Compute breakdown percentages from non-overlapping primary server stages
  const validStt = stt_latency && stt_latency > 0 ? stt_latency : 0;
  const validGroq = groq_latency && groq_latency > 0 ? groq_latency : 0;
  const validTts = tts_latency && tts_latency > 0 ? tts_latency : 0;
  const sumStages = validStt + validGroq + validTts + networkOverhead;
  const baseTotal = sumStages > 0 ? sumStages : (validClientRtt > 0 ? validClientRtt : 1);

  const sttPct = Math.round((validStt / baseTotal) * 100);
  const groqPct = Math.round((validGroq / baseTotal) * 100);
  const ttsPct = validTts > 0 ? Math.round((validTts / baseTotal) * 100) : 0;
  const netPct = Math.max(0, 100 - sttPct - groqPct - ttsPct);

  return (
    <section className="glass-panel" style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
      {/* Header with Title and RTF Benchmark */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.5rem' }}>
        <h3 style={{ fontSize: '1rem', fontWeight: 600, color: 'var(--text-main)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Activity size={18} color="#38bdf8" />
          <span>Latency & Telephony Voice Benchmark</span>
        </h3>
        <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
          RTF:{' '}
          {rtf && rtf > 0 ? (
            <strong style={{ color: rtf < 0.2 ? '#34d399' : '#fbbf24' }}>
              {rtf}x {speed_factor ? `(${speed_factor}x real-time)` : `(${+(1 / rtf).toFixed(1)}x real-time)`}
            </strong>
          ) : (
            <strong style={{ color: '#94a3b8' }}>N/A</strong>
          )}
        </span>
      </div>

      {/* Primary Metrics Grid */}
      <div className="metrics-grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))' }}>
        {/* 1. Voice-to-Voice RTT */}
        <div className="metric-card">
          <span className="metric-label">Voice-to-Voice RTT</span>
          <span className="metric-val" style={{ color: client_rtt && client_rtt < 1.0 ? '#34d399' : '#38bdf8' }}>
            {formatSec(client_rtt)}
          </span>
          <span className="metric-sub">Speech end → voice reply</span>
        </div>

        {/* 2. First Audible Response */}
        <div className="metric-card" style={hasFirstAudible ? { border: '1px solid rgba(52, 211, 153, 0.3)', background: 'rgba(52, 211, 153, 0.04)' } : undefined}>
          <span className="metric-label" style={{ color: hasFirstAudible ? '#34d399' : undefined }}>First Audible Audio</span>
          <span className="metric-val" style={{ color: '#34d399' }}>
            {formatMs(first_audible_ms || tts_first_chunk_ms)}
          </span>
          <span className="metric-sub">Speaker playback start</span>
        </div>

        {/* 3. First Partial Transcript (Streaming STT) */}
        <div className="metric-card">
          <span className="metric-label">First Partial STT</span>
          <span className="metric-val" style={{ color: hasFirstPartial ? '#38bdf8' : '#94a3b8' }}>
            {formatMs(first_partial_latency_ms)}
          </span>
          <span className="metric-sub">Speech start → partial</span>
        </div>

        {/* 4. Final STT Latency */}
        <div className="metric-card">
          <span className="metric-label">Speech Recognition</span>
          <span className="metric-val" style={{ color: '#34d399' }}>
            {formatSec(stt_latency)}
          </span>
          <span className="metric-sub">
            {audio_duration && audio_duration > 0 ? `For ${Number(audio_duration).toFixed(2)}s speech` : 'Utterance finalization'}
          </span>
        </div>

        {/* 5. Groq TTFT (First Token) */}
        <div className="metric-card">
          <span className="metric-label">Groq First Token</span>
          <span className="metric-val" style={{ color: hasGroqTTFT ? '#c084fc' : '#94a3b8' }}>
            {formatMs(groq_first_token_ms)}
          </span>
          <span className="metric-sub">Time to first token (TTFT)</span>
        </div>

        {/* 6. Groq LLM Completion */}
        <div className="metric-card">
          <span className="metric-label">Groq LLM Total</span>
          <span className="metric-val" style={{ color: '#c084fc' }}>
            {formatSec(groq_latency)}
          </span>
          <span className="metric-sub">Conversational completion</span>
        </div>

        {/* 7. TTS First Audio Chunk (TTFA) */}
        {hasTTS && (
          <div className="metric-card">
            <span className="metric-label">TTS First Chunk</span>
            <span className="metric-val" style={{ color: '#f59e0b' }}>
              {formatMs(tts_first_chunk_ms)}
            </span>
            <span className="metric-sub">Time to first audio (TTFA)</span>
          </div>
        )}

        {/* 8. TTS Synthesis Total */}
        {hasTTS && (
          <div className="metric-card">
            <span className="metric-label">TTS Synthesis</span>
            <span className="metric-val" style={{ color: '#f59e0b' }}>
              {formatSec(tts_latency)}
            </span>
            <span className="metric-sub">Audio stream generation</span>
          </div>
        )}

        {/* 9. Accent Detection (Acoustic) */}
        {hasAccent && (
          <div className="metric-card" style={{ border: '1px solid rgba(56, 189, 248, 0.35)', background: 'rgba(56, 189, 248, 0.05)' }}>
            <span className="metric-label" style={{ color: '#38bdf8' }}>Accent Detection</span>
            <span className="metric-val" style={{ color: '#38bdf8' }}>
              {formatMs(accent_latency_ms)}
            </span>
            <span className="metric-sub">Acoustic ECAPA-TDNN</span>
          </div>
        )}

        {/* 10. Network & Overhead */}
        <div className="metric-card">
          <span className="metric-label">Network / Misc</span>
          <span className="metric-val" style={{ color: '#94a3b8' }}>
            {formatSec(networkOverhead)}
          </span>
          <span className="metric-sub">Transport & serialization</span>
        </div>
      </div>

      {/* Latency Pipeline Breakdown Bar */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
          <span>Pipeline Stage Breakdown</span>
          <span>
            Server Compute: <strong>{total_server_time ? formatSec(total_server_time) : 'N/A'}</strong>
          </span>
        </div>
        <div style={{ height: '8px', background: 'rgba(255,255,255,0.06)', borderRadius: '999px', overflow: 'hidden', display: 'flex' }}>
          {sttPct > 0 && <div style={{ width: `${sttPct}%`, background: '#34d399' }} title={`STT: ${sttPct}% (${formatSec(stt_latency)})`} />}
          {groqPct > 0 && <div style={{ width: `${groqPct}%`, background: '#a855f7' }} title={`Groq LLM: ${groqPct}% (${formatSec(groq_latency)})`} />}
          {ttsPct > 0 && <div style={{ width: `${ttsPct}%`, background: '#f59e0b' }} title={`TTS Synthesis: ${ttsPct}% (${formatSec(tts_latency)})`} />}
          {netPct > 0 && <div style={{ width: `${netPct}%`, background: '#38bdf8' }} title={`Network/Overhead: ${netPct}% (${formatSec(networkOverhead)})`} />}
        </div>
        <div style={{ display: 'flex', gap: '1rem', fontSize: '0.7rem', color: 'var(--text-muted)', flexWrap: 'wrap', marginTop: '0.2rem' }}>
          <span style={{ display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
            <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#34d399', display: 'inline-block' }} /> STT ({sttPct}%)
          </span>
          <span style={{ display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
            <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#a855f7', display: 'inline-block' }} /> Groq LLM ({groqPct}%)
          </span>
          {hasTTS && (
            <span style={{ display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
              <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#f59e0b', display: 'inline-block' }} /> TTS ({ttsPct}%)
            </span>
          )}
          <span style={{ display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
            <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#38bdf8', display: 'inline-block' }} /> Network / Misc ({netPct}%)
          </span>
        </div>
      </div>
    </section>
  );
}
