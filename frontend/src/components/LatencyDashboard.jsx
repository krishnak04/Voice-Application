import React from 'react';
import { Clock, Zap, Cpu, Activity, Volume2 } from 'lucide-react';

export default function LatencyDashboard({ latency }) {
  if (!latency) return null;

  const {
    audio_duration = 0,
    stt_latency = 0,
    accent_latency_ms = null,
    accent_latency_sec = null,
    rtf = 0,
    speed_factor = 0,
    groq_latency = 0,
    tts_latency = 0,
    total_server_time = 0,
    client_rtt = 0,
  } = latency;

  const hasTTS = tts_latency > 0;
  const hasAccent = accent_latency_ms !== null && accent_latency_ms !== undefined;
  const networkOverhead = Math.max(0, +(client_rtt - total_server_time).toFixed(3));

  const totalTime = Math.max(client_rtt, total_server_time, 0.01);
  const sttPct = Math.round((stt_latency / totalTime) * 100);
  const groqPct = Math.round((groq_latency / totalTime) * 100);
  const ttsPct = hasTTS ? Math.round((tts_latency / totalTime) * 100) : 0;
  const netPct = Math.max(0, 100 - sttPct - groqPct - ttsPct);

  return (
    <section className="glass-panel" style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h3 style={{ fontSize: '1rem', fontWeight: 600, color: 'var(--text-main)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Activity size={18} color="#38bdf8" />
          <span>Latency & Telephony Benchmark</span>
        </h3>
        <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
          RTF: <strong style={{ color: rtf < 0.2 ? '#34d399' : '#fbbf24' }}>{rtf}x</strong> ({speed_factor}x real-time)
        </span>
      </div>

      <div className="metrics-grid" style={{ gridTemplateColumns: hasTTS || hasAccent ? 'repeat(auto-fit, minmax(135px, 1fr))' : undefined }}>
        <div className="metric-card">
          <span className="metric-label">Voice-to-Voice RTT</span>
          <span className="metric-val" style={{ color: client_rtt < 1.0 ? '#34d399' : '#38bdf8' }}>{client_rtt}s</span>
          <span className="metric-sub">Full round-trip browser time</span>
        </div>

        <div className="metric-card">
          <span className="metric-label">Speech Recognition (STT)</span>
          <span className="metric-val" style={{ color: '#34d399' }}>{stt_latency}s</span>
          <span className="metric-sub">For {audio_duration}s speech</span>
        </div>

        {hasAccent && (
          <div className="metric-card" style={{ border: '1px solid rgba(56, 189, 248, 0.35)', background: 'rgba(56, 189, 248, 0.05)' }}>
            <span className="metric-label" style={{ color: '#38bdf8' }}>Accent Detection</span>
            <span className="metric-val" style={{ color: '#38bdf8' }}>{accent_latency_ms}ms</span>
            <span className="metric-sub">Acoustic ECAPA-TDNN</span>
          </div>
        )}

        <div className="metric-card">
          <span className="metric-label">Groq LLM</span>
          <span className="metric-val" style={{ color: '#c084fc' }}>{groq_latency}s</span>
          <span className="metric-sub">Conversational reply</span>
        </div>

        {hasTTS && (
          <div className="metric-card">
            <span className="metric-label">TTS Synthesis</span>
            <span className="metric-val" style={{ color: '#f59e0b' }}>{tts_latency}s</span>
            <span className="metric-sub">Voice audio generation</span>
          </div>
        )}

        <div className="metric-card">
          <span className="metric-label">Network / Misc</span>
          <span className="metric-val" style={{ color: '#94a3b8' }}>{networkOverhead}s</span>
          <span className="metric-sub">Upload + serialization</span>
        </div>
      </div>

      {/* Latency Pipeline Breakdown Bar */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
          <span>Pipeline Breakdown</span>
          <span>Server Processing: {total_server_time}s</span>
        </div>
        <div style={{ height: '8px', background: 'rgba(255,255,255,0.06)', borderRadius: '999px', overflow: 'hidden', display: 'flex' }}>
          <div style={{ width: `${sttPct}%`, background: '#34d399' }} title={`STT: ${sttPct}%`} />
          <div style={{ width: `${groqPct}%`, background: '#a855f7' }} title={`Groq: ${groqPct}%`} />
          {hasTTS && <div style={{ width: `${ttsPct}%`, background: '#f59e0b' }} title={`TTS: ${ttsPct}%`} />}
          <div style={{ width: `${netPct}%`, background: '#38bdf8' }} title={`Network/Overhead: ${netPct}%`} />
        </div>
      </div>
    </section>
  );
}
