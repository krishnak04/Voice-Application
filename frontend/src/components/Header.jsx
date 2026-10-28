import React, { useState, useEffect } from 'react';
import { Mic2, Settings, Cpu, CheckCircle2, XCircle, Sun, Moon } from 'lucide-react';

export default function Header({ health, onOpenSettings, onOpenSherpa, sttEngine, sttModel }) {
  const isOnline = health && health.status === 'online';
  const isCuda = health && health.device === 'cuda';

  const [theme, setTheme] = useState(() => {
    return localStorage.getItem('app_theme') || 'light';
  });

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('app_theme', theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme(prev => (prev === 'light' ? 'dark' : 'light'));
  };

  const getSherpaLabel = () => {
    const active = sttModel || health?.sherpa_active_model || '';
    if (active.includes('20M')) return '20M Small (Streaming)';
    if (active.includes('large')) return 'Large (~146M Offline)';
    if (active.includes('small-en')) return 'Small (~63M Offline)';
    if (active.includes('gigaspeech')) return 'GigaSpeech (~129M)';
    if (active.includes('2023-02-21')) return 'Zipformer Standard (~65M)';
    return 'Zipformer Standard (Streaming)';
  };

  const activeEngine = sttEngine || (sttModel && sttModel.startsWith('sherpa') ? 'sherpa' : 'whisper');

  return (
    <header className="app-header glass-panel">
      <div className="brand-title">
        <Mic2 size={24} color={theme === 'light' ? '#0284c7' : '#38bdf8'} />
        <span>Voice Latency Benchmark</span>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem', flexWrap: 'wrap' }}>
        <div className={`badge ${isOnline ? 'badge-green' : 'badge-amber'}`}>
          {isOnline ? <CheckCircle2 size={13} /> : <XCircle size={13} />}
          <span>{isOnline ? 'FastAPI Online' : 'Connecting...'}</span>
        </div>

        {/* Active STT Engine & Model Interactive Badge */}
        {isOnline && (
          <button
            type="button"
            className="badge"
            onClick={onOpenSherpa}
            style={{
              background: activeEngine === 'sherpa' ? 'rgba(168, 85, 247, 0.14)' : 'rgba(2, 132, 199, 0.14)',
              color: activeEngine === 'sherpa' ? '#9333ea' : '#0284c7',
              border: `1px solid ${activeEngine === 'sherpa' ? 'rgba(168, 85, 247, 0.35)' : 'rgba(2, 132, 199, 0.35)'}`,
              display: 'flex',
              alignItems: 'center',
              gap: '0.35rem',
              cursor: 'pointer',
              fontWeight: 600,
              padding: '0.3rem 0.7rem',
              borderRadius: '9999px',
              transition: 'all 0.2s ease'
            }}
            title="Click to switch STT engines, select models, or view benchmark metrics"
          >
            <span style={{ fontSize: '0.75rem', fontWeight: 600 }}>
              {activeEngine === 'sherpa' ? `⚡ Sherpa: ${getSherpaLabel()}` : `🎙️ Whisper: ${sttModel || health.whisper_model} (${health.device.toUpperCase()})`}
            </span>
          </button>
        )}


        {/* Theme Toggle (Light / Dark) */}
        <button
          className="btn-icon"
          onClick={toggleTheme}
          title={theme === 'light' ? 'Switch to Dark Mode' : 'Switch to Light Mode'}
          id="btn-theme-toggle"
          aria-label="Toggle Theme"
        >
          {theme === 'light' ? (
            <Moon size={18} color="#475569" />
          ) : (
            <Sun size={18} color="#fbbf24" />
          )}
        </button>

        <button
          className="btn-icon"
          onClick={onOpenSettings}
          title="Server & Model Settings"
          id="btn-settings"
        >
          <Settings size={18} />
        </button>
      </div>
    </header>
  );
}
