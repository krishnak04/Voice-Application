import React, { useState, useEffect } from 'react';
import { Mic2, Settings, Cpu, CheckCircle2, XCircle, Sun, Moon } from 'lucide-react';

export default function Header({ health, onOpenSettings }) {
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

        {isOnline && (
          <div className={`badge ${isCuda ? 'badge-blue' : 'badge-amber'}`}>
            <Cpu size={13} />
            <span>{health.whisper_model} ({health.device.toUpperCase()})</span>
          </div>
        )}

        {isOnline && health?.sherpa_available && (
          <div
            className="badge"
            style={{
              background: 'rgba(168, 85, 247, 0.12)',
              color: '#9333ea',
              border: '1px solid rgba(168, 85, 247, 0.3)',
              display: 'flex',
              alignItems: 'center',
              gap: '0.35rem'
            }}
            title="Sherpa-ONNX Zipformer Streaming ASR Engine Ready"
          >
            <span style={{ fontSize: '0.75rem', fontWeight: 600 }}>⚡ Sherpa Full-Duplex</span>
          </div>
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
