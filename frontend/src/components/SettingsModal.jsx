import React, { useState, useEffect } from 'react';
import { X, Server, Key, Cpu, Save } from 'lucide-react';
import { getServerUrl, setServerUrl, getDefaultServerUrl } from '../services/api';

export default function SettingsModal({ isOpen, onClose, onSaveSettings }) {
  const defaultUrl = getDefaultServerUrl();
  const [url, setUrl] = useState('');
  const [apiKey, setApiKey] = useState('');
  const [model, setModel] = useState('openai/gpt-oss-20b');
  const [whisperModel, setWhisperModel] = useState('base');

  useEffect(() => {
    if (isOpen) {
      const saved = localStorage.getItem('voice_benchmark_server_url');
      if (saved && saved !== 'http://localhost:8000' && saved !== 'http://127.0.0.1:8000') {
        setUrl(saved.includes(':8000') ? saved.replace(':8000', ':8024') : saved);
      } else {
        setUrl('');
      }
      setApiKey(localStorage.getItem('groq_api_key') || '');
      setModel(localStorage.getItem('groq_model') || 'openai/gpt-oss-20b');
      setWhisperModel(localStorage.getItem('whisper_model') || 'base');
    }
  }, [isOpen]);

  if (!isOpen) return null;

  const handleSave = (e) => {
    e.preventDefault();
    setServerUrl(url);
    localStorage.setItem('groq_api_key', apiKey.trim());
    localStorage.setItem('groq_model', model.trim());
    localStorage.setItem('whisper_model', whisperModel.trim());
    onSaveSettings();
    onClose();
  };

  return (
    <div style={{
      position: 'fixed', inset: 0, background: 'rgba(15, 23, 42, 0.45)',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      zIndex: 100, backdropFilter: 'blur(6px)', padding: '1rem'
    }}>
      <div className="glass-panel" style={{ width: '100%', maxWidth: '440px', background: 'var(--bg-card-solid)', boxShadow: 'var(--shadow-dropdown)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem' }}>
          <h3 style={{ fontSize: '1.1rem', fontWeight: 600, color: 'var(--text-main)' }}>Connection & Model Settings</h3>
          <button className="btn-icon" onClick={onClose}><X size={18} /></button>
        </div>

        <form onSubmit={handleSave} style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          <div>
            <label style={{ fontSize: '0.8rem', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '0.4rem', marginBottom: '0.4rem' }}>
              <Server size={14} /> FastAPI Dev Server URL
            </label>
            <input
              type="text"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              placeholder={defaultUrl}
              style={{
                width: '100%', padding: '0.65rem 0.8rem', borderRadius: '8px',
                background: 'var(--bg-input)', border: '1px solid var(--border)',
                color: 'var(--text-main)', fontFamily: 'var(--font-mono)', fontSize: '0.85rem'
              }}
            />
            <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '0.2rem', display: 'block' }}>
              Active: <strong style={{ color: 'var(--accent-call)' }}>{url || defaultUrl}</strong> (Leave blank to use host in browser URL)
            </span>
          </div>

          <div>
            <label style={{ fontSize: '0.8rem', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '0.4rem', marginBottom: '0.4rem' }}>
              <Key size={14} /> Groq API Key (Optional)
            </label>
            <input
              type="password"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder="gsk_..."
              style={{
                width: '100%', padding: '0.65rem 0.8rem', borderRadius: '8px',
                background: 'var(--bg-input)', border: '1px solid var(--border)',
                color: 'var(--text-main)', fontFamily: 'var(--font-mono)', fontSize: '0.85rem'
              }}
            />
          </div>

          <div>
            <label style={{ fontSize: '0.8rem', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '0.4rem', marginBottom: '0.4rem' }}>
              <Cpu size={14} /> Groq Model Name
            </label>
            <input
              type="text"
              value={model}
              onChange={(e) => setModel(e.target.value)}
              placeholder="openai/gpt-oss-20b"
              style={{
                width: '100%', padding: '0.65rem 0.8rem', borderRadius: '8px',
                background: 'var(--bg-input)', border: '1px solid var(--border)',
                color: 'var(--text-main)', fontFamily: 'var(--font-mono)', fontSize: '0.85rem'
              }}
            />
            <div style={{ display: 'flex', gap: '0.4rem', marginTop: '0.35rem', flexWrap: 'wrap' }}>
              <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>Active fast:</span>
              <button
                type="button"
                onClick={() => setModel('openai/gpt-oss-20b')}
                style={{
                  background: model === 'openai/gpt-oss-20b' ? 'var(--accent-call-bg)' : 'var(--bg-subtle)',
                  border: model === 'openai/gpt-oss-20b' ? '1px solid var(--accent-call-border)' : '1px solid var(--border)',
                  borderRadius: '4px',
                  color: model === 'openai/gpt-oss-20b' ? 'var(--accent-call)' : 'var(--text-muted)',
                  fontSize: '0.7rem',
                  padding: '0.1rem 0.4rem',
                  cursor: 'pointer'
                }}
              >
                openai/gpt-oss-20b (Recommended)
              </button>
              <button
                type="button"
                onClick={() => setModel('qwen/qwen3.8-27b')}
                style={{
                  background: model === 'qwen/qwen3.8-27b' ? 'var(--accent-call-bg)' : 'var(--bg-subtle)',
                  border: model === 'qwen/qwen3.8-27b' ? '1px solid var(--accent-call-border)' : '1px solid var(--border)',
                  borderRadius: '4px',
                  color: model === 'qwen/qwen3.8-27b' ? 'var(--accent-call)' : 'var(--text-muted)',
                  fontSize: '0.7rem',
                  padding: '0.1rem 0.4rem',
                  cursor: 'pointer'
                }}
              >
                qwen/qwen3.8-27b
              </button>
            </div>
          </div>

          <div>
            <label style={{ fontSize: '0.8rem', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '0.4rem', marginBottom: '0.4rem' }}>
              <Cpu size={14} /> Whisper Model
            </label>
            <select
              value={whisperModel}
              onChange={(e) => setWhisperModel(e.target.value)}
              style={{
                width: '100%', padding: '0.65rem 0.8rem', borderRadius: '8px',
                background: 'var(--bg-input)', border: '1px solid var(--border)',
                color: 'var(--text-main)', fontFamily: 'var(--font-mono)', fontSize: '0.85rem'
              }}
            >
              <option value="base">base (Installed on Dev PC - Fast Multilingual)</option>
              <option value="small">small (Multilingual higher accuracy)</option>
              <option value="medium">medium (Production Indic quality)</option>
              <option value="large-v3-turbo">large-v3-turbo (State of the art)</option>
              <option value="tiny">tiny (Ultra fast, lowest memory)</option>
              <option value="base.en">base.en (English-only ultra fast)</option>
            </select>
          </div>

          <button
            type="submit"
            style={{
              marginTop: '0.5rem', padding: '0.75rem', borderRadius: '8px', border: 'none',
              background: 'linear-gradient(135deg, #0284c7, #2563eb)', color: '#fff',
              fontWeight: 600, display: 'flex', alignItems: 'center', justifyContent: 'center',
              gap: '0.5rem', cursor: 'pointer'
            }}
          >
            <Save size={16} /> Save Settings
          </button>
        </form>
      </div>
    </div>
  );
}
