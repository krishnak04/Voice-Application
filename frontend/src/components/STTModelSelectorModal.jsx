import React, { useState, useEffect, useCallback } from 'react';
import {
  X,
  Zap,
  Download,
  CheckCircle2,
  AlertCircle,
  Loader2,
  HardDrive,
  Cpu,
  Layers,
  Activity,
  ChevronDown,
  ChevronUp,
  Info,
  Mic,
  Globe
} from 'lucide-react';
import {
  getSttModels,
  selectSttModel,
  downloadSherpaModel,
  getSherpaBenchmarks
} from '../services/api';

export default function STTModelSelectorModal({ isOpen, onClose, onModelSelected, initialEngine = 'sherpa' }) {
  const [selectedEngine, setSelectedEngine] = useState(() => localStorage.getItem('stt_engine') || initialEngine);
  const [sttData, setSttData] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [actionInProgress, setActionInProgress] = useState(null); // { type: 'select' | 'download', engine: string, id: string }
  const [statusMessage, setStatusMessage] = useState(null); // { type: 'success' | 'error', text: string }
  const [showBenchmarks, setShowBenchmarks] = useState(false);
  const [benchmarks, setBenchmarks] = useState(null);
  const [loadingBenchmarks, setLoadingBenchmarks] = useState(false);

  const fetchModels = useCallback(async () => {
    setIsLoading(true);
    try {
      const data = await getSttModels();
      if (data) {
        setSttData(data);
      }
    } catch (err) {
      setStatusMessage({ type: 'error', text: err.message || 'Failed to load STT models' });
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    if (isOpen) {
      fetchModels();
      setStatusMessage(null);
      const savedEngine = localStorage.getItem('stt_engine');
      if (savedEngine) setSelectedEngine(savedEngine);
    }
  }, [isOpen, fetchModels]);

  const handleEngineSwitch = (engine) => {
    setSelectedEngine(engine);
    localStorage.setItem('stt_engine', engine);
    setStatusMessage(null);
  };

  const handleSelectModel = async (engine, modelId) => {
    setActionInProgress({ type: 'select', engine, id: modelId });
    setStatusMessage(null);
    try {
      const res = await selectSttModel(engine, modelId);
      setStatusMessage({ type: 'success', text: res.message || `Switched to ${modelId}` });
      localStorage.setItem('stt_engine', engine);
      if (engine === 'whisper') {
        localStorage.setItem('whisper_model', modelId);
      } else {
        localStorage.setItem('sherpa_model', modelId);
      }
      await fetchModels();
      if (onModelSelected) {
        onModelSelected({ engine, modelId });
      }
    } catch (err) {
      setStatusMessage({ type: 'error', text: err.message || 'Failed to switch model' });
    } finally {
      setActionInProgress(null);
    }
  };

  const handleDownloadSherpa = async (modelId) => {
    setActionInProgress({ type: 'download', engine: 'sherpa', id: modelId });
    setStatusMessage(null);
    try {
      const res = await downloadSherpaModel(modelId);
      setStatusMessage({ type: 'success', text: res.message || `Downloaded model ${modelId}` });
      await fetchModels();
    } catch (err) {
      setStatusMessage({ type: 'error', text: err.message || 'Failed to download model' });
    } finally {
      setActionInProgress(null);
    }
  };

  const handleToggleBenchmarks = async () => {
    const nextState = !showBenchmarks;
    setShowBenchmarks(nextState);
    if (nextState && !benchmarks) {
      setLoadingBenchmarks(true);
      try {
        const res = await getSherpaBenchmarks();
        if (res && res.benchmarks) {
          setBenchmarks(res.benchmarks);
        }
      } catch (err) {
        console.error('Benchmark error:', err);
      } finally {
        setLoadingBenchmarks(false);
      }
    }
  };

  if (!isOpen) return null;

  const whisper = sttData?.whisper || {};
  const sherpa = sttData?.sherpa || {};

  const activeWhisperModel = whisper.active_model || 'base';
  const activeSherpaModel = sherpa.active_model || 'sherpa-onnx-streaming-zipformer-en-2023-06-26';

  const currentModels = selectedEngine === 'whisper' ? (whisper.models || []) : (sherpa.models || []);

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        background: 'rgba(15, 23, 42, 0.65)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 120,
        backdropFilter: 'blur(8px)',
        padding: '1rem'
      }}
    >
      <div
        className="glass-panel"
        style={{
          width: '100%',
          maxWidth: '820px',
          maxHeight: '92vh',
          display: 'flex',
          flexDirection: 'column',
          background: 'var(--bg-card-solid)',
          boxShadow: 'var(--shadow-dropdown)',
          borderRadius: '16px',
          overflow: 'hidden'
        }}
      >
        {/* Header */}
        <div
          style={{
            padding: '1.25rem 1.5rem',
            borderBottom: '1px solid var(--border)',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center'
          }}
        >
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
              <div
                style={{
                  width: '32px',
                  height: '32px',
                  borderRadius: '8px',
                  background: 'linear-gradient(135deg, #0284c7, #9333ea)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: '#ffffff'
                }}
              >
                <Mic size={18} />
              </div>
              <div>
                <h3 style={{ fontSize: '1.15rem', fontWeight: 700, margin: 0, color: 'var(--text-main)' }}>
                  Speech-to-Text (STT) Engine & Model Selector
                </h3>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                  Support for both Faster-Whisper and Sherpa-ONNX Next-Gen Kaldi
                </span>
              </div>
            </div>
          </div>
          <button
            type="button"
            className="btn-icon"
            onClick={onClose}
            aria-label="Close dialog"
            style={{ borderRadius: '50%' }}
          >
            <X size={18} />
          </button>
        </div>

        {/* Engine Tabs */}
        <div
          style={{
            display: 'flex',
            borderBottom: '1px solid var(--border)',
            background: 'var(--bg-subtle)'
          }}
        >
          <button
            type="button"
            onClick={() => handleEngineSwitch('sherpa')}
            style={{
              flex: 1,
              padding: '0.85rem 1rem',
              border: 'none',
              background: selectedEngine === 'sherpa' ? 'var(--bg-card-solid)' : 'transparent',
              color: selectedEngine === 'sherpa' ? '#9333ea' : 'var(--text-muted)',
              fontWeight: 700,
              fontSize: '0.88rem',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '0.5rem',
              borderBottom: selectedEngine === 'sherpa' ? '2px solid #9333ea' : '2px solid transparent',
              transition: 'all 0.15s ease'
            }}
          >
            <Zap size={16} />
            <span>Sherpa-ONNX Zipformer (Streaming & Offline)</span>
            {sherpa.active_model && (
              <span style={{ fontSize: '0.68rem', padding: '0.15rem 0.45rem', borderRadius: '4px', background: 'rgba(168, 85, 247, 0.15)', color: '#9333ea' }}>
                Active
              </span>
            )}
          </button>

          <button
            type="button"
            onClick={() => handleEngineSwitch('whisper')}
            style={{
              flex: 1,
              padding: '0.85rem 1rem',
              border: 'none',
              background: selectedEngine === 'whisper' ? 'var(--bg-card-solid)' : 'transparent',
              color: selectedEngine === 'whisper' ? 'var(--accent-call)' : 'var(--text-muted)',
              fontWeight: 700,
              fontSize: '0.88rem',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '0.5rem',
              borderBottom: selectedEngine === 'whisper' ? '2px solid var(--accent-call)' : '2px solid transparent',
              transition: 'all 0.15s ease'
            }}
          >
            <Cpu size={16} />
            <span>Faster-Whisper (Multi-lingual & English)</span>
            {whisper.active_model && (
              <span style={{ fontSize: '0.68rem', padding: '0.15rem 0.45rem', borderRadius: '4px', background: 'rgba(2, 132, 199, 0.15)', color: 'var(--accent-call)' }}>
                {whisper.active_model}
              </span>
            )}
          </button>
        </div>

        {/* Content Body */}
        <div style={{ padding: '1.25rem 1.5rem', overflowY: 'auto', flex: 1, display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          {/* Active Model Summary Banner */}
          <div
            style={{
              padding: '0.85rem 1.1rem',
              borderRadius: '10px',
              background: selectedEngine === 'sherpa' ? 'rgba(168, 85, 247, 0.08)' : 'rgba(2, 132, 199, 0.08)',
              border: `1px solid ${selectedEngine === 'sherpa' ? 'rgba(168, 85, 247, 0.25)' : 'rgba(2, 132, 199, 0.25)'}`,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              flexWrap: 'wrap',
              gap: '0.6rem'
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
              <div
                style={{
                  width: '10px',
                  height: '10px',
                  borderRadius: '50%',
                  background: '#10b981',
                  boxShadow: '0 0 8px #10b981'
                }}
              />
              <div>
                <div style={{ fontSize: '0.72rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                  Active {selectedEngine === 'sherpa' ? 'Sherpa-ONNX' : 'Faster-Whisper'} Model
                </div>
                <div style={{ fontSize: '0.92rem', fontWeight: 700, color: selectedEngine === 'sherpa' ? '#9333ea' : 'var(--accent-call)' }}>
                  {selectedEngine === 'sherpa'
                    ? (sherpa.models?.find(m => m.id === activeSherpaModel)?.name || activeSherpaModel)
                    : (whisper.models?.find(m => m.id === activeWhisperModel)?.name || activeWhisperModel)}
                </div>
              </div>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap' }}>
              <span
                style={{
                  fontSize: '0.72rem',
                  fontWeight: 600,
                  padding: '0.2rem 0.55rem',
                  borderRadius: '6px',
                  background: selectedEngine === 'sherpa' && sherpa.active_type === 'streaming' ? 'rgba(5, 150, 105, 0.15)' : 'rgba(124, 58, 237, 0.15)',
                  color: selectedEngine === 'sherpa' && sherpa.active_type === 'streaming' ? '#059669' : '#7c3aed',
                  fontFamily: 'var(--font-mono)'
                }}
              >
                {selectedEngine === 'sherpa'
                  ? (sherpa.active_type === 'streaming' ? '⚡ Causal Streaming' : '📦 Offline Transducer')
                  : '📦 Offline Turn-Based'}
              </span>

              <span
                style={{
                  fontSize: '0.72rem',
                  fontWeight: 600,
                  padding: '0.2rem 0.55rem',
                  borderRadius: '6px',
                  background: 'var(--bg-subtle)',
                  color: 'var(--text-muted)',
                  fontFamily: 'var(--font-mono)'
                }}
              >
                Device: {selectedEngine === 'whisper' ? `${whisper.device || 'cpu'} (${whisper.compute_type || 'int8'})` : 'CPU (int8/fp32)'}
              </span>
            </div>
          </div>

          {/* Status Message */}
          {statusMessage && (
            <div
              style={{
                padding: '0.75rem 1rem',
                borderRadius: '8px',
                fontSize: '0.82rem',
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem',
                background: statusMessage.type === 'success' ? 'rgba(16, 185, 129, 0.12)' : 'rgba(239, 68, 68, 0.12)',
                border: `1px solid ${statusMessage.type === 'success' ? '#10b981' : '#ef4444'}`,
                color: statusMessage.type === 'success' ? '#059669' : '#dc2626'
              }}
            >
              {statusMessage.type === 'success' ? <CheckCircle2 size={16} /> : <AlertCircle size={16} />}
              <span>{statusMessage.text}</span>
            </div>
          )}

          {/* Models List */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
            {isLoading ? (
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '3rem', color: 'var(--text-muted)', gap: '0.6rem' }}>
                <Loader2 size={20} className="spin" />
                <span>Checking verified models...</span>
              </div>
            ) : (
              currentModels.map((m) => {
                const isActive = selectedEngine === 'sherpa'
                  ? (m.id === activeSherpaModel)
                  : (m.id === activeWhisperModel);

                const isSelectedBusy = actionInProgress && actionInProgress.id === m.id;
                const isDownloaded = m.is_downloaded || m.downloaded;

                return (
                  <div
                    key={m.id}
                    style={{
                      padding: '1rem',
                      borderRadius: '12px',
                      background: isActive
                        ? (selectedEngine === 'sherpa' ? 'rgba(168, 85, 247, 0.08)' : 'var(--accent-call-bg)')
                        : 'var(--bg-card)',
                      border: isActive
                        ? (selectedEngine === 'sherpa' ? '1.5px solid #9333ea' : '1.5px solid var(--accent-call)')
                        : '1px solid var(--border)',
                      display: 'flex',
                      flexDirection: 'column',
                      gap: '0.65rem',
                      transition: 'all 0.2s ease',
                      boxShadow: isActive ? '0 4px 16px rgba(0, 0, 0, 0.06)' : 'var(--shadow-xs)'
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '0.8rem', flexWrap: 'wrap' }}>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.2rem', flex: 1, minWidth: '220px' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap' }}>
                          <span style={{ fontWeight: 700, fontSize: '0.94rem', color: 'var(--text-main)' }}>
                            {m.name}
                          </span>

                          <span
                            style={{
                              fontSize: '0.68rem',
                              fontWeight: 700,
                              padding: '0.15rem 0.45rem',
                              borderRadius: '4px',
                              background: 'var(--bg-subtle)',
                              color: 'var(--text-muted)'
                            }}
                          >
                            {selectedEngine === 'whisper'
                              ? (m.is_english_only ? '🇺🇸 English Only' : '🌐 Multilingual (Indic & Global)')
                              : `🇺🇸 ${m.language} (${m.lang_code})`}
                          </span>
                        </div>

                        <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                          {m.description}
                        </div>
                      </div>

                      {/* Action Button */}
                      <div>
                        {isActive ? (
                          <div
                            style={{
                              display: 'flex',
                              alignItems: 'center',
                              gap: '0.35rem',
                              fontSize: '0.78rem',
                              fontWeight: 700,
                              color: '#10b981',
                              background: 'rgba(16, 185, 129, 0.1)',
                              border: '1px solid rgba(16, 185, 129, 0.3)',
                              padding: '0.4rem 0.75rem',
                              borderRadius: '6px'
                            }}
                          >
                            <CheckCircle2 size={14} /> Active
                          </div>
                        ) : isDownloaded ? (
                          <button
                            type="button"
                            onClick={() => handleSelectModel(selectedEngine, m.id)}
                            disabled={Boolean(actionInProgress)}
                            style={{
                              padding: '0.45rem 0.9rem',
                              borderRadius: '6px',
                              border: selectedEngine === 'sherpa' ? '1px solid #9333ea' : '1px solid var(--accent-call)',
                              background: selectedEngine === 'sherpa' ? '#9333ea' : 'var(--accent-call)',
                              color: '#ffffff',
                              fontSize: '0.8rem',
                              fontWeight: 600,
                              cursor: actionInProgress ? 'not-allowed' : 'pointer',
                              display: 'flex',
                              alignItems: 'center',
                              gap: '0.4rem',
                              transition: 'all 0.15s ease'
                            }}
                          >
                            {isSelectedBusy && actionInProgress.type === 'select' ? (
                              <>
                                <Loader2 size={13} className="spin" /> Switching...
                              </>
                            ) : (
                              'Select Model'
                            )}
                          </button>
                        ) : (
                          <button
                            type="button"
                            onClick={() => handleDownloadSherpa(m.id)}
                            disabled={Boolean(actionInProgress)}
                            style={{
                              padding: '0.45rem 0.9rem',
                              borderRadius: '6px',
                              border: '1px solid var(--border)',
                              background: 'var(--bg-subtle)',
                              color: 'var(--text-main)',
                              fontSize: '0.8rem',
                              fontWeight: 600,
                              cursor: actionInProgress ? 'not-allowed' : 'pointer',
                              display: 'flex',
                              alignItems: 'center',
                              gap: '0.4rem',
                              transition: 'all 0.15s ease'
                            }}
                          >
                            {isSelectedBusy && actionInProgress.type === 'download' ? (
                              <>
                                <Loader2 size={13} className="spin" /> Downloading...
                              </>
                            ) : (
                              <>
                                <Download size={13} /> Download
                              </>
                            )}
                          </button>
                        )}
                      </div>
                    </div>

                    {/* Metadata Badges */}
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', flexWrap: 'wrap', paddingTop: '0.35rem', borderTop: '1px solid var(--border-light)' }}>
                      <span
                        style={{
                          fontSize: '0.72rem',
                          fontWeight: 600,
                          padding: '0.2rem 0.5rem',
                          borderRadius: '4px',
                          background: m.type === 'streaming' ? 'rgba(5, 150, 105, 0.12)' : 'rgba(124, 58, 237, 0.12)',
                          color: m.type === 'streaming' ? '#059669' : '#7c3aed',
                          fontFamily: 'var(--font-mono)'
                        }}
                      >
                        {m.type === 'streaming' ? '⚡ Streaming Engine' : '📦 Offline Transducer'}
                      </span>

                      <span
                        style={{
                          fontSize: '0.72rem',
                          fontWeight: 600,
                          padding: '0.2rem 0.5rem',
                          borderRadius: '4px',
                          background: 'var(--bg-subtle)',
                          color: 'var(--text-main)',
                          fontFamily: 'var(--font-mono)'
                        }}
                      >
                        Size: {m.category || 'Standard'} ({m.params})
                      </span>

                      <span
                        style={{
                          fontSize: '0.72rem',
                          display: 'flex',
                          alignItems: 'center',
                          gap: '0.25rem',
                          color: isDownloaded ? '#059669' : '#d97706',
                          fontWeight: 600
                        }}
                      >
                        <HardDrive size={12} />
                        {selectedEngine === 'whisper'
                          ? (m.is_loaded ? 'Loaded in RAM' : 'Available')
                          : (isDownloaded ? `Downloaded (${m.disk_size_mb} MB)` : 'Not Downloaded')}
                      </span>
                    </div>
                  </div>
                );
              })
            )}
          </div>

          {/* Benchmark Collapsible Section (for Sherpa Models) */}
          {selectedEngine === 'sherpa' && (
            <div
              style={{
                marginTop: '0.75rem',
                border: '1px solid var(--border)',
                borderRadius: '12px',
                overflow: 'hidden',
                background: 'var(--bg-card)'
              }}
            >
              <button
                type="button"
                onClick={handleToggleBenchmarks}
                style={{
                  width: '100%',
                  padding: '0.85rem 1.1rem',
                  border: 'none',
                  background: 'var(--bg-subtle)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  cursor: 'pointer',
                  color: 'var(--text-main)',
                  fontWeight: 600,
                  fontSize: '0.85rem'
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <Activity size={16} color="var(--accent-call)" />
                  <span>Empirical Hardware Benchmark Comparisons (Measured Results)</span>
                </div>
                {showBenchmarks ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
              </button>

              {showBenchmarks && (
                <div style={{ padding: '1rem', borderTop: '1px solid var(--border)' }}>
                  {loadingBenchmarks ? (
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '1.5rem', gap: '0.5rem', color: 'var(--text-muted)' }}>
                      <Loader2 size={16} className="spin" />
                      <span>Reading benchmark measurements...</span>
                    </div>
                  ) : benchmarks && benchmarks.length > 0 ? (
                    <div style={{ overflowX: 'auto' }}>
                      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.78rem', fontFamily: 'var(--font-mono)' }}>
                        <thead>
                          <tr style={{ background: 'var(--bg-subtle)', borderBottom: '1px solid var(--border)' }}>
                            <th style={{ padding: '0.5rem 0.75rem', textAlign: 'left' }}>Model</th>
                            <th style={{ padding: '0.5rem 0.75rem', textAlign: 'left' }}>Type</th>
                            <th style={{ padding: '0.5rem 0.75rem', textAlign: 'right' }}>Load Time</th>
                            <th style={{ padding: '0.5rem 0.75rem', textAlign: 'right' }}>RAM Footprint</th>
                            <th style={{ padding: '0.5rem 0.75rem', textAlign: 'right' }}>Avg RTF</th>
                            <th style={{ padding: '0.5rem 0.75rem', textAlign: 'right' }}>1st Partial</th>
                            <th style={{ padding: '0.5rem 0.75rem', textAlign: 'right' }}>WER</th>
                          </tr>
                        </thead>
                        <tbody>
                          {benchmarks.map((b) => (
                            <tr key={b.model_id} style={{ borderBottom: '1px solid var(--border-light)' }}>
                              <td style={{ padding: '0.55rem 0.75rem', fontWeight: 600, color: 'var(--text-main)' }}>
                                {b.name}
                              </td>
                              <td style={{ padding: '0.55rem 0.75rem' }}>
                                <span style={{ color: b.type === 'streaming' ? '#059669' : '#7c3aed' }}>
                                  {b.type}
                                </span>
                              </td>
                              <td style={{ padding: '0.55rem 0.75rem', textAlign: 'right' }}>
                                {b.load_time_sec}s
                              </td>
                              <td style={{ padding: '0.55rem 0.75rem', textAlign: 'right' }}>
                                {b.ram_delta_mb} MB
                              </td>
                              <td style={{ padding: '0.55rem 0.75rem', textAlign: 'right' }}>
                                {b.avg_rtf}
                              </td>
                              <td style={{ padding: '0.55rem 0.75rem', textAlign: 'right' }}>
                                {b.first_partial_ms ? `${b.first_partial_ms}ms` : 'N/A (Batch)'}
                              </td>
                              <td style={{ padding: '0.55rem 0.75rem', textAlign: 'right', fontWeight: 700, color: b.avg_wer < 3 ? '#059669' : '#d97706' }}>
                                {b.avg_wer}%
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  ) : (
                    <div style={{ color: 'var(--text-muted)', fontSize: '0.8rem', textAlign: 'center', padding: '1rem' }}>
                      No benchmark cache found. Benchmarks will compute on first test.
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
