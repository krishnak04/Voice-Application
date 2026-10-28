import React, { useState, useEffect, useCallback, useRef } from 'react';
import Header from './components/Header';
import AudioRecorder from './components/AudioRecorder';
import VoiceCallSimulator from './components/VoiceCallSimulator';
import LatencyDashboard from './components/LatencyDashboard';
import ChatTranscript from './components/ChatTranscript';
import SettingsModal from './components/SettingsModal';
import SherpaModelSelector from './components/SherpaModelSelector';
import { useAudioRecorder } from './hooks/useAudioRecorder';
import { checkServerHealth, sendAudioForBenchmark } from './services/api';
import { PhoneCall, Activity, Mic, Volume2, ChevronDown, Check, Sparkles } from 'lucide-react';

export default function App() {
  const [health, setHealth] = useState(null);
  const [activeTab, setActiveTab] = useState('call'); // 'call' | 'benchmark'
  const [simulatorMode, setSimulatorMode] = useState(() => localStorage.getItem('sim_active_mode') || 'call'); // 'call' | 'stt' | 'tts'
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  const dropdownRef = useRef(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [processError, setProcessError] = useState(null);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [isSherpaModalOpen, setIsSherpaModalOpen] = useState(false);
  const [result, setResult] = useState(null);
  const [language, setLanguage] = useState(() => localStorage.getItem('whisper_language') || 'auto');
  const [model, setModel] = useState(() => localStorage.getItem('whisper_model') || 'sherpa-onnx-streaming-zipformer-en-2023-06-26');
  const [engine, setEngine] = useState(() => localStorage.getItem('stt_engine') || 'sherpa');

  const handleEngineChange = (eng) => {
    setEngine(eng);
    localStorage.setItem('stt_engine', eng);
  };

  // Close dropdown on outside click
  useEffect(() => {
    const handleClickOutside = (e) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target)) {
        setIsDropdownOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const {
    isRecording,
    recordingDuration,
    error: micError,
    analyser,
    startRecording,
    stopRecording,
  } = useAudioRecorder();

  const handleLanguageChange = (lang) => {
    setLanguage(lang);
    localStorage.setItem('whisper_language', lang);
  };

  const handleModelChange = (m) => {
    setModel(m);
    localStorage.setItem('whisper_model', m);
  };

  const handleSimulatorModeChange = (newMode) => {
    setSimulatorMode(newMode);
    localStorage.setItem('sim_active_mode', newMode);
  };

  const fetchHealth = useCallback(async () => {
    const data = await checkServerHealth();
    setHealth(data);
    const savedModel = localStorage.getItem('whisper_model');
    if (savedModel) setModel(savedModel);
  }, []);

  useEffect(() => {
    fetchHealth();
    const interval = setInterval(fetchHealth, 15000);
    return () => clearInterval(interval);
  }, [fetchHealth]);

  const handleStopRecording = async () => {
    const audioBlob = await stopRecording();
    if (!audioBlob || audioBlob.size === 0) return;

    setIsProcessing(true);
    setProcessError(null);

    const groqApiKey = localStorage.getItem('groq_api_key') || undefined;
    const groqModel = localStorage.getItem('groq_model') || 'openai/gpt-oss-20b';

    try {
      const data = await sendAudioForBenchmark(audioBlob, {
        groqApiKey,
        groqModel,
        language,
        model,
        engine,
        stt_engine: engine
      });
      setResult(data);
    } catch (err) {
      setProcessError(err.message || 'Failed to process audio');
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="app-container">
      <Header
        health={health}
        sttEngine={engine}
        sttModel={model}
        onOpenSettings={() => setIsSettingsOpen(true)}
        onOpenSherpa={() => setIsSherpaModalOpen(true)}
      />

      {/* Mode Navigation Tabs */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          gap: '0.85rem',
          margin: '0.65rem 0 1.35rem',
          position: 'relative',
          zIndex: 40
        }}
      >
        {/* Sleek Custom Dropdown Pill (No Duplicate Buttons) */}
        <div ref={dropdownRef} style={{ position: 'relative' }}>
          <button
            type="button"
            onClick={() => {
              if (activeTab !== 'call') {
                setActiveTab('call');
                setIsDropdownOpen(true);
              } else {
                setIsDropdownOpen(!isDropdownOpen);
              }
            }}
            title="Click to switch modes between Phone Call Simulator, Voice to Text, and Text to Voice"
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.55rem',
              padding: '0.6rem 1.4rem',
              borderRadius: '999px',
              border:
                activeTab === 'call'
                  ? simulatorMode === 'stt'
                    ? '1px solid var(--accent-stt-border)'
                    : simulatorMode === 'tts'
                    ? '1px solid var(--accent-tts-border)'
                    : '1px solid var(--accent-call-border)'
                  : '1px solid var(--border)',
              background:
                activeTab === 'call'
                  ? simulatorMode === 'stt'
                    ? 'var(--accent-stt-bg)'
                    : simulatorMode === 'tts'
                    ? 'var(--accent-tts-bg)'
                    : 'var(--accent-call-bg)'
                  : 'var(--bg-card)',
              color:
                activeTab === 'call'
                  ? simulatorMode === 'stt'
                    ? 'var(--accent-stt)'
                    : simulatorMode === 'tts'
                    ? 'var(--accent-tts)'
                    : 'var(--accent-call)'
                  : 'var(--text-muted)',
              boxShadow:
                activeTab === 'call'
                  ? simulatorMode === 'stt'
                    ? '0 2px 14px var(--accent-stt-glow)'
                    : simulatorMode === 'tts'
                    ? '0 2px 14px var(--accent-tts-glow)'
                    : '0 2px 14px var(--accent-call-glow)'
                  : 'var(--shadow-xs)',
              fontWeight: 600,
              fontSize: '0.9rem',
              cursor: 'pointer',
              transition: 'all 0.2s cubic-bezier(0.4, 0, 0.2, 1)',
              outline: 'none',
              userSelect: 'none'
            }}
          >
            {simulatorMode === 'stt' ? (
              <Mic size={16} color="var(--accent-stt)" />
            ) : simulatorMode === 'tts' ? (
              <Volume2 size={16} color="var(--accent-tts)" />
            ) : (
              <PhoneCall size={16} color="var(--accent-call)" />
            )}

            <span>
              {simulatorMode === 'stt'
                ? 'Voice to Text (STT)'
                : simulatorMode === 'tts'
                ? 'Text to Voice (TTS)'
                : 'Phone Call Simulator'}
            </span>

            <ChevronDown
              size={14}
              style={{
                transform: isDropdownOpen ? 'rotate(180deg)' : 'rotate(0deg)',
                transition: 'transform 0.2s ease',
                opacity: 0.85
              }}
            />
          </button>

          {/* Floating Glassmorphic Dropdown Menu */}
          {isDropdownOpen && (
            <div
              style={{
                position: 'absolute',
                top: 'calc(100% + 8px)',
                left: '50%',
                transform: 'translateX(-50%)',
                minWidth: '290px',
                background: 'var(--bg-card-solid)',
                backdropFilter: 'blur(20px)',
                border: '1px solid var(--border)',
                borderRadius: '16px',
                boxShadow: 'var(--shadow-dropdown)',
                padding: '0.5rem',
                display: 'flex',
                flexDirection: 'column',
                gap: '0.3rem',
                zIndex: 100
              }}
            >
              <div
                style={{
                  padding: '0.35rem 0.65rem 0.2rem',
                  fontSize: '0.68rem',
                  fontWeight: 700,
                  textTransform: 'uppercase',
                  letterSpacing: '0.8px',
                  color: 'var(--text-muted)'
                }}
              >
                Select Mode
              </div>

              {/* Option 1: Phone Call Simulator */}
              <button
                type="button"
                onClick={() => {
                  setActiveTab('call');
                  handleSimulatorModeChange('call');
                  setIsDropdownOpen(false);
                }}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  padding: '0.6rem 0.85rem',
                  borderRadius: '10px',
                  border:
                    simulatorMode === 'call' && activeTab === 'call'
                      ? '1px solid var(--accent-call-border)'
                      : '1px solid transparent',
                  background:
                    simulatorMode === 'call' && activeTab === 'call'
                      ? 'var(--accent-call-bg)'
                      : 'transparent',
                  color: simulatorMode === 'call' && activeTab === 'call' ? 'var(--accent-call)' : 'var(--text-main)',
                  cursor: 'pointer',
                  textAlign: 'left',
                  transition: 'all 0.15s ease'
                }}
                onMouseEnter={(e) => {
                  if (simulatorMode !== 'call' || activeTab !== 'call') {
                    e.currentTarget.style.background = 'var(--bg-subtle)';
                  }
                }}
                onMouseLeave={(e) => {
                  if (simulatorMode !== 'call' || activeTab !== 'call') {
                    e.currentTarget.style.background = 'transparent';
                  }
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                  <div
                    style={{
                      width: '30px',
                      height: '30px',
                      borderRadius: '8px',
                      background: 'var(--accent-call-bg)',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      color: 'var(--accent-call)'
                    }}
                  >
                    <PhoneCall size={16} />
                  </div>
                  <div>
                    <div style={{ fontWeight: 600, fontSize: '0.86rem' }}>Phone Call Simulator</div>
                    <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
                      Full duplex conversational agent
                    </div>
                  </div>
                </div>
                {simulatorMode === 'call' && activeTab === 'call' && <Check size={16} color="var(--accent-call)" />}
              </button>

              {/* Option 2: Voice to Text */}
              <button
                type="button"
                onClick={() => {
                  setActiveTab('call');
                  handleSimulatorModeChange('stt');
                  setIsDropdownOpen(false);
                }}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  padding: '0.6rem 0.85rem',
                  borderRadius: '10px',
                  border:
                    simulatorMode === 'stt' && activeTab === 'call'
                      ? '1px solid var(--accent-stt-border)'
                      : '1px solid transparent',
                  background:
                    simulatorMode === 'stt' && activeTab === 'call'
                      ? 'var(--accent-stt-bg)'
                      : 'transparent',
                  color: simulatorMode === 'stt' && activeTab === 'call' ? 'var(--accent-stt)' : 'var(--text-main)',
                  cursor: 'pointer',
                  textAlign: 'left',
                  transition: 'all 0.15s ease'
                }}
                onMouseEnter={(e) => {
                  if (simulatorMode !== 'stt' || activeTab !== 'call') {
                    e.currentTarget.style.background = 'var(--bg-subtle)';
                  }
                }}
                onMouseLeave={(e) => {
                  if (simulatorMode !== 'stt' || activeTab !== 'call') {
                    e.currentTarget.style.background = 'transparent';
                  }
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                  <div
                    style={{
                      width: '30px',
                      height: '30px',
                      borderRadius: '8px',
                      background: 'var(--accent-stt-bg)',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      color: 'var(--accent-stt)'
                    }}
                  >
                    <Mic size={16} />
                  </div>
                  <div>
                    <div style={{ fontWeight: 600, fontSize: '0.86rem' }}>Voice to Text (STT)</div>
                    <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
                      Live microphone transcriber & upload
                    </div>
                  </div>
                </div>
                {simulatorMode === 'stt' && activeTab === 'call' && <Check size={16} color="var(--accent-stt)" />}
              </button>

              {/* Option 3: Text to Voice */}
              <button
                type="button"
                onClick={() => {
                  setActiveTab('call');
                  handleSimulatorModeChange('tts');
                  setIsDropdownOpen(false);
                }}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  padding: '0.6rem 0.85rem',
                  borderRadius: '10px',
                  border:
                    simulatorMode === 'tts' && activeTab === 'call'
                      ? '1px solid var(--accent-tts-border)'
                      : '1px solid transparent',
                  background:
                    simulatorMode === 'tts' && activeTab === 'call'
                      ? 'var(--accent-tts-bg)'
                      : 'transparent',
                  color: simulatorMode === 'tts' && activeTab === 'call' ? 'var(--accent-tts)' : 'var(--text-main)',
                  cursor: 'pointer',
                  textAlign: 'left',
                  transition: 'all 0.15s ease'
                }}
                onMouseEnter={(e) => {
                  if (simulatorMode !== 'tts' || activeTab !== 'call') {
                    e.currentTarget.style.background = 'var(--bg-subtle)';
                  }
                }}
                onMouseLeave={(e) => {
                  if (simulatorMode !== 'tts' || activeTab !== 'call') {
                    e.currentTarget.style.background = 'transparent';
                  }
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                  <div
                    style={{
                      width: '30px',
                      height: '30px',
                      borderRadius: '8px',
                      background: 'var(--accent-tts-bg)',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      color: 'var(--accent-tts)'
                    }}
                  >
                    <Volume2 size={16} />
                  </div>
                  <div>
                    <div style={{ fontWeight: 600, fontSize: '0.86rem' }}>Text to Voice (TTS)</div>
                    <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
                      Neural speech generator & player
                    </div>
                  </div>
                </div>
                {simulatorMode === 'tts' && activeTab === 'call' && <Check size={16} color="var(--accent-tts)" />}
              </button>
            </div>
          )}
        </div>

        {/* Benchmark Tab */}
        <button
          onClick={() => {
            setActiveTab('benchmark');
            setIsDropdownOpen(false);
          }}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
            padding: '0.6rem 1.35rem',
            borderRadius: '999px',
            border: activeTab === 'benchmark' ? '1px solid var(--accent-benchmark-border)' : '1px solid var(--border)',
            background: activeTab === 'benchmark' ? 'var(--accent-benchmark-bg)' : 'var(--bg-card)',
            color: activeTab === 'benchmark' ? 'var(--accent-benchmark)' : 'var(--text-muted)',
            fontWeight: 600,
            fontSize: '0.9rem',
            cursor: 'pointer',
            boxShadow: activeTab === 'benchmark' ? '0 2px 14px var(--accent-benchmark-glow)' : 'var(--shadow-xs)',
            transition: 'all 0.2s ease'
          }}
        >
          <Activity size={16} />
          <span>Push-to-Talk Benchmark</span>
        </button>
      </div>

      <main style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
        {activeTab === 'call' ? (
          <VoiceCallSimulator
            language={language}
            onLanguageChange={handleLanguageChange}
            model={model}
            onModelChange={handleModelChange}
            engine={engine}
            onEngineChange={handleEngineChange}
            onCallResult={(res) => setResult(res)}
            health={health}
            mode={simulatorMode}
            onModeChange={handleSimulatorModeChange}
            onOpenSherpa={() => setIsSherpaModalOpen(true)}
          />
        ) : (
          <AudioRecorder
            isRecording={isRecording}
            recordingDuration={recordingDuration}
            isProcessing={isProcessing}
            error={micError || processError}
            analyser={analyser}
            language={language}
            onLanguageChange={handleLanguageChange}
            model={model}
            onModelChange={handleModelChange}
            engine={engine}
            onEngineChange={handleEngineChange}
            onStart={startRecording}
            onStop={handleStopRecording}
          />
        )}

        {result && <LatencyDashboard latency={result.latency} />}

        {result && activeTab === 'benchmark' && (
          <ChatTranscript
            transcript={result.transcript}
            reply={result.reply}
            language={result.language}
          />
        )}
      </main>

      <SettingsModal
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
        onSaveSettings={fetchHealth}
      />

      <SherpaModelSelector
        isOpen={isSherpaModalOpen}
        onClose={() => setIsSherpaModalOpen(false)}
        onModelSelected={fetchHealth}
      />
    </div>
  );
}
