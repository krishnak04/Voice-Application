import React from 'react';
import { User, Bot, Copy, Check } from 'lucide-react';

export default function ChatTranscript({ transcript, reply, language }) {
  const [copied, setCopied] = React.useState(false);

  const hasTranscript = Boolean(transcript && transcript.trim());
  const hasReply = Boolean(reply && reply.trim());

  if (!hasTranscript && !hasReply) return null;

  const handleCopy = (text) => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <section className="glass-panel chat-container">
      {hasTranscript && (
        <div className="chat-bubble user">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div className="chat-bubble-header" style={{ color: 'var(--accent-cyan)' }}>
              <User size={15} />
              <span>You (Voice Input)</span>
            </div>
            {language && (
              <span style={{ fontSize: '0.72rem', background: 'rgba(56, 189, 248, 0.15)', color: '#38bdf8', padding: '0.15rem 0.5rem', borderRadius: '6px', fontFamily: 'var(--font-mono)' }}>
                Lang: {language.toUpperCase()}
              </span>
            )}
          </div>
          <p style={{ color: 'var(--text-main)', fontSize: '1.05rem', fontStyle: 'italic', marginTop: '0.25rem' }}>
            "{transcript}"
          </p>
        </div>
      )}

      {hasReply && (
        <div className="chat-bubble ai">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div className="chat-bubble-header" style={{ color: 'var(--accent-purple)' }}>
              <Bot size={15} />
              <span>Groq AI Response</span>
            </div>
            <button
              onClick={() => handleCopy(reply)}
              className="btn-icon"
              style={{ padding: '0.3rem', border: 'none' }}
              title="Copy text"
            >
              {copied ? <Check size={14} color="#34d399" /> : <Copy size={14} />}
            </button>
          </div>
          <p style={{ color: 'var(--text-main)', fontSize: '1rem', whiteSpace: 'pre-wrap' }}>
            {reply}
          </p>
        </div>
      )}
    </section>
  );
}
