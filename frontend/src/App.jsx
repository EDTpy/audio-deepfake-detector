import React     from 'react';
import { useState } from 'react'
import Header   from './components/Header'
import Uploader from './components/Uploader'
import Recorder from './components/Recorder'
import Verdict  from './components/Verdict'
import axios    from 'axios'

export default function App() {
  const [tab, setTab]         = useState('upload')
  const [result, setResult]   = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError]     = useState(null)
  const [page, setPage]       = useState('analyze') // 'analyze' | 'about'

  async function handleAnalyze(file) {
    setError(null)
    setResult(null)
    setLoading(true)
    try {
      const form = new FormData()
      form.append('file', file)
      const { data } = await axios.post('/analyze', form)
      setResult(data)
    } catch {
      setError('Cannot reach the backend server. Make sure it is running on port 8000.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <>
      <Header page={page} setPage={setPage} />

      {page === 'about' ? (
        <About />
      ) : (
        <>
          {/* Hero */}
          <div className="hero">
            <div className="hero-tag">
              <span className="pulse" />
              AI Voice Detection System
            </div>
            <h1>Detect <span>Deepfake</span> Audio<br />With Confidence</h1>
            <p>
              Upload or record any audio clip. Our system analyzes vocal patterns
              to determine if the voice is real or AI-generated.
            </p>
          </div>

          {/* Two-column layout */}
          <div className="main">
            {/* LEFT — Input */}
            <div className="card">
              <div className="card-header">
                <div className="card-icon">🎙️</div>
                <div>
                  <div className="card-title">Audio Input</div>
                  <div className="card-subtitle">Upload a file or record from your mic</div>
                </div>
              </div>
              <div className="card-body">
                <div className="tabs">
                  <button className={`tab ${tab === 'upload' ? 'active' : ''}`} onClick={() => setTab('upload')}>
                    📁 Upload File
                  </button>
                  <button className={`tab ${tab === 'record' ? 'active' : ''}`} onClick={() => setTab('record')}>
                    🎙️ Record Live
                  </button>
                </div>

                {tab === 'upload' && <Uploader onAnalyze={handleAnalyze} loading={loading} />}
                {tab === 'record' && <Recorder onAnalyze={handleAnalyze} loading={loading} />}

                {error && (
                  <div className="error-bar">
                    ⚠️ {error}
                  </div>
                )}
              </div>
            </div>

            {/* RIGHT — Results */}
            <div className="right-col">
              {loading && (
                <div className="loading-card">
                  <div className="spinner" />
                  <div className="loading-title">Analyzing Audio…</div>
                  <div className="loading-sub">Extracting vocal features and running detection</div>
                </div>
              )}

              {!loading && !result && (
                <div className="empty-state">
                  <div className="empty-icon">📊</div>
                  <h3>No Analysis Yet</h3>
                  <p>Upload an audio file or record your voice,<br />then click "Run Analysis" to see results here.</p>
                </div>
              )}

              {result && !loading && <Verdict data={result} />}
            </div>
          </div>
        </>
      )}
    </>
  )
}

function About() {
  return (
    <div style={{ maxWidth: 680, margin: '60px auto', padding: '0 24px' }}>
      <div className="card">
        <div className="card-header">
          <div className="card-icon">ℹ️</div>
          <div>
            <div className="card-title">About This Project</div>
            <div className="card-subtitle">Audio Deepfake Detection System</div>
          </div>
        </div>
        <div className="card-body" style={{ lineHeight: 1.8, color: 'var(--text-muted)' }}>
          <p style={{ marginBottom: 16 }}>
            This tool uses signal processing and machine learning to detect AI-generated voices.
            It analyzes features like <strong style={{ color: 'var(--text)' }}>MFCCs</strong>,{' '}
            <strong style={{ color: 'var(--text)' }}>Mel-Spectrograms</strong>, and spectral
            patterns to distinguish human voices from synthetic ones.
          </p>
          <p style={{ marginBottom: 16 }}>
            <strong style={{ color: 'var(--text)' }}>Stack:</strong> FastAPI backend ·
            React frontend · Wav2Vec 2.0 (planned) · librosa feature extraction
          </p>
          <p style={{ padding: '14px 16px', background: 'rgba(234,179,8,0.08)', border: '1px solid rgba(234,179,8,0.2)', borderRadius: 10, fontSize: '0.85rem', color: '#ca8a04' }}>
            ⚠️ Current version uses a placeholder model. Real ML integration is in progress.
            Confidence scores shown are not yet based on a trained model.
          </p>
        </div>
      </div>
    </div>
  )
}
