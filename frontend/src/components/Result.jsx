export default function Result({ data }) {
  const isReal = data.verdict === 'REAL'
  const pctReal = Math.round(data.confidence_real * 100)
  const pctFake = Math.round(data.confidence_fake * 100)

  return (
    <div className={`result-card ${isReal ? 'real' : 'fake'}`}>
      <div className="result-glow" />
      <div className="result-inner">

        {/* Verdict row */}
        <div className="verdict-row">
          <div className="verdict-icon">{isReal ? '✅' : '🚨'}</div>
          <div>
            <div className="verdict-label">{isReal ? 'Real Voice' : 'AI Generated'}</div>
            <div className="verdict-sub">
              {isReal
                ? 'This audio appears to be a genuine human voice.'
                : 'This audio shows signs of AI voice synthesis.'}
            </div>
          </div>
        </div>

        {/* Confidence bars */}
        <div className="score-bars">
          <div className="score-bar-row">
            <span className="score-bar-label" style={{ color: 'var(--green)' }}>Human</span>
            <div className="score-bar-track">
              <div className="score-bar-fill real-bar" style={{ width: `${pctReal}%` }} />
            </div>
            <span className="score-bar-pct" style={{ color: 'var(--green)' }}>{pctReal}%</span>
          </div>
          <div className="score-bar-row">
            <span className="score-bar-label" style={{ color: 'var(--red)' }}>AI Fake</span>
            <div className="score-bar-track">
              <div className="score-bar-fill fake-bar" style={{ width: `${pctFake}%` }} />
            </div>
            <span className="score-bar-pct" style={{ color: 'var(--red)' }}>{pctFake}%</span>
          </div>
        </div>

        {/* File meta */}
        <div className="file-meta">
          <div className="meta-item">
            <span className="meta-label">File</span>
            <span className="meta-value">{data.filename}</span>
          </div>
          <div className="meta-item">
            <span className="meta-label">Size</span>
            <span className="meta-value">{data.size_kb} KB</span>
          </div>
          <div className="meta-item">
            <span className="meta-label">Model</span>
            <span className="meta-value">Stub v0.1</span>
          </div>
        </div>

        {/* Disclaimer */}
        <div className="disclaimer">
          ⚠ LIMITATION: Confidence scores are probabilistic estimates only.
          Do not use as sole evidence in legal, journalistic, or medical decisions.
        </div>
      </div>
    </div>
  )
}
