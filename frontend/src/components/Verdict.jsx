export default function Verdict({ data }) {
  const isReal = data.verdict === 'REAL'
  const pReal  = Math.round(data.confidence_real * 100)
  const pFake  = Math.round(data.confidence_fake * 100)

  // SVG semi-circle gauge
  const R      = 60
  const arcLen = Math.PI * R
  const pct    = isReal ? pReal : pFake
  const offset = arcLen - (arcLen * pct / 100)
  const color  = isReal ? '#22c55e' : '#ef4444'

  const feats = data.features || {}

  return (
    <div className={`verdict-card ${isReal ? 'real' : 'fake'}`}>

      {/* ── Verdict heading ── */}
      <div className="verdict-header">
        <div className="verdict-emoji">{isReal ? '✅' : '🚨'}</div>
        <div className="verdict-text-group">
          <div className="verdict-label">{isReal ? 'Real Voice' : 'AI Generated'}</div>
          <div className="verdict-desc">
            {isReal
              ? 'This audio appears to be a genuine human voice.'
              : 'This audio shows signs of AI voice synthesis.'}
          </div>
        </div>
      </div>

      {/* ── Confidence gauge ── */}
      <div className="gauge-wrap">
        <div className="gauge-label">Confidence Score</div>
        <svg className="gauge-svg" viewBox="0 0 160 90">
          <path className="gauge-track"
            d={`M ${80 - R} 80 A ${R} ${R} 0 0 1 ${80 + R} 80`} />
          <path className="gauge-fill"
            d={`M ${80 - R} 80 A ${R} ${R} 0 0 1 ${80 + R} 80`}
            stroke={color}
            strokeDasharray={`${arcLen}`}
            strokeDashoffset={`${offset}`} />
          <text x="80" y="72" textAnchor="middle" fill={color}
            fontSize="20" fontWeight="700" fontFamily="Inter,sans-serif">{pct}%</text>
          <text x="80" y="86" textAnchor="middle" fill="#64748b"
            fontSize="8" fontFamily="Inter,sans-serif">
            {isReal ? 'HUMAN CONFIDENCE' : 'AI CONFIDENCE'}
          </text>
        </svg>
      </div>

      {/* ── Score bars ── */}
      <div className="score-bars">
        <div className="bar-row">
          <span className="bar-label" style={{ color:'var(--green)' }}>Human</span>
          <div className="bar-track">
            <div className="bar-fill human" style={{ width:`${pReal}%` }} />
          </div>
          <span className="bar-pct" style={{ color:'var(--green)' }}>{pReal}%</span>
        </div>
        <div className="bar-row">
          <span className="bar-label" style={{ color:'var(--red)' }}>AI Fake</span>
          <div className="bar-track">
            <div className="bar-fill ai" style={{ width:`${pFake}%` }} />
          </div>
          <span className="bar-pct" style={{ color:'var(--red)' }}>{pFake}%</span>
        </div>
      </div>

      {/* ── Acoustic features breakdown ── */}
      {Object.keys(feats).length > 0 && (
        <div style={{ padding:'16px 24px', borderBottom:'1px solid var(--border)' }}>
          <div style={{ fontSize:'0.68rem', letterSpacing:'0.1em', textTransform:'uppercase',
            color:'var(--text-dim)', fontWeight:500, marginBottom:12 }}>
            Acoustic Indicators
          </div>
          <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:10 }}>
            {[
              { label:'MFCC Variance',     val: feats.mfcc_variance,     hint:'Higher = more human' },
              { label:'Pitch Variation',   val: feats.pitch_variation + ' Hz', hint:'Higher = more natural' },
              { label:'Spectral Flatness', val: feats.spectral_flatness, hint:'Lower = more human' },
              { label:'ZCR Variation',     val: feats.zcr_variation,     hint:'Higher = more natural' },
            ].map(({ label, val, hint }) => (
              <div key={label} style={{ background:'var(--bg-secondary)', border:'1px solid var(--border)',
                borderRadius:8, padding:'10px 12px' }}>
                <div style={{ fontSize:'0.68rem', color:'var(--text-dim)', marginBottom:4 }}>{label}</div>
                <div style={{ fontSize:'0.92rem', fontWeight:600, color:'var(--text)' }}>{val}</div>
                <div style={{ fontSize:'0.65rem', color:'var(--text-dim)', marginTop:2 }}>{hint}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ── File metadata ── */}
      <div className="file-meta">
        <div className="meta-item">
          <span className="meta-label">File</span>
          <span className="meta-val" style={{ maxWidth:160, overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap' }}>
            {data.filename}
          </span>
        </div>
        <div className="meta-item">
          <span className="meta-label">Size</span>
          <span className="meta-val">{data.size_kb} KB</span>
        </div>
        {data.duration_sec && (
          <div className="meta-item">
            <span className="meta-label">Duration</span>
            <span className="meta-val">{data.duration_sec}s</span>
          </div>
        )}
        <div className="meta-item">
          <span className="meta-label">Verdict</span>
          <span className="meta-val" style={{ color: isReal ? 'var(--green)' : 'var(--red)' }}>
            {data.verdict}
          </span>
        </div>
      </div>

      {/* ── Disclaimer ── */}
      <div className="disclaimer">
        <span className="disclaimer-icon">⚠️</span>
        <span>
          <strong>Limitation Notice:</strong> This system uses acoustic heuristics based on known
          differences between human and AI voices. Results may vary with unusual audio conditions.
          Always verify with additional methods for critical decisions.
        </span>
      </div>
    </div>
  )
}
