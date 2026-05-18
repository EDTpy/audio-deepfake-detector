import { useState, useRef, useEffect } from 'react'
import WaveSurfer from 'wavesurfer.js'

export default function Uploader({ onAnalyze, loading }) {
  const [file, setFile]     = useState(null)
  const [drag, setDrag]     = useState(false)
  const [objURL, setObjURL] = useState(null)
  const inputRef            = useRef()
  const waveRef             = useRef()
  const wsRef               = useRef(null)

  useEffect(() => {
    if (!objURL || !waveRef.current) return
    wsRef.current?.destroy()
    wsRef.current = WaveSurfer.create({
      container:     waveRef.current,
      waveColor:     '#3b82f6',
      progressColor: '#1d4ed8',
      height:        64,
      barWidth:      2,
      barGap:        1,
      barRadius:     2,
      cursorColor:   '#60a5fa',
    })
    wsRef.current.load(objURL)
    return () => wsRef.current?.destroy()
  }, [objURL])

  function pick(f) {
    if (!f || !f.type.startsWith('audio/')) return alert('Please select an audio file.')
    setFile(f)
    if (objURL) URL.revokeObjectURL(objURL)
    setObjURL(URL.createObjectURL(f))
  }

  return (
    <div>
      <div
        className={`drop-zone ${drag ? 'active' : ''}`}
        onClick={() => inputRef.current.click()}
        onDragOver={(e) => { e.preventDefault(); setDrag(true) }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => { e.preventDefault(); setDrag(false); pick(e.dataTransfer.files[0]) }}
      >
        <div className="drop-icon">🎵</div>
        <h3>Drop audio here or click to browse</h3>
        <p>Supports WAV · MP3 · OGG · FLAC · M4A — max 250 MB</p>
        {file && (
          <div className="file-pill">
            <span>✓</span>
            <span>{file.name}</span>
          </div>
        )}
        <input ref={inputRef} type="file" accept="audio/*" style={{ display:'none' }}
          onChange={(e) => pick(e.target.files[0])} />
      </div>

      {objURL && (
        <div className="waveform-card">
          <div className="waveform-label">Waveform Preview</div>
          <div ref={waveRef} id="waveform" />
          <div className="waveform-controls">
            <button className="wave-btn" onClick={() => wsRef.current?.playPause()}>▶ Play / Pause</button>
            <button className="wave-btn" onClick={() => wsRef.current?.stop()}>⏹ Stop</button>
          </div>
        </div>
      )}

      {file && (
        <button className="btn-analyze" disabled={loading} onClick={() => onAnalyze(file)}>
          {loading ? 'Analyzing…' : 'Run Analysis →'}
        </button>
      )}
    </div>
  )
}
