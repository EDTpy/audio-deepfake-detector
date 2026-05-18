  import { useState, useRef, useEffect } from 'react'
  import WaveSurfer from 'wavesurfer.js'

  function beep(freq, duration, vol = 0.3) {
    try {
      const ctx  = new (window.AudioContext || window.webkitAudioContext)()
      const osc  = ctx.createOscillator()
      const gain = ctx.createGain()
      osc.connect(gain)
      gain.connect(ctx.destination)
      osc.frequency.value = freq
      gain.gain.setValueAtTime(vol, ctx.currentTime)
      gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + duration)
      osc.start()
      osc.stop(ctx.currentTime + duration)
    } catch (_) {}
  }

  function playStartSound() {
    beep(600, 0.08, 0.2)
    setTimeout(() => beep(1000, 0.15, 0.3), 90)
  }

  function playStopSound() {
    beep(1000, 0.08, 0.2)
    setTimeout(() => beep(500, 0.18, 0.3), 90)
  }

  // Convert any audio blob → 16kHz mono WAV blob
  // This fixes the 422 error because librosa can always read WAV
  async function convertToWav(blob) {
    const arrayBuffer = await blob.arrayBuffer()
    const ctx         = new (window.AudioContext || window.webkitAudioContext)()
    const audioBuffer = await ctx.decodeAudioData(arrayBuffer)

    const sampleRate  = 16000
    const offlineCtx  = new OfflineAudioContext(1, audioBuffer.duration * sampleRate, sampleRate)
    const source      = offlineCtx.createBufferSource()
    source.buffer     = audioBuffer
    source.connect(offlineCtx.destination)
    source.start()

    const rendered = await offlineCtx.startRendering()
    const samples  = rendered.getChannelData(0)

    // Encode as WAV
    const wavBuffer = new ArrayBuffer(44 + samples.length * 2)
    const view      = new DataView(wavBuffer)

    function writeStr(offset, str) {
      for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i))
    }

    writeStr(0,  'RIFF')
    view.setUint32(4,  36 + samples.length * 2, true)
    writeStr(8,  'WAVE')
    writeStr(12, 'fmt ')
    view.setUint32(16, 16, true)
    view.setUint16(20, 1,          true)  // PCM
    view.setUint16(22, 1,          true)  // mono
    view.setUint32(24, sampleRate, true)
    view.setUint32(28, sampleRate * 2, true)
    view.setUint16(32, 2,          true)
    view.setUint16(34, 16,         true)
    writeStr(36, 'data')
    view.setUint32(40, samples.length * 2, true)

    let offset = 44
    for (let i = 0; i < samples.length; i++) {
      const s = Math.max(-1, Math.min(1, samples[i]))
      view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true)
      offset += 2
    }

    await ctx.close()
    return new Blob([wavBuffer], { type: 'audio/wav' })
  }

  export default function Recorder({ onAnalyze, loading }) {
    const [status,   setStatus]   = useState('idle')
    const [seconds,  setSeconds]  = useState(0)
    const [audioURL, setAudioURL] = useState(null)
    const [wavBlob,  setWavBlob]  = useState(null)
    const [converting, setConverting] = useState(false)

    const recorderRef = useRef(null)
    const chunksRef   = useRef([])
    const timerRef    = useRef(null)
    const audioRef    = useRef(null)
    const waveRef     = useRef(null)
    const wsRef       = useRef(null)

    useEffect(() => {
      return () => {
        clearInterval(timerRef.current)
        wsRef.current?.destroy()
      }
    }, [])

    useEffect(() => {
      if (!audioURL || status !== 'done') return
      const t = setTimeout(() => {
        if (!waveRef.current) return
        wsRef.current?.destroy()
        wsRef.current = WaveSurfer.create({
          container:     waveRef.current,
          waveColor:     '#ef4444',
          progressColor: '#b91c1c',
          height:        56,
          barWidth:      2,
          barGap:        1,
          barRadius:     2,
          cursorColor:   '#f87171',
        })
        wsRef.current.load(audioURL)
      }, 120)
      return () => clearTimeout(t)
    }, [audioURL, status])

    async function start() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
        const mime   = ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg', 'audio/mp4']
          .find(m => MediaRecorder.isTypeSupported(m)) || ''

        const rec = new MediaRecorder(stream, mime ? { mimeType: mime } : {})
        recorderRef.current = rec
        chunksRef.current   = []

        rec.ondataavailable = (e) => {
          if (e.data && e.data.size > 0) chunksRef.current.push(e.data)
        }

        rec.onstop = async () => {
          stream.getTracks().forEach(t => t.stop())
          const raw = new Blob(chunksRef.current, { type: mime || 'audio/webm' })

          // Preview URL (for the audio player — works in browser even as webm)
          const previewURL = URL.createObjectURL(raw)
          setAudioURL(previewURL)
          setStatus('done')

          // Convert to WAV in background for backend compatibility
          setConverting(true)
          try {
            const wav = await convertToWav(raw)
            setWavBlob(wav)
          } catch (e) {
            console.error('WAV conversion failed, using raw blob:', e)
            setWavBlob(raw)   // fallback to original
          } finally {
            setConverting(false)
          }
        }

        rec.start(100)
        setStatus('recording')
        setSeconds(0)
        playStartSound()
        timerRef.current = setInterval(() => setSeconds(s => s + 1), 1000)

      } catch {
        alert('Microphone access denied. Please allow microphone access in your browser.')
      }
    }

    function stop() {
      if (recorderRef.current?.state === 'recording') recorderRef.current.stop()
      clearInterval(timerRef.current)
      playStopSound()
    }

    function reset() {
      wsRef.current?.destroy()
      wsRef.current = null
      if (audioURL) URL.revokeObjectURL(audioURL)
      setStatus('idle')
      setSeconds(0)
      setAudioURL(null)
      setWavBlob(null)
      setConverting(false)
    }

    const fmt = s =>
      `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`

    return (
      <div className="recorder-wrap">
        <div className={`recorder-timer ${status === 'recording' ? 'live' : ''}`}>
          {fmt(seconds)}
        </div>

        {status === 'recording' && (
          <div className="rec-live-badge"><span className="dot" /> Recording</div>
        )}

        {status === 'idle'      && <button className="rec-ring-btn idle" onClick={start}>⏺</button>}
        {status === 'recording' && <button className="rec-ring-btn stop" onClick={stop}>⏹</button>}
        {status === 'done'      && <button className="rec-ring-btn done" onClick={reset}>🔄</button>}

        <p className="rec-hint">
          {status === 'idle'      && 'Click ⏺ to start — no time limit'}
          {status === 'recording' && 'Click ⏹ to stop recording'}
          {status === 'done'      && 'Click 🔄 to record again'}
        </p>

        {status === 'done' && audioURL && (
          <>
            <div className="audio-preview" style={{ width: '100%' }}>
              <div className="audio-label">▶ Your Recording</div>
              <audio ref={audioRef} controls src={audioURL} style={{ width: '100%' }} />
            </div>

            <div className="waveform-card" style={{ width: '100%' }}>
              <div className="waveform-label">Waveform</div>
              <div ref={waveRef} style={{ minHeight: 56 }} />
              <div className="waveform-controls">
                <button className="wave-btn" onClick={() => wsRef.current?.playPause()}>▶ Play / Pause</button>
                <button className="wave-btn" onClick={() => wsRef.current?.stop()}>⏹ Stop</button>
              </div>
            </div>

            <button
              className="btn-analyze"
              style={{ width: '100%' }}
              disabled={loading || converting || !wavBlob}
              onClick={() => {
                const f = new File([wavBlob], 'recording.wav', { type: 'audio/wav' })
                onAnalyze(f)
              }}
            >
              {converting ? 'Preparing audio…' : loading ? 'Analyzing…' : 'Run Analysis →'}
            </button>
          </>
        )}
      </div>
    )
  }

  