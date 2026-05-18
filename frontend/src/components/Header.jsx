export default function Header({ page, setPage }) {
  return (
    <header className="header">
      <div className="header-logo">
        <div className="header-logo-icon">🎙️</div>
        <span>DeepfakeAudio</span>
        <span className="header-badge">BETA</span>
      </div>
      <nav className="header-nav">
        <button className={`nav-link ${page === 'analyze' ? 'active' : ''}`} onClick={() => setPage('analyze')}>
          Analyze
        </button>
        <button className={`nav-link ${page === 'about' ? 'active' : ''}`} onClick={() => setPage('about')}>
          About
        </button>
      </nav>
    </header>
  )
}
