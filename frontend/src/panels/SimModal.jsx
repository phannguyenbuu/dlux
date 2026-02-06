export default function SimModal(props) {
  const {
    showSim,
    setShowSim,
    simActiveLabel,
    simStage,
    simProgress,
    setSimProgress,
    simPlaying,
    handleSimPlayToggle,
    handleSimVideoDownload,
    simVideoLoading,
    simWrapRef,
  } = props;

  if (!showSim) return null;

  return (
    <div className="modal-backdrop">
      <div className="modal sim-modal">
        <button className="modal-close" onClick={() => setShowSim(false)}>
          X
        </button>
        <div className="modal-title">Simulate</div>
        <div className="sim-status">
          {simActiveLabel ? `Moving index: ${simActiveLabel}` : "Moving index: -"}
        </div>
        <div className="sim-body" ref={simWrapRef}>
          {simStage}
        </div>
        <div className="sim-controls">
          <button className="icon-button" onClick={handleSimPlayToggle}>
            {simPlaying ? (
              <svg viewBox="0 0 20 20" width="16" height="16" aria-hidden="true">
                <rect x="4" y="3" width="4" height="14" fill="currentColor" />
                <rect x="12" y="3" width="4" height="14" fill="currentColor" />
              </svg>
            ) : (
              <svg viewBox="0 0 20 20" width="16" height="16" aria-hidden="true">
                <polygon points="6,4 16,10 6,16" fill="currentColor" />
              </svg>
            )}
          </button>
          <input
            type="range"
            min="0"
            max="1"
            step="0.001"
            value={simProgress}
            onChange={(e) => setSimProgress(parseFloat(e.target.value))}
          />
          <button className="btn" onClick={handleSimVideoDownload} disabled={simVideoLoading}>
            {simVideoLoading ? "Creating..." : "Download GIF"}
          </button>
        </div>
      </div>
    </div>
  );
}
