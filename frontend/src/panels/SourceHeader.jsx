export default function SourceHeader({
  edgeMode,
  addNodeMode,
  deleteEdgeMode,
  setEdgeMode,
  setAddNodeMode,
  setDeleteEdgeMode,
  setEdgeCandidate,
  setDeleteEdgeCandidate,
  handleOverlayPick,
  overlayInputRef,
  handleOverlayFileChange,
  overlayFill,
  setOverlayFill,
  selectedOverlayId,
  updateOverlayColor,
  downloadStage,
  stageRef,
  scene,
}) {
  return (
    <div className="preview-header">
      <div className="preview-title">Source (Konva)</div>
      <div className="preview-controls">
        <button
          className={`icon-button ${edgeMode ? "active" : ""}`}
          title="Create Edge"
          onClick={() => {
            setEdgeMode((v) => !v);
            setAddNodeMode(false);
            setDeleteEdgeMode(false);
            setEdgeCandidate(null);
            setDeleteEdgeCandidate(null);
          }}
        >
          <svg viewBox="0 0 20 20" width="16" height="16" aria-hidden="true">
            <circle cx="4" cy="4" r="2" fill="currentColor" />
            <circle cx="16" cy="16" r="2" fill="currentColor" />
            <line x1="5.5" y1="5.5" x2="14.5" y2="14.5" stroke="currentColor" strokeWidth="2" />
          </svg>
        </button>
        <button
          className={`icon-button ${addNodeMode ? "active" : ""}`}
          title="Add Node"
          onClick={() => {
            setAddNodeMode((v) => !v);
            setEdgeMode(false);
            setDeleteEdgeMode(false);
            setEdgeCandidate(null);
            setDeleteEdgeCandidate(null);
          }}
        >
          <svg viewBox="0 0 20 20" width="16" height="16" aria-hidden="true">
            <circle cx="10" cy="10" r="3" fill="currentColor" />
            <line x1="10" y1="4" x2="10" y2="16" stroke="currentColor" strokeWidth="2" />
            <line x1="4" y1="10" x2="16" y2="10" stroke="currentColor" strokeWidth="2" />
          </svg>
        </button>
        <button
          className={`icon-button ${deleteEdgeMode ? "active" : ""}`}
          title="Delete Edge"
          onClick={() => {
            setDeleteEdgeMode((v) => !v);
            setEdgeMode(false);
            setAddNodeMode(false);
            setEdgeCandidate(null);
            setDeleteEdgeCandidate(null);
          }}
        >
          <svg viewBox="0 0 20 20" width="16" height="16" aria-hidden="true">
            <line x1="4" y1="4" x2="16" y2="16" stroke="currentColor" strokeWidth="2" />
            <line x1="16" y1="4" x2="4" y2="16" stroke="currentColor" strokeWidth="2" />
          </svg>
        </button>
        <button className="icon-button" title="Overlay" onClick={handleOverlayPick}>
          <svg viewBox="0 0 20 20" width="16" height="16" aria-hidden="true">
            <rect x="3" y="4" width="12" height="10" rx="1" ry="1" stroke="currentColor" strokeWidth="2" fill="none" />
            <rect x="7" y="6" width="10" height="10" rx="1" ry="1" stroke="currentColor" strokeWidth="2" fill="none" opacity="0.7" />
          </svg>
        </button>
        <input
          ref={overlayInputRef}
          type="file"
          accept=".svg"
          style={{ display: "none" }}
          onChange={handleOverlayFileChange}
        />
        <label className="mini-input">
          Overlay Fill
          <input
            type="color"
            value={overlayFill}
            onChange={(e) => {
              const color = e.target.value;
              setOverlayFill(color);
              if (selectedOverlayId) updateOverlayColor(selectedOverlayId, color);
            }}
          />
        </label>
        <button
          className="icon-button"
          title="Download"
          onClick={() => downloadStage(stageRef, "source-konva.svg", scene?.canvas || null)}
        >
          {"\u2193"}
        </button>
      </div>
    </div>
  );
}
