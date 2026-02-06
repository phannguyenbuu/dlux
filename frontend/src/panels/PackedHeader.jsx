export default function PackedHeader(props) {
  const {
    showImages,
    setShowImages,
    showStroke,
    setShowStroke,
    showLabels,
    setShowLabels,
    labelFontFamily,
    setLabelFontFamily,
    labelFontSize,
    setLabelFontSize,
    downloadStage,
    regionRef,
    scene,
  } = props;

  return (
    <div className="preview-header">
      <div className="preview-title">Packed (Konva)</div>
      <div className="preview-controls">
        <label className="checkbox">
          <input
            type="checkbox"
            checked={showImages}
            onChange={(e) => {
              setShowImages(e.target.checked);
            }}
          />
          Image
        </label>
        <label className="checkbox">
          <input
            type="checkbox"
            checked={showStroke}
            onChange={(e) => {
              setShowStroke(e.target.checked);
            }}
          />
          Stroke
        </label>
        <label className="checkbox">
          <input
            type="checkbox"
            checked={showLabels}
            onChange={(e) => {
              setShowLabels(e.target.checked);
            }}
          />
          Label
        </label>
        <label className="mini-input">
          Font
          <select value={labelFontFamily} onChange={(e) => setLabelFontFamily(e.target.value)}>
            <option value="Arial">Arial</option>
            <option value="Helvetica">Helvetica</option>
            <option value="Verdana">Verdana</option>
            <option value="Tahoma">Tahoma</option>
            <option value="Georgia">Georgia</option>
            <option value="Times New Roman">Times New Roman</option>
            <option value="Courier New">Courier New</option>
          </select>
        </label>
        <label className="mini-input">
          Size
          <input
            type="number"
            min="4"
            max="64"
            value={labelFontSize}
            onChange={(e) => setLabelFontSize(parseFloat(e.target.value || "12"))}
          />
        </label>
        <button
          className="icon-button"
          title="Download"
          onClick={() =>
            downloadStage(
              regionRef,
              "packed-konva.svg",
              scene?.canvas ? { w: scene.canvas.w, h: scene.canvas.h } : null
            )
          }
        >
          {"\u2193"}
        </button>
      </div>
    </div>
  );
}
