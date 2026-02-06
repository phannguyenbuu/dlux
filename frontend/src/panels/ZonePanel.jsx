import { Stage, Layer, Line, Text, Rect, Image } from "react-konva";

export default function ZonePanel(props) {
  const {
    zoneWrapRef,
    zoneRef,
    zoneScene,
    scene,
    sceneLoading,
    zoneStageSize,
    zoneScale,
    zonePos,
    handleZoneWheel,
    showImages,
    showLabels,
    selectedZoneId,
    setSelectedZoneId,
    labelFontFamily,
    labelFontSize,
    overlayItems,
    loadScene,
    downloadStage,
    toPoints,
    measureText,
  } = props;
  const source = zoneScene || scene;

  return (
    <div className={`preview half ${sceneLoading ? "is-loading" : ""}`} ref={zoneWrapRef}>
      <div className="preview-header">
        <div className="preview-title">Zone (Konva)</div>
        <div className="preview-controls">
          <button className="btn" onClick={() => loadScene(false, true, false)}>
            Compute
          </button>
          <button
            className="icon-button"
            title="Download"
            onClick={() => downloadStage(zoneRef, "zone-konva.svg", scene?.canvas || null)}
          >
            {"\u2193"}
          </button>
        </div>
      </div>
      {source ? (
        <Stage
          width={zoneStageSize.w}
          height={zoneStageSize.h}
          draggable
          scaleX={zoneScale}
          scaleY={zoneScale}
          x={zonePos.x}
          y={zonePos.y}
          onWheel={handleZoneWheel}
          ref={zoneRef}
        >
          <Layer>
            {source?.canvas ? (
              <Rect
                x={0}
                y={0}
                width={source.canvas.w}
                height={source.canvas.h}
                stroke="#ffffff"
                strokeWidth={2 / zoneScale}
                listening={false}
              />
            ) : null}
          </Layer>
          <Layer name="zone-image" visible={showImages}>
            {source.region_colors
              ? source.regions.map((poly, idx) => (
                  <Line
                    key={`zf-${idx}`}
                    points={toPoints(poly)}
                    closed
                    fill={source.region_colors[idx]}
                    strokeScaleEnabled={false}
                  />
                ))
              : null}
          </Layer>
          <Layer name="zone-overlay">
            {overlayItems.map((item) =>
              item?.img ? (
                <Image
                  key={`zo-${item.id}`}
                  image={item.img}
                  x={item.x}
                  y={item.y}
                  width={item.width}
                  height={item.height}
                  offsetX={item.width / 2}
                  offsetY={item.height / 2}
                  scaleX={item.scaleX}
                  scaleY={item.scaleY}
                  rotation={item.rotation}
                  listening={false}
                />
              ) : null
            )}
          </Layer>
          <Layer name="zone-stroke">
            {Object.entries(source.zone_boundaries || {}).flatMap(([zid, paths]) =>
              paths.map((p, i) => (
                <Line
                  key={`zb2-${zid}-${i}`}
                  points={toPoints(p)}
                  stroke={String(zid) === String(selectedZoneId) ? "#ff3b30" : "#f5f6ff"}
                  strokeWidth={String(zid) === String(selectedZoneId) ? 3 : 1}
                  strokeScaleEnabled={false}
                />
              ))
            )}
          </Layer>
          <Layer name="zone-label" visible={showLabels}>
            {Object.entries(source.zone_labels || {}).map(([zid, lbl]) => {
              const selectedShuffle =
                source?.zone_label_map?.[selectedZoneId] ??
                source?.zone_label_map?.[parseInt(selectedZoneId, 10)];
              const targetLabel = selectedShuffle != null ? selectedShuffle : selectedZoneId;
              const isSelected = String(lbl.label) === String(targetLabel);
              const size = Math.max(labelFontSize / zoneScale, 6 / zoneScale);
              const metrics = measureText(lbl.label, size, labelFontFamily);
              return (
                <Text
                  key={`zl-${zid}`}
                  x={lbl.x}
                  y={lbl.y}
                  text={`${lbl.label}`}
                  fill={isSelected ? "#ff3b30" : "#ffffff"}
                  fontSize={size}
                  fontFamily={labelFontFamily}
                  align="center"
                  verticalAlign="middle"
                  offsetX={metrics.width / 2}
                  offsetY={metrics.height / 2}
                  listening
                  hitStrokeWidth={10 / zoneScale}
                  onClick={() => {
                    setSelectedZoneId(String(zid));
                  }}
                  onTap={() => {
                    setSelectedZoneId(String(zid));
                  }}
                  onMouseDown={() => {
                    setSelectedZoneId(String(zid));
                  }}
                  onTouchStart={() => {
                    setSelectedZoneId(String(zid));
                  }}
                />
              );
            })}
          </Layer>
        </Stage>
      ) : null}
      {sceneLoading ? <div className="loading-overlay">Loading...</div> : null}
    </div>
  );
}
