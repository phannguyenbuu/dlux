import { Stage, Layer, Line, Rect } from "react-konva";

export default function RegionPanel(props) {
  const {
    scene,
    sceneLoading,
    region2WrapRef,
    region2Ref,
    region2Scale,
    region2Pos,
    region2StageSize,
    handleRegion2Wheel,
    saveSvg,
    saveState,
    loadScene,
    nodes,
    segs,
    overlayItems,
    downloadStage,
    toPoints,
  } = props;

  return (
    <div className={`preview half ${sceneLoading ? "is-loading" : ""}`} ref={region2WrapRef}>
      <div className="preview-header">
        <div className="preview-title">Region (Konva)</div>
        <div className="preview-controls">
          <button
            className="btn"
            onClick={() => {
              saveSvg(nodes, segs, overlayItems).then(async () => {
                await saveState();
                await loadScene(true, false, true);
              });
            }}
          >
            Save
          </button>
          <button
            className="icon-button"
            title="Download"
            onClick={() => downloadStage(region2Ref, "region-konva.svg", scene?.canvas || null)}
          >
            {"\u2193"}
          </button>
        </div>
      </div>
      {scene ? (
        <Stage
          width={region2StageSize.w}
          height={region2StageSize.h}
          draggable
          scaleX={region2Scale}
          scaleY={region2Scale}
          x={region2Pos.x}
          y={region2Pos.y}
          onWheel={handleRegion2Wheel}
          ref={region2Ref}
        >
          <Layer>
            {scene?.canvas ? (
              <Rect
                x={0}
                y={0}
                width={scene.canvas.w}
                height={scene.canvas.h}
                stroke="#ffffff"
                strokeWidth={2 / region2Scale}
                listening={false}
              />
            ) : null}
          </Layer>
          <Layer>
            {scene.regions.map((poly, idx) => (
              <Line
                key={`r2-${idx}`}
                points={toPoints(poly)}
                closed
                stroke="#f5f6ff"
                fill="#bbb"
                strokeWidth={1 / region2Scale}
                strokeScaleEnabled={false}
              />
            ))}
          </Layer>
        </Stage>
      ) : null}
      {sceneLoading ? <div className="loading-overlay">Loading...</div> : null}
    </div>
  );
}
