import { Stage, Layer, Line, Text, Rect, Path, Group, Image } from "react-konva";
import PackedHeader from "./PackedHeader";

export default function PackedPanel(props) {
  const {
    scene,
    sceneLoading,
    regionWrapRef,
    regionRef,
    regionScale,
    regionPos,
    regionStageSize,
    handleRegionWheel,
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
    packedFillPaths,
    packedBleedPaths,
    packedFillPaths2,
    packedBleedPaths2,
    packedBleedError,
    packedBleedError2,
    overlayItems,
    transformOverlayToPacked,
    selectedZoneId,
    setSelectedZoneId,
    packedLabels,
    setPackedLabels,
    measureText,
    toPoints,
    transformPath,
    offsetPoints,
  } = props;

  return (
    <div className={`preview tall region-stage ${sceneLoading ? "is-loading" : ""}`} ref={regionWrapRef}>
      <PackedHeader
        showImages={showImages}
        setShowImages={setShowImages}
        showStroke={showStroke}
        setShowStroke={setShowStroke}
        showLabels={showLabels}
        setShowLabels={setShowLabels}
        labelFontFamily={labelFontFamily}
        setLabelFontFamily={setLabelFontFamily}
        labelFontSize={labelFontSize}
        setLabelFontSize={setLabelFontSize}
        downloadStage={downloadStage}
        regionRef={regionRef}
        scene={scene}
      />
      {scene ? (
        <Stage
          width={regionStageSize.w}
          height={regionStageSize.h}
          draggable
          scaleX={regionScale}
          scaleY={regionScale}
          x={regionPos.x}
          y={regionPos.y}
          onWheel={handleRegionWheel}
          ref={regionRef}
        >
          <Layer>
            {scene?.canvas ? (
              <Rect
                x={0}
                y={0}
                width={scene.canvas.w}
                height={scene.canvas.h}
                stroke="#ffffff"
                strokeWidth={2 / regionScale}
                listening={false}
              />
            ) : null}
          </Layer>
          <Layer name="packed-image" visible={showImages}>
            <>
              <Group
                x={(scene?.canvas?.w || 0) / 2}
                y={(scene?.canvas?.h || 0) / 2}
                offsetX={(scene?.canvas?.w || 0) / 2}
                offsetY={(scene?.canvas?.h || 0) / 2}
              >
                {packedFillPaths.map((p, idx) => (
                  <Path
                    key={`fill-path-${idx}`}
                    data={p.d}
                    fill={p.fill}
                    strokeWidth={0}
                    listening={false}
                  />
                ))}
                {packedBleedPaths.map((p, idx) => (
                  <Path
                    key={`bleed-path-${idx}`}
                    data={p.d}
                    fill={p.fill}
                    strokeWidth={0}
                    listening={false}
                  />
                ))}
              </Group>
              <Group
                x={(scene?.canvas?.w || 0) / 2 + (scene?.canvas?.w || 0) + 40}
                y={(scene?.canvas?.h || 0) / 2}
                offsetX={(scene?.canvas?.w || 0) / 2}
                offsetY={(scene?.canvas?.h || 0) / 2}
              >
                {packedFillPaths2.map((p, idx) => (
                  <Path
                    key={`fill-path-2-${idx}`}
                    data={p.d}
                    fill={p.fill}
                    strokeWidth={0}
                    listening={false}
                  />
                ))}
                {packedBleedPaths2.map((p, idx) => (
                  <Path
                    key={`bleed-path-2-${idx}`}
                    data={p.d}
                    fill={p.fill}
                    strokeWidth={0}
                    listening={false}
                  />
                ))}
              </Group>
            </>
          </Layer>
          <Layer name="packed-overlay">
            {overlayItems.map((item) => {
              if (!item?.img || item.zid == null) return null;
              const packed = transformOverlayToPacked(item);
              const bin = scene?.placement_bin?.[item.zid] ?? scene?.placement_bin?.[parseInt(item.zid, 10)];
              const page = bin === 1 ? 1 : 0;
              const xOffset = page === 1 ? (scene?.canvas?.w || 0) + 40 : 0;
              return (
                <Image
                  key={`po-${item.id}`}
                  image={packed.img}
                  x={packed.x + xOffset}
                  y={packed.y}
                  width={packed.width}
                  height={packed.height}
                  offsetX={packed.width / 2}
                  offsetY={packed.height / 2}
                  scaleX={packed.scaleX}
                  scaleY={packed.scaleY}
                  rotation={packed.rotation}
                  listening={false}
                />
              );
            })}
          </Layer>
          <Layer name="packed-stroke" visible={showStroke}>
            {Object.entries(scene.zone_boundaries || {}).flatMap(([zid, paths]) => {
              const bin = scene?.placement_bin?.[zid] ?? scene?.placement_bin?.[parseInt(zid, 10)];
              const page = bin === 1 ? 1 : 0;
              const xOffset = page === 1 ? (scene?.canvas?.w || 0) + 40 : 0;
              return (paths || []).map((p, i) => {
                const shift = scene.zone_shift?.[zid] || scene.zone_shift?.[parseInt(zid, 10)];
                if (!shift) return null;
                const rot = scene.zone_rot?.[zid] ?? scene.zone_rot?.[parseInt(zid, 10)] ?? 0;
                const center =
                  scene.zone_center?.[zid] || scene.zone_center?.[parseInt(zid, 10)] || [0, 0];
                const tpts = transformPath(p, shift, rot, center);
                const isSelected = String(zid) === String(selectedZoneId);
                return (
                  <Line
                    key={`pz-outline-${zid}-${i}`}
                    points={toPoints(offsetPoints(tpts, xOffset, 0))}
                    stroke={isSelected ? "#ff3b30" : "#f5f6ff"}
                    strokeWidth={isSelected ? 3 : 1}
                    strokeScaleEnabled={false}
                    listening={false}
                  />
                );
              });
            })}
          </Layer>
          <Layer name="packed-hit">
            {Object.entries(scene.zone_boundaries || {}).flatMap(([zid, paths]) => {
              const bin = scene?.placement_bin?.[zid] ?? scene?.placement_bin?.[parseInt(zid, 10)];
              const page = bin === 1 ? 1 : 0;
              const xOffset = page === 1 ? (scene?.canvas?.w || 0) + 40 : 0;
              return (paths || []).map((p, i) => {
                const shift = scene.zone_shift?.[zid] || scene.zone_shift?.[parseInt(zid, 10)];
                if (!shift) return null;
                const rot = scene.zone_rot?.[zid] ?? scene.zone_rot?.[parseInt(zid, 10)] ?? 0;
                const center =
                  scene.zone_center?.[zid] || scene.zone_center?.[parseInt(zid, 10)] || [0, 0];
                const tpts = transformPath(p, shift, rot, center);
                return (
                  <Line
                    key={`pz-hit-${zid}-${i}`}
                    points={toPoints(offsetPoints(tpts, xOffset, 0))}
                    stroke="rgba(0,0,0,0)"
                    strokeWidth={8 / regionScale}
                    strokeScaleEnabled={false}
                    onClick={() => setSelectedZoneId(String(zid))}
                  />
                );
              });
            })}
          </Layer>
          <Layer name="packed-label" visible={showLabels}>
            <Group>
              {packedLabels.map((lbl) => {
                const bin =
                  scene?.placement_bin?.[lbl.zid] ?? scene?.placement_bin?.[parseInt(lbl.zid, 10)];
                const page = bin === 1 ? 1 : 0;
                const xOffset = page === 1 ? (scene?.canvas?.w || 0) + 40 : 0;
                const size = Math.max(labelFontSize / regionScale, 6 / regionScale);
                const metrics = measureText(lbl.label, size, labelFontFamily);
                const isSelected = String(lbl.zid) === String(selectedZoneId);
                return (
                  <Text
                    key={lbl.id}
                    x={lbl.x + xOffset}
                    y={lbl.y}
                    text={lbl.label}
                    fill={isSelected ? "#ff3b30" : "#ffffff"}
                    stroke="rgba(0,0,0,0.5)"
                    strokeWidth={1 / regionScale}
                    fontSize={size}
                    fontFamily={labelFontFamily}
                    align="center"
                    verticalAlign="middle"
                    offsetX={metrics.width / 2}
                    offsetY={metrics.height / 2}
                    draggable
                    onDragEnd={(e) => {
                      const next = packedLabels.map((p) =>
                        p.id === lbl.id ? { ...p, x: e.target.x(), y: e.target.y() } : p
                      );
                      setPackedLabels(next);
                      try {
                        fetch("/api/packed_labels", {
                          method: "POST",
                          headers: { "Content-Type": "application/json" },
                          body: JSON.stringify({
                            [String(lbl.zid)]: {
                              x: e.target.x(),
                              y: e.target.y(),
                              label: lbl.label,
                            },
                          }),
                        });
                      } catch {
                        // ignore storage errors
                      }
                    }}
                  />
                );
              })}
            </Group>
          </Layer>
        </Stage>
      ) : null}
      {sceneLoading ? <div className="loading-overlay">Loading...</div> : null}
      {packedBleedError ? <div className="error">{packedBleedError}</div> : null}
      {packedBleedError2 ? <div className="error">{packedBleedError2}</div> : null}
    </div>
  );
}
