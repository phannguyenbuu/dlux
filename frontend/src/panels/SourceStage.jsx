import { Stage, Layer, Rect, Line } from "react-konva";
import SourceEdgePreview from "./SourceEdgePreview";
import SourceOverlayLayer from "./SourceOverlayLayer";
import SourceNodesLayer from "./SourceNodesLayer";

export default function SourceStage(props) {
  const {
    stageSize,
    scale,
    pos,
    handleWheel,
    stageRef,
    onMouseMove,
    onMouseLeave,
    onMouseDown,
    scene,
    nodeLayer,
    borderLayer,
    edgeCandidate,
    deleteEdgeCandidate,
    nodes,
    overlayItems,
    edgeMode,
    deleteEdgeMode,
    addNodeMode,
    setSelectedOverlayId,
    findZoneAtPoint,
    overlayNodeRefs,
    overlayTransformerRef,
    setOverlayItems,
    setNodes,
    setSegs,
    mergeNodesIfClose,
    segs,
    snap,
  } = props;

  return (
    <Stage
      width={stageSize.w}
      height={stageSize.h}
      draggable
      scaleX={scale}
      scaleY={scale}
      x={pos.x}
      y={pos.y}
      onWheel={handleWheel}
      onMouseMove={onMouseMove}
      onMouseLeave={onMouseLeave}
      onMouseDown={onMouseDown}
      ref={stageRef}
    >
      <Layer>
        {scene?.canvas ? (
          <Rect
            x={0}
            y={0}
            width={scene.canvas.w}
            height={scene.canvas.h}
            stroke="#ffffff"
            strokeWidth={2 / scale}
            listening={false}
          />
        ) : null}
      </Layer>
      <Layer>{nodeLayer}</Layer>
      <Layer>{borderLayer}</Layer>
      {edgeCandidate || deleteEdgeCandidate ? (
        <Layer>
          <SourceEdgePreview
            edgeCandidate={edgeCandidate}
            deleteEdgeCandidate={deleteEdgeCandidate}
            nodes={nodes}
            scale={scale}
          />
        </Layer>
      ) : null}
      <Layer name="source-overlay">
        <SourceOverlayLayer
          overlayItems={overlayItems}
          edgeMode={edgeMode}
          deleteEdgeMode={deleteEdgeMode}
          addNodeMode={addNodeMode}
          setSelectedOverlayId={setSelectedOverlayId}
          findZoneAtPoint={findZoneAtPoint}
          overlayNodeRefs={overlayNodeRefs}
          overlayTransformerRef={overlayTransformerRef}
          setOverlayItems={setOverlayItems}
        />
      </Layer>
      <Layer>
        <SourceNodesLayer
          nodes={nodes}
          scale={scale}
          edgeMode={edgeMode}
          deleteEdgeMode={deleteEdgeMode}
          addNodeMode={addNodeMode}
          setNodes={setNodes}
          setSegs={setSegs}
          mergeNodesIfClose={mergeNodesIfClose}
          segs={segs}
          snap={snap}
        />
      </Layer>
    </Stage>
  );
}
