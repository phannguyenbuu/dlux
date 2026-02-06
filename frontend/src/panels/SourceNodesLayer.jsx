import { Circle } from "react-konva";

export default function SourceNodesLayer({
  nodes,
  scale,
  edgeMode,
  deleteEdgeMode,
  addNodeMode,
  setNodes,
  setSegs,
  mergeNodesIfClose,
  segs,
  snap,
}) {
  return (
    <>
      {nodes.map((n) => (
        <Circle
          key={`n-${n.id}`}
          x={n.x}
          y={n.y}
          radius={3 / scale}
          fill="red"
          strokeScaleEnabled={false}
          draggable={!edgeMode && !deleteEdgeMode && !addNodeMode}
          onDragMove={(e) => {
            const next = nodes.map((p) =>
              p.id === n.id ? { ...p, x: e.target.x(), y: e.target.y() } : p
            );
            setNodes(next);
          }}
          onDragEnd={(e) => {
            const next = nodes.map((p) =>
              p.id === n.id ? { ...p, x: e.target.x(), y: e.target.y() } : p
            );
            const merged = mergeNodesIfClose(next, segs, n.id, snap);
            setNodes(merged.nodes);
            setSegs(merged.segs);
          }}
        />
      ))}
    </>
  );
}
