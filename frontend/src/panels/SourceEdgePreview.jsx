import { Line } from "react-konva";

export default function SourceEdgePreview({ edgeCandidate, deleteEdgeCandidate, nodes, scale }) {
  return (
    <>
      {edgeCandidate ? (
        <Line
          points={[
            nodes[edgeCandidate.a].x,
            nodes[edgeCandidate.a].y,
            nodes[edgeCandidate.b].x,
            nodes[edgeCandidate.b].y,
          ]}
          stroke="#cfd6ff"
          opacity={0.4}
          strokeWidth={(1 / scale) * 2}
          strokeScaleEnabled={false}
        />
      ) : null}
      {deleteEdgeCandidate ? (
        <Line
          points={[
            nodes[deleteEdgeCandidate.a].x,
            nodes[deleteEdgeCandidate.a].y,
            nodes[deleteEdgeCandidate.b].x,
            nodes[deleteEdgeCandidate.b].y,
          ]}
          stroke="#ff3b30"
          opacity={0.6}
          strokeWidth={(1 / scale) * 2}
          strokeScaleEnabled={false}
        />
      ) : null}
    </>
  );
}
