import { useMemo } from "react";
import { Line } from "react-konva";
import { toPoints } from "../utils/geometry";

export default function useSourceLayers(segs, nodes, borderSegments, scale) {
  const nodeLayer = useMemo(() => {
    if (!segs.length || !nodes.length) return null;
    return segs.map(([a, b], idx) => {
      const p1 = nodes[a];
      const p2 = nodes[b];
      return (
        <Line
          key={`s-${idx}`}
          points={[p1.x, p1.y, p2.x, p2.y]}
          stroke="#f5f6ff"
          strokeWidth={(1 / scale) * 2}
          strokeScaleEnabled={false}
        />
      );
    });
  }, [segs, nodes, scale]);

  const borderLayer = useMemo(() => {
    if (!borderSegments.length) return null;
    return borderSegments.map((seg, idx) => (
      <Line
        key={`b-${idx}`}
        points={toPoints(seg)}
        stroke="#f5f6ff"
        strokeWidth={(1 / scale) * 2}
        strokeScaleEnabled={false}
      />
    ));
  }, [borderSegments, scale]);

  return { nodeLayer, borderLayer };
}
