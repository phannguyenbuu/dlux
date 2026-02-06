import {
  findEdgeCandidate as findEdgeCandidateCore,
  findExistingEdgeCandidate as findExistingEdgeCandidateCore,
  pruneIsolatedNodes,
} from "../utils/sourceGraph";
import { edgeKey } from "../utils/geometry";

export default function createSourceHandlers({
  stageRef,
  pos,
  scale,
  edgeMode,
  deleteEdgeMode,
  addNodeMode,
  edgeCandidate,
  deleteEdgeCandidate,
  nodes,
  segs,
  setEdgeCandidate,
  setDeleteEdgeCandidate,
  setNodes,
  setSegs,
  pointSegDist,
}) {
  const handleSourceMouseMove = () => {
    const stage = stageRef.current;
    const pointer = stage?.getPointerPosition?.();
    if (!pointer) return;
    const world = {
      x: (pointer.x - pos.x) / scale,
      y: (pointer.y - pos.y) / scale,
    };
    if (edgeMode) {
      const cand = findEdgeCandidateCore(nodes, segs, world, edgeKey);
      setEdgeCandidate(cand);
      setDeleteEdgeCandidate(null);
    } else if (deleteEdgeMode) {
      const cand = findExistingEdgeCandidateCore(nodes, segs, world, pointSegDist);
      setDeleteEdgeCandidate(cand);
      setEdgeCandidate(null);
    } else {
      setEdgeCandidate(null);
      setDeleteEdgeCandidate(null);
    }
  };

  const handleSourceMouseLeave = () => {
    if (edgeMode) setEdgeCandidate(null);
    if (deleteEdgeMode) setDeleteEdgeCandidate(null);
  };

  const handleSourceMouseDown = () => {
    const stage = stageRef.current;
    const pointer = stage?.getPointerPosition?.();
    if (!pointer) return;
    const world = {
      x: (pointer.x - pos.x) / scale,
      y: (pointer.y - pos.y) / scale,
    };
    if (edgeMode) {
      if (!edgeCandidate) return;
      const key = edgeKey(edgeCandidate.a, edgeCandidate.b);
      const segSet = new Set(segs.map(([a, b]) => edgeKey(a, b)));
      if (segSet.has(key)) return;
      const nextSegs = [...segs, [edgeCandidate.a, edgeCandidate.b]];
      setSegs(nextSegs);
      return;
    }
    if (deleteEdgeMode) {
      if (!deleteEdgeCandidate) return;
      const nextSegs = segs.filter((_, idx) => idx !== deleteEdgeCandidate.idx);
      const pruned = pruneIsolatedNodes(nodes, nextSegs);
      setNodes(pruned.nodes);
      setSegs(pruned.segs);
      return;
    }
    if (addNodeMode) {
      const nextNodes = [...nodes, { id: nodes.length, x: world.x, y: world.y }];
      let nextSegs = [...segs];
      if (nodes.length) {
        let nearest = 0;
        let best = Infinity;
        nodes.forEach((n, idx) => {
          const d = Math.hypot(n.x - world.x, n.y - world.y);
          if (d < best) {
            best = d;
            nearest = idx;
          }
        });
        nextSegs.push([nextNodes.length - 1, nearest]);
      }
      setNodes(nextNodes);
      setSegs(nextSegs);
    }
  };

  return {
    handleSourceMouseMove,
    handleSourceMouseLeave,
    handleSourceMouseDown,
  };
}
