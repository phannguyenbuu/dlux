import { useMemo } from "react";
import createSourceHandlers from "../services/sourceHandlers";

export default function useSourceHandlers({
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
  return useMemo(
    () =>
      createSourceHandlers({
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
      }),
    [
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
    ]
  );
}
