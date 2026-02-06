import { useCallback } from "react";

export default function useSaveState({
  scene,
  nodes,
  segs,
  labels,
  snap,
  view,
  saveState,
}) {
  return useCallback(async () => {
    if (!scene) return;
    await saveState({
      canvas: scene.canvas,
      regions: scene.regions,
      zone_boundaries: scene.zone_boundaries,
      svg_nodes: nodes,
      svg_segments: segs,
      labels,
      snap,
      view,
    });
  }, [scene, nodes, segs, labels, snap, view, saveState]);
}
