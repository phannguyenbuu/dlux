import { useEffect } from "react";

export default function useSceneEffects({
  autoFit,
  svgSize,
  stageSize,
  fitToView,
  scene,
  overlayItems,
  setOverlayItems,
  overlayTransformerRef,
  overlayNodeRefs,
  selectedOverlayId,
  setSelectedOverlayId,
  findZoneAtPoint,
  regionStageSize,
  region2StageSize,
  zoneStageSize,
  fitRegionToView,
  fitRegion2ToView,
  fitZoneToView,
  calcBounds,
  calcBoundsFromLines,
}) {
  useEffect(() => {
    if (autoFit && svgSize.w && svgSize.h) {
      fitToView(svgSize.w, svgSize.h);
    }
  }, [svgSize, stageSize, autoFit, fitToView]);

  useEffect(() => {
    if (!scene || !overlayItems.length) return;
    const updated = overlayItems.map((item) => {
      if (!item) return item;
      const zid = findZoneAtPoint({ x: item.x, y: item.y });
      return { ...item, zid: zid ?? item.zid ?? null };
    });
    const changed = updated.some((item, idx) => item?.zid !== overlayItems[idx]?.zid);
    if (changed) setOverlayItems(updated);
  }, [scene, overlayItems, setOverlayItems, findZoneAtPoint]);

  useEffect(() => {
    const tr = overlayTransformerRef.current;
    if (!tr) return;
    const node = selectedOverlayId ? overlayNodeRefs.current[selectedOverlayId] : null;
    if (node) {
      tr.nodes([node]);
    } else {
      tr.nodes([]);
    }
    tr.getLayer()?.batchDraw?.();
  }, [selectedOverlayId, overlayItems, overlayTransformerRef, overlayNodeRefs]);

  useEffect(() => {
    const onKey = (e) => {
      if (!selectedOverlayId) return;
      if (e.key !== "Delete" && e.key !== "Backspace") return;
      setOverlayItems((items) => items.filter((item) => item.id !== selectedOverlayId));
      setSelectedOverlayId(null);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [selectedOverlayId, setOverlayItems, setSelectedOverlayId]);

  useEffect(() => {
    if (autoFit && (scene?.canvas || scene?.regions?.length)) {
      if (scene?.canvas) {
        fitRegionToView({ minx: 0, miny: 0, maxx: scene.canvas.w, maxy: scene.canvas.h });
      } else {
        fitRegionToView(calcBounds(scene.regions || []));
      }
      fitRegion2ToView(calcBounds(scene.regions || []));
      fitZoneToView(calcBoundsFromLines(scene.zone_boundaries));
    }
  }, [
    scene,
    regionStageSize,
    region2StageSize,
    zoneStageSize,
    autoFit,
    fitRegionToView,
    fitRegion2ToView,
    fitZoneToView,
    calcBounds,
    calcBoundsFromLines,
  ]);
}
