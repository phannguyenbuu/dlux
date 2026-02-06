import { transformPath } from "../utils/geometry";
import { parseSvgSize, parseOverlayItems, buildSegmentsFromSvg } from "../utils/svg";
import { buildPackedEmptyCells, buildPackedPolyData, logPackedPreview } from "../utils/packed";
import { loadImageFromSrc } from "../utils/overlay";
import {
  calcBounds,
  calcBoundsFromLines,
  snapNodes,
  splitAtIntersections,
} from "../utils/sourceGraph";
const serializeOverlays = (items) =>
  (items || []).map((item) => ({
    id: item.id,
    src: item.src,
    x: item.x,
    y: item.y,
    width: item.width,
    height: item.height,
    scaleX: item.scaleX,
    scaleY: item.scaleY,
    rotation: item.rotation,
    zid: item.zid ?? null,
  }));

export default function createSceneLoader(deps) {
  const {
    snap,
    packPadding,
    packMarginX,
    packMarginY,
    drawScale,
    packGrid,
    packAngle,
    packMode,
    setError,
    setAutoFit,
    setSceneLoading,
    setSvgSize,
    setOverlayItems,
    setSelectedOverlayId,
    setSvgFallback,
    setBorderSegments,
    setRawSegments,
    setNodes,
    setSegs,
    setScene,
    setZoneScene,
    setDrawScale,
    setLabels,
    setPackedImageSrc,
    setPackedImageSrc2,
    setPackedLabels,
    setScale,
    setPos,
    setRegionScale,
    setRegionPos,
    setRegion2Scale,
    setRegion2Pos,
    setZoneScale,
    setZonePos,
    stageSize,
    fitRegionToView,
    fitRegion2ToView,
    fitZoneToView,
  } = deps;
  const loadScene = async (fit = true, updatePacked = true, updateZone = true) => {
    try {
      setError("");
      setAutoFit(fit);
      setSceneLoading(true);
      let savedView = null;
      try {
        const stateRes = await fetch("/api/state");
        if (stateRes.ok) {
          const stateJson = await stateRes.json();
          savedView = stateJson?.view || null;
        }
      } catch {
        savedView = null;
      }
      const svgRes = await fetch("/out/convoi.svg");
      if (!svgRes.ok) throw new Error(`svg fetch failed: ${svgRes.status}`);
      const svgText = await svgRes.text();
      const parsedSize = parseSvgSize(svgText);
      setSvgSize(parsedSize);
      const overlayParsed = parseOverlayItems(svgText);
      if (overlayParsed.length) {
        const hydrated = (
          await Promise.all(
            overlayParsed.map(async (item) => {
              const img = await loadImageFromSrc(item.src);
              const width = item.width || img?.width || 0;
              const height = item.height || img?.height || 0;
              return {
                ...item,
                width,
                height,
                scaleX: Number.isFinite(item.scaleX) ? item.scaleX : 1,
                scaleY: Number.isFinite(item.scaleY) ? item.scaleY : 1,
                rotation: Number.isFinite(item.rotation) ? item.rotation : 0,
                img,
              };
            })
          )
        ).filter(Boolean);
        setOverlayItems(hydrated);
      } else {
        setOverlayItems([]);
      }
      setSelectedOverlayId(null);
      const parsed = buildSegmentsFromSvg(svgText);
      const segments = parsed.segments;
      const borders = parsed.borderSegments;
      setSvgFallback(segments);
      setBorderSegments(borders);
      setRawSegments(segments);
      const nonBorder = segments.filter(
        (seg) =>
          !borders.some(
            (b) =>
              b[0][0] === seg[0][0] &&
              b[0][1] === seg[0][1] &&
              b[1][0] === seg[1][0] &&
              b[1][1] === seg[1][1]
          )
      );
      const splitSegments = splitAtIntersections(nonBorder);
      const snapped = snapNodes(splitSegments, snap);
      setNodes(snapped.nodes);
      setSegs(snapped.segs);

      const res = await fetch(
        `/api/scene?snap=${snap}&pack_padding=${packPadding}&pack_margin_x=${packMarginX}&pack_margin_y=${packMarginY}&draw_scale=${drawScale}&pack_grid=${packGrid}&pack_angle=${packAngle}&pack_mode=${packMode}`
      );
      if (!res.ok) {
        throw new Error(`scene fetch failed: ${res.status}`);
      }
      const data = await res.json();
      setScene(data);
      if (updateZone) {
        setZoneScene(data);
      }
      logPackedPreview(data);
      if (typeof data.draw_scale === "number") {
        setDrawScale(data.draw_scale);
      }
      const initLabels = Object.values(data.zone_labels || {}).map((v) => ({
        id: `z-${v.label}`,
        x: v.x,
        y: v.y,
        label: `${v.label}`,
      }));
      setLabels(initLabels);

      if (updatePacked) {
        setPackedImageSrc(`/out/packed.svg?t=${Date.now()}`);
        setPackedImageSrc2(`/out/packed_page2.svg?t=${Date.now()}`);
        const packedPolyData = buildPackedPolyData(data);
        const emptyCells = buildPackedEmptyCells(data, packedPolyData);
        let cachedPacked = {};
        try {
          const labelRes = await fetch("/api/packed_labels");
          if (labelRes.ok) {
            cachedPacked = (await labelRes.json()) || {};
          }
        } catch {
          cachedPacked = {};
        }
        const usedCell = new Set();
        const cellIndex = (pt) => `${Math.round(pt[0] / 10)}:${Math.round(pt[1] / 10)}`;
        const nextPackedLabels = Object.entries(data.zone_labels || {}).map(([zid, v]) => {
          const shift = data.zone_shift?.[zid] || data.zone_shift?.[parseInt(zid, 10)];
          const rot = data.zone_rot?.[zid] ?? data.zone_rot?.[parseInt(zid, 10)] ?? 0;
          const center = data.zone_center?.[zid] || data.zone_center?.[parseInt(zid, 10)] || [0, 0];
          let px = v.x;
          let py = v.y;
          const cached = cachedPacked?.[String(zid)];
          if (cached && Number.isFinite(cached.x) && Number.isFinite(cached.y)) {
            px = cached.x;
            py = cached.y;
          } else {
            let tx = v.x;
            let ty = v.y;
            if (shift) {
              const [pt] = transformPath([[v.x, v.y]], shift, rot, center);
              if (pt) {
                tx = pt[0];
                ty = pt[1];
              }
            }
            let best = null;
            let bestScore = Infinity;
            const lx = Math.round(tx / 10);
            const ly = Math.round(ty / 10);
            const minCx = lx - 10;
            const maxCx = lx + 10;
            const minCy = ly - 10;
            const maxCy = ly + 10;
            for (const cell of emptyCells) {
              const idx = cellIndex(cell);
              if (usedCell.has(idx)) continue;
              const cx = Math.round(cell[0] / 10);
              const cy = Math.round(cell[1] / 10);
              if (cx < minCx || cx > maxCx || cy < minCy || cy > maxCy) continue;
              const score = Math.abs(cx - lx) + Math.abs(cy - ly);
              if (score < bestScore) {
                bestScore = score;
                best = cell;
              }
            }
            if (best) {
              px = best[0];
              py = best[1];
              usedCell.add(cellIndex(best));
            } else {
              px = tx;
              py = ty;
            }
          }
          if (data.canvas) {
            const r = 3;
            const maxX = data.canvas.w - r;
            const maxY = data.canvas.h - r;
            px = Math.max(r, Math.min(maxX, px));
            py = Math.max(r, Math.min(maxY, py));
          }
          const mapped = data.zone_label_map?.[zid] ?? data.zone_label_map?.[parseInt(zid, 10)];
          const label = mapped != null ? mapped : v.label;
          return { id: `pz-${zid}`, zid: String(zid), x: px, y: py, label: `${label}` };
        });
        setPackedLabels(nextPackedLabels);
      }

      if (savedView?.source?.scale && savedView?.source?.pos) {
        setScale(savedView.source.scale);
        setPos(savedView.source.pos);
        setAutoFit(false);
      }
      if (savedView?.region?.scale && savedView?.region?.pos) {
        setRegionScale(savedView.region.scale);
        setRegionPos(savedView.region.pos);
      }
      if (savedView?.region2?.scale && savedView?.region2?.pos) {
        setRegion2Scale(savedView.region2.scale);
        setRegion2Pos(savedView.region2.pos);
      }
      if (savedView?.zone?.scale && savedView?.zone?.pos) {
        setZoneScale(savedView.zone.scale);
        setZonePos(savedView.zone.pos);
      }
      if (fit && !(savedView?.source?.scale && savedView?.source?.pos)) {
        const w = parsedSize.w || data.canvas?.w || 1200;
        const h = parsedSize.h || data.canvas?.h || 800;
        const viewW = stageSize.w;
        const viewH = stageSize.h;
        const fitScale = Math.min(viewW / w, viewH / h) * 0.95;
        setScale(fitScale);
        setPos({
          x: (viewW - w * fitScale) / 2,
          y: (viewH - h * fitScale) / 2,
        });
        if (data.canvas) {
          fitRegionToView({ minx: 0, miny: 0, maxx: data.canvas.w, maxy: data.canvas.h });
        } else {
          fitRegionToView(calcBounds(data.regions || []));
        }
        const regionBounds = calcBounds(data.regions || []);
        fitRegion2ToView(regionBounds);
        const zoneBounds = calcBoundsFromLines(data.zone_boundaries);
        fitZoneToView(zoneBounds);
      }
      setSceneLoading(false);
    } catch (err) {
      setError(err.message || String(err));
      setSceneLoading(false);
    }
  };

  const saveSvg = (nextNodes, nextSegs, nextOverlays) =>
    fetch("/api/save_svg", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        nodes: nextNodes,
        segs: nextSegs,
        overlays: serializeOverlays(nextOverlays),
      }),
    });

  const saveState = async (payload) => {
    if (!payload) return;
    await fetch("/api/state", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  };

  return { loadScene, saveSvg, saveState };
}
