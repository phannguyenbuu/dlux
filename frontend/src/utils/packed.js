import { bboxFromPts, pointInPolyWithOffset, transformPath } from "./geometry";

export const buildPackedPolyData = (data) => {
  if (!data?.regions || !data?.zone_id) return [];
  return data.regions.map((poly, rid) => {
    const zid = data.zone_id[rid];
    const shift = data.zone_shift?.[zid] || data.zone_shift?.[parseInt(zid, 10)];
    const rot = data.zone_rot?.[zid] ?? data.zone_rot?.[parseInt(zid, 10)] ?? 0;
    const center = data.zone_center?.[zid] || data.zone_center?.[parseInt(zid, 10)] || [0, 0];
    const tpts = shift ? transformPath(poly, shift, rot, center) : poly;
    return { pts: tpts, bbox: bboxFromPts(tpts) };
  });
};

export const buildPackedEmptyCells = (data, packedPolyData) => {
  if (!data?.canvas || !packedPolyData?.length) return [];
  const cellSize = 6;
  const radius = 3;
  const pts = [];
  const w = data.canvas.w;
  const h = data.canvas.h;
  for (let y = cellSize; y + cellSize <= h; y += cellSize) {
    for (let x = cellSize; x + cellSize <= w; x += cellSize) {
      const cx = x + radius;
      const cy = y + radius;
      if (cx < radius || cy < radius || cx > w - radius || cy > h - radius) continue;
      const corners = [
        [x, y],
        [x + cellSize, y],
        [x, y + cellSize],
        [x + cellSize, y + cellSize],
        [x + cellSize * 0.5, y],
        [x + cellSize * 0.5, y + cellSize],
        [x, y + cellSize * 0.5],
        [x + cellSize, y + cellSize * 0.5],
      ];
      let inside = false;
      for (const poly of packedPolyData) {
        const bb = poly.bbox;
        if (!bb) continue;
        const minx = bb.minx - radius;
        const maxx = bb.maxx + radius;
        const miny = bb.miny - radius;
        const maxy = bb.maxy + radius;
        if (cx < minx || cx > maxx || cy < miny || cy > maxy) continue;
        for (const pt of corners) {
          if (pointInPolyWithOffset(pt, poly.pts, radius)) {
            inside = true;
            break;
          }
        }
        if (inside) break;
      }
      if (!inside) pts.push([cx, cy]);
    }
  }
  return pts;
};

export const logPackedPreview = (data) => {
  if (!data) return;
  const placements = data.placements || [];
  const binW = data.canvas?.w || 0;
  const binH = data.canvas?.h || 0;
  let placed = 0;
  let unplaced = 0;
  let placedArea = 0;
  const unplacedIds = [];
  const pageCounts = {};
  placements.forEach((p, idx) => {
    const dx = p?.[0] ?? -1;
    const dy = p?.[1] ?? -1;
    const bw = p?.[2] ?? 0;
    const bh = p?.[3] ?? 0;
    if (dx < 0 || dy < 0 || bw <= 0 || bh <= 0) {
      unplaced += 1;
      unplacedIds.push(idx);
    } else {
      placed += 1;
      placedArea += bw * bh;
    }
    const bin = data.placement_bin?.[idx];
    if (bin != null) {
      pageCounts[bin] = (pageCounts[bin] || 0) + 1;
    }
  });
  const binArea = binW * binH;
  const fillRatio = binArea ? placedArea / binArea : 0;
  const debug = data.debug || {};
  console.groupCollapsed(
    `[packed preview] placed=${placed} unplaced=${unplaced} area=${placedArea}/${binArea} (${(
      fillRatio * 100
    ).toFixed(2)}%)`
  );
  console.log("canvas", data.canvas);
  console.log("placed/unplaced", { placed, unplaced, placedArea, binArea, fillRatio });
  console.log("unplaced_ids", unplacedIds);
  console.log("page_counts", pageCounts);
  console.log("debug", debug);
  console.log("pack_settings", {
    packPadding: data.pack_padding,
    packMarginX: data.pack_margin_x,
    packMarginY: data.pack_margin_y,
    packGrid: data.pack_grid,
    packAngle: data.pack_angle,
    packMode: data.pack_mode,
    drawScale: data.draw_scale,
  });
  console.groupEnd();
};
