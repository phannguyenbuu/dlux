import React, { useMemo, useRef, useState, useEffect } from "react";
import Konva from "konva";
import { Stage, Layer, Line, Text, Circle, Rect, Path, Group } from "react-konva";

const toPoints = (pts) => pts.flatMap((p) => [p[0], p[1]]);

const measureText = (text, fontSize, fontFamily) => {
  const size = Number.isFinite(fontSize) ? fontSize : 12;
  const family = fontFamily || "Arial";
  if (Konva?.Util?.getTextWidth) {
    const width = Konva.Util.getTextWidth(text || "", size, family);
    return { width, height: size };
  }
  const width = (text ? text.length : 0) * size * 0.6;
  return { width, height: size };
};
const rotatePt = (pt, angleDeg, cx, cy) => {
  if (!angleDeg) return pt;
  const ang = (angleDeg * Math.PI) / 180;
  const c = Math.cos(ang);
  const s = Math.sin(ang);
  const x = pt[0] - cx;
  const y = pt[1] - cy;
  return [cx + x * c - y * s, cy + x * s + y * c];
};

const transformPath = (pts, shift, rot, center) => {
  if (!pts || !pts.length) return [];
  const dx = shift?.[0] ?? 0;
  const dy = shift?.[1] ?? 0;
  const ang = rot ?? 0;
  const cx = center?.[0] ?? 0;
  const cy = center?.[1] ?? 0;
  return pts.map((p) => {
    const r = rotatePt(p, ang, cx, cy);
    return [r[0] + dx, r[1] + dy];
  });
};

const bboxFromPts = (pts) => {
  if (!pts || !pts.length) return null;
  let minx = pts[0][0];
  let maxx = pts[0][0];
  let miny = pts[0][1];
  let maxy = pts[0][1];
  for (let i = 1; i < pts.length; i++) {
    const x = pts[i][0];
    const y = pts[i][1];
    if (x < minx) minx = x;
    if (x > maxx) maxx = x;
    if (y < miny) miny = y;
    if (y > maxy) maxy = y;
  }
  return { minx, maxx, miny, maxy };
};

const pointInPoly = (pt, poly) => {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const xi = poly[i][0];
    const yi = poly[i][1];
    const xj = poly[j][0];
    const yj = poly[j][1];
    const intersect = yi > pt[1] !== yj > pt[1] &&
      pt[0] < ((xj - xi) * (pt[1] - yi)) / (yj - yi + 0.0) + xi;
    if (intersect) inside = !inside;
  }
  return inside;
};

const pointSegDist = (pt, a, b) => {
  const vx = b[0] - a[0];
  const vy = b[1] - a[1];
  const wx = pt[0] - a[0];
  const wy = pt[1] - a[1];
  const c1 = vx * wx + vy * wy;
  if (c1 <= 0) return Math.hypot(pt[0] - a[0], pt[1] - a[1]);
  const c2 = vx * vx + vy * vy;
  if (c2 <= c1) return Math.hypot(pt[0] - b[0], pt[1] - b[1]);
  const t = c1 / c2;
  const px = a[0] + t * vx;
  const py = a[1] + t * vy;
  return Math.hypot(pt[0] - px, pt[1] - py);
};

const pointInPolyWithOffset = (pt, poly, offset) => {
  if (pointInPoly(pt, poly)) return true;
  for (let i = 0; i < poly.length; i++) {
    const a = poly[i];
    const b = poly[(i + 1) % poly.length];
    if (pointSegDist(pt, a, b) <= offset) return true;
  }
  return false;
};

const buildPackedPolyData = (data) => {
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

const buildPackedEmptyCells = (data, packedPolyData) => {
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
        [x + cellSize, y + cellSize],
        [x, y + cellSize],
        [cx, cy],
      ];
      let inside = false;
      for (const poly of packedPolyData) {
        const bb = poly.bbox;
        if (!bb) continue;
        const minx = bb.minx - radius;
        const maxx = bb.maxx + radius;
        const miny = bb.miny - radius;
        const maxy = bb.maxy + radius;
        if (x > maxx || x + cellSize < minx || y > maxy || y + cellSize < miny) continue;
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

const logPackedPreview = (data) => {
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
    `[packed preview] placed=${placed} unplaced=${unplaced} area=${placedArea}/${binArea} (${(fillRatio * 100).toFixed(
      2
    )}%)`
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

// packed zone boundaries are transformed on backend

const parsePoints = (str) => {
  if (!str) return [];
  const raw = str
    .trim()
    .replace(/\s+/g, " ")
    .split(" ")
    .flatMap((p) => p.split(","))
    .map((v) => v.trim())
    .filter(Boolean);
  const pts = [];
  for (let i = 0; i + 1 < raw.length; i += 2) {
    const x = parseFloat(raw[i]);
    const y = parseFloat(raw[i + 1]);
    if (Number.isFinite(x) && Number.isFinite(y)) pts.push([x, y]);
  }
  return pts;
};

const parseSvgSize = (svgText) => {
  const doc = new DOMParser().parseFromString(svgText, "image/svg+xml");
  const svg = doc.querySelector("svg");
  if (!svg) return { w: 1000, h: 1000 };
  const vb = svg.getAttribute("viewBox");
  if (vb) {
    const parts = vb.replace(/,/g, " ").trim().split(/\s+/).map(parseFloat);
    if (parts.length === 4 && parts.every(Number.isFinite)) {
      return { w: parts[2], h: parts[3] };
    }
  }
  const w = parseFloat(svg.getAttribute("width") || "1000");
  const h = parseFloat(svg.getAttribute("height") || "1000");
  return {
    w: Number.isFinite(w) ? w : 1000,
    h: Number.isFinite(h) ? h : 1000,
  };
};

const buildSegmentsFromSvg = (svgText) => {
  const doc = new DOMParser().parseFromString(svgText, "image/svg+xml");
  const segments = [];
  const borderSegments = [];
  const svgSize = parseSvgSize(svgText);
  const isOuterBorder = (pts) => {
    if (!pts || pts.length < 4) return false;
    const xs = pts.map((p) => p[0]);
    const ys = pts.map((p) => p[1]);
    const minx = Math.min(...xs);
    const maxx = Math.max(...xs);
    const miny = Math.min(...ys);
    const maxy = Math.max(...ys);
    const tol = 1.0;
    return (
      Math.abs(minx - 0) < tol ||
      Math.abs(miny - 0) < tol ||
      Math.abs(maxx - svgSize.w) < tol ||
      Math.abs(maxy - svgSize.h) < tol
    );
  };
  doc.querySelectorAll("line").forEach((el) => {
    const x1 = parseFloat(el.getAttribute("x1") || "0");
    const y1 = parseFloat(el.getAttribute("y1") || "0");
    const x2 = parseFloat(el.getAttribute("x2") || "0");
    const y2 = parseFloat(el.getAttribute("y2") || "0");
    segments.push([[x1, y1], [x2, y2]]);
  });
  doc.querySelectorAll("polyline").forEach((el) => {
    const pts = parsePoints(el.getAttribute("points"));
    const isBorder = isOuterBorder(pts);
    for (let i = 0; i + 1 < pts.length; i++) {
      const seg = [pts[i], pts[i + 1]];
      segments.push(seg);
      if (isBorder) borderSegments.push(seg);
    }
  });
  doc.querySelectorAll("polygon").forEach((el) => {
    const pts = parsePoints(el.getAttribute("points"));
    const isBorder = isOuterBorder(pts);
    for (let i = 0; i + 1 < pts.length; i++) {
      const seg = [pts[i], pts[i + 1]];
      segments.push(seg);
      if (isBorder) borderSegments.push(seg);
    }
    if (pts.length > 2) {
      const seg = [pts[pts.length - 1], pts[0]];
      segments.push(seg);
      if (isBorder) borderSegments.push(seg);
    }
  });
  return { segments, borderSegments };
};

const snapNodes = (segments, snap) => {
  const cells = new Map();
  const nodes = [];
  const nodeSum = [];
  const nodeCnt = [];

  const cellKey = (x, y) => `${Math.floor(x / snap)},${Math.floor(y / snap)}`;

  const findOrCreate = (pt) => {
    const [x, y] = pt;
    if (!snap || snap <= 0) {
      const id = nodes.length;
      nodes.push({ id, x, y });
      return id;
    }
    const cx = Math.floor(x / snap);
    const cy = Math.floor(y / snap);
    let found = -1;
    for (let dx = -1; dx <= 1; dx++) {
      for (let dy = -1; dy <= 1; dy++) {
        const key = `${cx + dx},${cy + dy}`;
        const ids = cells.get(key);
        if (!ids) continue;
        for (const id of ids) {
          const nx = nodes[id].x;
          const ny = nodes[id].y;
          const d2 = (nx - x) * (nx - x) + (ny - y) * (ny - y);
          if (d2 <= snap * snap) {
            found = id;
            break;
          }
        }
        if (found !== -1) break;
      }
      if (found !== -1) break;
    }
    if (found === -1) {
      const id = nodes.length;
      nodes.push({ id, x, y });
      const key = cellKey(x, y);
      if (!cells.has(key)) cells.set(key, []);
      cells.get(key).push(id);
      nodeSum[id] = [x, y];
      nodeCnt[id] = 1;
      return id;
    }
    nodeSum[found][0] += x;
    nodeSum[found][1] += y;
    nodeCnt[found] += 1;
    return found;
  };

  const segs = segments.map(([a, b]) => {
    const ai = findOrCreate(a);
    const bi = findOrCreate(b);
    return [ai, bi];
  });

  // recompute centroid positions
  nodes.forEach((n, i) => {
    if (nodeCnt[i]) {
      n.x = nodeSum[i][0] / nodeCnt[i];
      n.y = nodeSum[i][1] / nodeCnt[i];
    }
  });

  return { nodes, segs };
};

const calcBounds = (polys) => {
  if (!polys || polys.length === 0) return { minx: 0, miny: 0, maxx: 1, maxy: 1 };
  let minx = Infinity;
  let miny = Infinity;
  let maxx = -Infinity;
  let maxy = -Infinity;
  polys.forEach((poly) => {
    poly.forEach((p) => {
      minx = Math.min(minx, p[0]);
      miny = Math.min(miny, p[1]);
      maxx = Math.max(maxx, p[0]);
      maxy = Math.max(maxy, p[1]);
    });
  });
  return { minx, miny, maxx, maxy };
};

const calcBoundsFromLines = (linesDict) => {
  if (!linesDict) return { minx: 0, miny: 0, maxx: 1, maxy: 1 };
  let minx = Infinity;
  let miny = Infinity;
  let maxx = -Infinity;
  let maxy = -Infinity;
  let found = false;
  Object.values(linesDict).forEach((lines) => {
    (lines || []).forEach((pts) => {
      pts.forEach((p) => {
        found = true;
        minx = Math.min(minx, p[0]);
        miny = Math.min(miny, p[1]);
        maxx = Math.max(maxx, p[0]);
        maxy = Math.max(maxy, p[1]);
      });
    });
  });
  if (!found) return { minx: 0, miny: 0, maxx: 1, maxy: 1 };
  return { minx, miny, maxx, maxy };
};

const mergeNodesIfClose = (nodes, segs, movedId, snap) => {
  if (!snap || snap <= 0) return { nodes, segs };
  const moved = nodes.find((n) => n.id === movedId);
  if (!moved) return { nodes, segs };
  let targetId = null;
  for (const n of nodes) {
    if (n.id === movedId) continue;
    const dx = n.x - moved.x;
    const dy = n.y - moved.y;
    if (dx * dx + dy * dy <= snap * snap) {
      targetId = n.id;
      break;
    }
  }
  if (targetId == null) return { nodes, segs };

  const merged = nodes
    .filter((n) => n.id !== movedId)
    .map((n) =>
      n.id === targetId
        ? { ...n, x: (n.x + moved.x) / 2, y: (n.y + moved.y) / 2 }
        : n
    );

  const remap = new Map();
  merged.forEach((n, idx) => {
    remap.set(n.id, idx);
  });
  const newSegs = segs
    .map(([a, b]) => {
      const na = a === movedId ? targetId : a;
      const nb = b === movedId ? targetId : b;
      if (na === nb) return null;
      return [remap.get(na), remap.get(nb)];
    })
    .filter(Boolean);

  const newNodes = merged.map((n, idx) => ({ ...n, id: idx }));
  return { nodes: newNodes, segs: newSegs };
};

const segmentIntersect = (a1, a2, b1, b2) => {
  const x1 = a1[0], y1 = a1[1], x2 = a2[0], y2 = a2[1];
  const x3 = b1[0], y3 = b1[1], x4 = b2[0], y4 = b2[1];
  const dx12 = x2 - x1;
  const dy12 = y2 - y1;
  const dx34 = x4 - x3;
  const dy34 = y4 - y3;
  const denom = dy12 * dx34 - dx12 * dy34;
  if (Math.abs(denom) < 1e-9) return null;
  const t = ((x1 - x3) * dy34 + (y3 - y1) * dx34) / denom;
  const u = ((x3 - x1) * dy12 + (y1 - y3) * dx12) / -denom;
  if (t < 0 || t > 1 || u < 0 || u > 1) return null;
  return { x: x1 + dx12 * t, y: y1 + dy12 * t, t, u };
};

const edgeKey = (a, b) => (a < b ? `${a}-${b}` : `${b}-${a}`);

const splitAtIntersections = (segments) => {
  const splits = segments.map(() => [0, 1]);
  for (let i = 0; i < segments.length; i++) {
    for (let j = i + 1; j < segments.length; j++) {
      const a = segments[i];
      const b = segments[j];
      const inter = segmentIntersect(a[0], a[1], b[0], b[1]);
      if (!inter) continue;
      splits[i].push(inter.t);
      splits[j].push(inter.u);
    }
  }
  const out = [];
  for (let i = 0; i < segments.length; i++) {
    const ts = Array.from(new Set(splits[i].map((v) => Math.max(0, Math.min(1, v))))).sort((a, b) => a - b);
    const [a, b] = segments[i];
    const dx = b[0] - a[0];
    const dy = b[1] - a[1];
    for (let k = 0; k < ts.length - 1; k++) {
      const t0 = ts[k];
      const t1 = ts[k + 1];
      if (t1 - t0 < 1e-6) continue;
      const p0 = [a[0] + dx * t0, a[1] + dy * t0];
      const p1 = [a[0] + dx * t1, a[1] + dy * t1];
      out.push([p0, p1]);
    }
  }
  return out;
};

export default function App() {
  const [snap, setSnap] = useState(1);
  const [scene, setScene] = useState(null);
  const [error, setError] = useState("");
  const [labels, setLabels] = useState([]);
  const [packedLabels, setPackedLabels] = useState([]);
  const [exportMsg, setExportMsg] = useState("");
  const [exportPdfInfo, setExportPdfInfo] = useState(null);
  const [exportPdfLoading, setExportPdfLoading] = useState(false);
  const [showSim, setShowSim] = useState(false);
  const [simPlaying, setSimPlaying] = useState(false);
  const [simProgress, setSimProgress] = useState(0);
  const [simSize, setSimSize] = useState({ w: 800, h: 500 });
  const [simVideoLoading, setSimVideoLoading] = useState(false);
  const simWrapRef = useRef(null);
  const [selectedZoneId, setSelectedZoneId] = useState(null);
  const [rawSegments, setRawSegments] = useState([]);
  const [borderSegments, setBorderSegments] = useState([]);
  const [nodes, setNodes] = useState([]);
  const [segs, setSegs] = useState([]);
  const [svgImage, setSvgImage] = useState(null);
  const [svgFallback, setSvgFallback] = useState([]);
  const [svgSize, setSvgSize] = useState({ w: 1000, h: 1000 });
  const stageRef = useRef(null);
  const leftRef = useRef(null);
  const [scale, setScale] = useState(1);
  const [pos, setPos] = useState({ x: 0, y: 0 });
  const [stageSize, setStageSize] = useState({ w: 800, h: 600 });
  const regionRef = useRef(null);
  const regionWrapRef = useRef(null);
  const [regionScale, setRegionScale] = useState(1);
  const [regionPos, setRegionPos] = useState({ x: 0, y: 0 });
  const [regionStageSize, setRegionStageSize] = useState({ w: 400, h: 400 });
  const region2Ref = useRef(null);
  const region2WrapRef = useRef(null);
  const [region2Scale, setRegion2Scale] = useState(1);
  const [region2Pos, setRegion2Pos] = useState({ x: 0, y: 0 });
  const [region2StageSize, setRegion2StageSize] = useState({ w: 300, h: 200 });
  const zoneRef = useRef(null);
  const zoneWrapRef = useRef(null);
  const [zoneScale, setZoneScale] = useState(1);
  const [zonePos, setZonePos] = useState({ x: 0, y: 0 });
  const [zoneStageSize, setZoneStageSize] = useState({ w: 300, h: 200 });
  const [autoFit, setAutoFit] = useState(true);
  const [showImages, setShowImages] = useState(false);
  const [showStroke, setShowStroke] = useState(true);
  const [showLabels, setShowLabels] = useState(true);
  const [labelFontFamily, setLabelFontFamily] = useState("Arial");
  const [labelFontSize, setLabelFontSize] = useState(12);
  const [packedImageSrc, setPackedImageSrc] = useState("/out/packed.svg");
  const [packedImageSrc2, setPackedImageSrc2] = useState("/out/packed_page2.svg");
  const [packedFillPaths, setPackedFillPaths] = useState([]);
  const [packedBleedPaths, setPackedBleedPaths] = useState([]);
  const [packedBleedError, setPackedBleedError] = useState("");
  const [packedFillPaths2, setPackedFillPaths2] = useState([]);
  const [packedBleedPaths2, setPackedBleedPaths2] = useState([]);
  const [packedBleedError2, setPackedBleedError2] = useState("");
  const [edgeMode, setEdgeMode] = useState(false);
  const [edgeCandidate, setEdgeCandidate] = useState(null);
  const [sceneLoading, setSceneLoading] = useState(true);
  const [packPadding, setPackPadding] = useState(4);
  const [packMarginX, setPackMarginX] = useState(30);
  const [packMarginY, setPackMarginY] = useState(30);
  const [packBleed, setPackBleed] = useState(10);
  const [drawScale, setDrawScale] = useState(0.5);
  const [packGrid, setPackGrid] = useState(5);
  const [packAngle, setPackAngle] = useState(5);
  const [packMode, setPackMode] = useState("fast");
  const [autoPack, setAutoPack] = useState(false);

  const simZoneIds = useMemo(() => {
    const ids = Object.keys(scene?.zone_boundaries || {});
    const getLabel = (zid) => {
      const lbl =
        scene?.zone_label_map?.[zid] ??
        scene?.zone_label_map?.[parseInt(zid, 10)] ??
        zid;
      const num = Number(lbl);
      return Number.isFinite(num) ? num : Number(zid) || 0;
    };
    return ids.sort((a, b) => getLabel(a) - getLabel(b));
  }, [scene]);
  const simZoneIndex = useMemo(() => {
    const map = {};
    simZoneIds.forEach((zid, idx) => {
      map[String(zid)] = idx;
    });
    return map;
  }, [simZoneIds]);
  const simTiming = useMemo(() => {
    const move = 1;
    const hold = 0.2;
    const per = move + hold;
    const total = simZoneIds.length ? simZoneIds.length * per : 1;
    return { move, hold, per, total };
  }, [simZoneIds]);
  const simMoveSeconds = simTiming.move;
  const simHoldSeconds = simTiming.hold;
  const simPerZone = simTiming.per;
  const simTotalSeconds = simTiming.total;
  const simActiveIdx = simZoneIds.length
    ? Math.min(
        simZoneIds.length - 1,
        Math.max(0, Math.floor((simProgress * simTotalSeconds) / simPerZone))
      )
    : -1;
  const simActiveZid = simActiveIdx >= 0 ? simZoneIds[simActiveIdx] : null;
  const simActiveLabel =
    simActiveZid != null
      ? scene?.zone_label_map?.[simActiveZid] ??
        scene?.zone_label_map?.[parseInt(simActiveZid, 10)] ??
        simActiveZid
      : "";
  const simLocalFor = (idx) => {
    if (idx == null || idx < 0) return 0;
    const t = simProgress * simTotalSeconds - idx * simPerZone;
    if (t <= 0) return 0;
    if (t >= simPerZone) return 1;
    if (t >= simMoveSeconds) return 1;
    const x = t / simMoveSeconds;
    return 1 - Math.pow(1 - x, 3);
  };

  const simStage = scene?.canvas
    ? (() => {
        const gap = 20;
        const totalW = (scene.canvas.w * 2) + gap;
        const totalH = scene.canvas.h;
        const fitScale = Math.min(simSize.w / totalW, simSize.h / totalH) * 1.06;
        const offsetX = (simSize.w - totalW * fitScale) / 2;
        const offsetY = (simSize.h - totalH * fitScale) / 2;
        return (
          <Stage
            width={simSize.w}
            height={simSize.h}
            scaleX={fitScale}
            scaleY={fitScale}
            x={offsetX}
            y={offsetY}
          >
            <Layer>
              <Rect
                x={0}
                y={0}
                width={scene.canvas.w}
                height={scene.canvas.h}
                stroke="#ffffff"
                strokeWidth={1}
              />
              <Rect
                x={scene.canvas.w + gap}
                y={0}
                width={scene.canvas.w}
                height={scene.canvas.h}
                stroke="#ffffff"
                strokeWidth={1}
              />
            </Layer>
            <Layer>
              {scene.region_colors
                ? scene.regions.map((poly, idx) => {
                    const zid = scene.zone_id?.[idx];
                    const zidKey = String(zid);
                    const zoneIdx = simZoneIndex[zidKey] ?? 0;
                    const local = simLocalFor(zoneIdx);
                    if (local > 0) return null;
                    const shift =
                      scene.zone_shift?.[zid] || scene.zone_shift?.[parseInt(zid, 10)];
                    if (!shift) return null;
                    const rot = scene.zone_rot?.[zid] ?? scene.zone_rot?.[parseInt(zid, 10)] ?? 0;
                    const center =
                      scene.zone_center?.[zid] || scene.zone_center?.[parseInt(zid, 10)] || [0, 0];
                    const tpts = transformPath(poly, shift, rot, center);
                    return (
                      <Line
                        key={`sim-pack-fill-${idx}`}
                        points={toPoints(tpts)}
                        closed
                        fill={scene.region_colors[idx]}
                        strokeScaleEnabled={false}
                      />
                    );
                  })
                : null}
              {packedLabels.map((lbl) => {
                const zidKey = String(lbl.zid);
                const zoneIdx = simZoneIndex[zidKey] ?? 0;
                const local = simLocalFor(zoneIdx);
                if (local > 0) return null;
                const size = Math.max(labelFontSize * 0.5, 6);
                const metrics = measureText(lbl.label, size, labelFontFamily);
                return (
                  <Text
                    key={`sim-pack-label-${lbl.id}`}
                    x={lbl.x}
                    y={lbl.y}
                    text={lbl.label}
                    fill="#ffffff"
                    fontSize={size}
                    fontFamily={labelFontFamily}
                    align="center"
                    verticalAlign="middle"
                    offsetX={metrics.width / 2}
                    offsetY={metrics.height / 2}
                  />
                );
              })}
            </Layer>
            <Layer>
              {Object.values(scene.zone_labels || {}).map((lbl) => {
                const size = Math.max(labelFontSize * 0.5, 6);
                const metrics = measureText(lbl.label, size, labelFontFamily);
                return (
                  <Text
                    key={`sim-zone-label-${lbl.label}`}
                    x={lbl.x + scene.canvas.w + gap}
                    y={lbl.y}
                    text={lbl.label}
                    fill="#ffffff"
                    fontSize={size}
                    fontFamily={labelFontFamily}
                    align="center"
                    verticalAlign="middle"
                    offsetX={metrics.width / 2}
                    offsetY={metrics.height / 2}
                  />
                );
              })}
            </Layer>
            <Layer>
              {simZoneIds.flatMap((zid) => {
                const paths = scene.zone_boundaries?.[zid] || [];
                return paths.map((p, i) => (
                  <Line
                    key={`sim-zone-${zid}-${i}`}
                    points={toPoints(offsetPoints(p, scene.canvas.w + gap, 0))}
                    stroke="#f5f6ff"
                    strokeWidth={1}
                    closed
                  />
                ));
              })}
            </Layer>
            <Layer>
              {scene.region_colors
                ? scene.regions.map((poly, idx) => {
                    const zid = scene.zone_id?.[idx];
                    const zidKey = String(zid);
                    const zoneIdx = simZoneIndex[zidKey] ?? 0;
                    const local = simLocalFor(zoneIdx);
                    const shift =
                      scene.zone_shift?.[zid] || scene.zone_shift?.[parseInt(zid, 10)];
                    if (!shift) return null;
                    const rot = scene.zone_rot?.[zid] ?? scene.zone_rot?.[parseInt(zid, 10)] ?? 0;
                    const center =
                      scene.zone_center?.[zid] ||
                      scene.zone_center?.[parseInt(zid, 10)] ||
                      [0, 0];
                    const src = transformPath(poly, shift, rot, center);
                    const dst = offsetPoints(poly, scene.canvas.w + gap, 0);
                    const pts =
                      local >= 1
                        ? dst
                        : src.map((sp, k) => {
                            const dp = dst[k] || sp;
                            return [lerp(sp[0], dp[0], local), lerp(sp[1], dp[1], local)];
                          });
                    if (local <= 0) return null;
                    return (
                      <Line
                        key={`sim-move-fill-${idx}`}
                        points={toPoints(pts)}
                        closed
                        fill={scene.region_colors[idx]}
                        strokeScaleEnabled={false}
                      />
                    );
                  })
                : null}
            </Layer>
          </Stage>
        );
      })()
    : null;

  useEffect(() => {
    loadScene();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const updateSize = () => {
      if (!leftRef.current) return;
      const rect = leftRef.current.getBoundingClientRect();
      setStageSize({ w: Math.max(300, rect.width), h: Math.max(300, rect.height) });
    };
    updateSize();
    window.addEventListener("resize", updateSize);
    return () => window.removeEventListener("resize", updateSize);
  }, []);

  useEffect(() => {
    const updateRegionSize = () => {
      if (!regionWrapRef.current) return;
      const rect = regionWrapRef.current.getBoundingClientRect();
      setRegionStageSize({ w: Math.max(200, rect.width), h: Math.max(200, rect.height) });
    };
    updateRegionSize();
    window.addEventListener("resize", updateRegionSize);
    return () => window.removeEventListener("resize", updateRegionSize);
  }, []);

  useEffect(() => {
    const updateRegion2Size = () => {
      if (!region2WrapRef.current) return;
      const rect = region2WrapRef.current.getBoundingClientRect();
      setRegion2StageSize({ w: Math.max(200, rect.width), h: Math.max(200, rect.height) });
    };
    updateRegion2Size();
    window.addEventListener("resize", updateRegion2Size);
    return () => window.removeEventListener("resize", updateRegion2Size);
  }, []);

  useEffect(() => {
    const updateZoneSize = () => {
      if (!zoneWrapRef.current) return;
      const rect = zoneWrapRef.current.getBoundingClientRect();
      setZoneStageSize({ w: Math.max(200, rect.width), h: Math.max(200, rect.height) });
    };
    updateZoneSize();
    window.addEventListener("resize", updateZoneSize);
    return () => window.removeEventListener("resize", updateZoneSize);
  }, []);

  useEffect(() => {
    const updateSimSize = () => {
      if (!simWrapRef.current) return;
      const rect = simWrapRef.current.getBoundingClientRect();
      setSimSize({ w: Math.max(300, rect.width), h: Math.max(200, rect.height) });
    };
    updateSimSize();
    window.addEventListener("resize", updateSimSize);
    return () => window.removeEventListener("resize", updateSimSize);
  }, []);

  const fitToView = (w, h) => {
    const viewW = stageSize.w;
    const viewH = stageSize.h;
    const fitScale = Math.min(viewW / w, viewH / h) * 0.95;
    setScale(fitScale);
    setPos({
      x: (viewW - w * fitScale) / 2,
      y: (viewH - h * fitScale) / 2,
    });
  };

  const fitRegionToView = (bounds) => {
    const rect = regionWrapRef.current?.getBoundingClientRect();
    const viewW = rect?.width || regionStageSize.w;
    const viewH = rect?.height || regionStageSize.h;
    const w = bounds.maxx - bounds.minx;
    const h = bounds.maxy - bounds.miny;
    const fitScale = Math.min(viewW / w, viewH / h) * 0.95;
    setRegionScale(fitScale);
    setRegionPos({
      x: (viewW - w * fitScale) / 2 - bounds.minx * fitScale,
      y: (viewH - h * fitScale) / 2 - bounds.miny * fitScale,
    });
  };

  const fitRegion2ToView = (bounds) => {
    const rect = region2WrapRef.current?.getBoundingClientRect();
    const viewW = rect?.width || region2StageSize.w;
    const viewH = rect?.height || region2StageSize.h;
    const w = bounds.maxx - bounds.minx;
    const h = bounds.maxy - bounds.miny;
    const fitScale = Math.min(viewW / w, viewH / h) * 0.95;
    setRegion2Scale(fitScale);
    setRegion2Pos({
      x: (viewW - w * fitScale) / 2 - bounds.minx * fitScale,
      y: (viewH - h * fitScale) / 2 - bounds.miny * fitScale,
    });
  };

  const fitZoneToView = (bounds) => {
    const rect = zoneWrapRef.current?.getBoundingClientRect();
    const viewW = rect?.width || zoneStageSize.w;
    const viewH = rect?.height || zoneStageSize.h;
    const w = bounds.maxx - bounds.minx;
    const h = bounds.maxy - bounds.miny;
    const fitScale = Math.min(viewW / w, viewH / h) * 0.95;
    setZoneScale(fitScale);
    setZonePos({
      x: (viewW - w * fitScale) / 2 - bounds.minx * fitScale,
      y: (viewH - h * fitScale) / 2 - bounds.miny * fitScale,
    });
  };

  useEffect(() => {
    if (autoFit && svgSize.w && svgSize.h) {
      fitToView(svgSize.w, svgSize.h);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [svgSize, stageSize, autoFit]);

  const loadScene = async (fit = true) => {
    try {
      setError("");
      setAutoFit(fit);
      setSceneLoading(true);
      const svgRes = await fetch("/out/convoi.svg");
      if (!svgRes.ok) throw new Error(`svg fetch failed: ${svgRes.status}`);
      const svgText = await svgRes.text();
      const parsedSize = parseSvgSize(svgText);
      setSvgSize(parsedSize);
      const parsed = buildSegmentsFromSvg(svgText);
      const segments = parsed.segments;
      const borders = parsed.borderSegments;
      setSvgFallback(segments);
      setBorderSegments(borders);
      // no background rendering; keep only geometry
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
      logPackedPreview(data);
      if (typeof data.draw_scale === "number") {
        setDrawScale(data.draw_scale);
      }
      setPackedImageSrc(`/out/packed.svg?t=${Date.now()}`);
      setPackedImageSrc2(`/out/packed_page2.svg?t=${Date.now()}`);
      const packedPolyData = buildPackedPolyData(data);
      const emptyCells = buildPackedEmptyCells(data, packedPolyData);
      const initLabels = Object.values(data.zone_labels || {}).map((v) => ({
        id: `z-${v.label}`,
        x: v.x,
        y: v.y,
        label: `${v.label}`,
      }));
      setLabels(initLabels);
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
      if (fit) {
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

  const handleWheel = (e) => {
    e.evt.preventDefault();
    const scaleBy = 1.05;
    const stage = stageRef.current;
    const oldScale = stage.scaleX();
    const pointer = stage.getPointerPosition();
    const mousePointTo = {
      x: (pointer.x - stage.x()) / oldScale,
      y: (pointer.y - stage.y()) / oldScale,
    };
    const direction = e.evt.deltaY > 0 ? 1 : -1;
    const newScale = direction > 0 ? oldScale / scaleBy : oldScale * scaleBy;
    setScale(newScale);
    setPos({
      x: pointer.x - mousePointTo.x * newScale,
      y: pointer.y - mousePointTo.y * newScale,
    });
  };

  const handleRegionWheel = (e) => {
    e.evt.preventDefault();
    const scaleBy = 1.05;
    const stage = regionRef.current;
    const oldScale = stage.scaleX();
    const pointer = stage.getPointerPosition();
    const mousePointTo = {
      x: (pointer.x - stage.x()) / oldScale,
      y: (pointer.y - stage.y()) / oldScale,
    };
    const direction = e.evt.deltaY > 0 ? 1 : -1;
    const newScale = direction > 0 ? oldScale / scaleBy : oldScale * scaleBy;
    setRegionScale(newScale);
    setRegionPos({
      x: pointer.x - mousePointTo.x * newScale,
      y: pointer.y - mousePointTo.y * newScale,
    });
  };

  const segmentWouldIntersect = (aIdx, bIdx) => {
    if (!nodes[aIdx] || !nodes[bIdx]) return true;
    const a1 = [nodes[aIdx].x, nodes[aIdx].y];
    const a2 = [nodes[bIdx].x, nodes[bIdx].y];
    for (const [s0, s1] of segs) {
      if (s0 === aIdx || s1 === aIdx || s0 === bIdx || s1 === bIdx) {
        continue;
      }
      const b1 = [nodes[s0].x, nodes[s0].y];
      const b2 = [nodes[s1].x, nodes[s1].y];
      const inter = segmentIntersect(a1, a2, b1, b2);
      if (!inter) continue;
      if (inter.t > 1e-6 && inter.t < 1 - 1e-6 && inter.u > 1e-6 && inter.u < 1 - 1e-6) {
        return true;
      }
    }
    return false;
  };

  const findEdgeCandidate = (worldPt) => {
    if (!nodes.length) return null;
    const EDGE_HOVER_DIST = 10;
    const EDGE_MAX_LEN = 80;
    const segSet = new Set(segs.map(([a, b]) => edgeKey(a, b)));
    let best = null;
    let bestDist = Infinity;
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const dx = nodes[j].x - nodes[i].x;
        const dy = nodes[j].y - nodes[i].y;
        const len = Math.hypot(dx, dy);
        if (len > EDGE_MAX_LEN) continue;
        if (segSet.has(edgeKey(i, j))) continue;
        const mx = (nodes[i].x + nodes[j].x) * 0.5;
        const my = (nodes[i].y + nodes[j].y) * 0.5;
        const d = Math.hypot(worldPt.x - mx, worldPt.y - my);
        if (d > EDGE_HOVER_DIST || d >= bestDist) continue;
        if (segmentWouldIntersect(i, j)) continue;
        bestDist = d;
        best = { a: i, b: j };
      }
    }
    return best;
  };

  const handleRegion2Wheel = (e) => {
    e.evt.preventDefault();
    const scaleBy = 1.05;
    const stage = region2Ref.current;
    const oldScale = stage.scaleX();
    const pointer = stage.getPointerPosition();
    const mousePointTo = {
      x: (pointer.x - stage.x()) / oldScale,
      y: (pointer.y - stage.y()) / oldScale,
    };
    const direction = e.evt.deltaY > 0 ? 1 : -1;
    const newScale = direction > 0 ? oldScale / scaleBy : oldScale * scaleBy;
    setRegion2Scale(newScale);
    setRegion2Pos({
      x: pointer.x - mousePointTo.x * newScale,
      y: pointer.y - mousePointTo.y * newScale,
    });
  };

  const handleZoneWheel = (e) => {
    e.evt.preventDefault();
    const scaleBy = 1.05;
    const stage = zoneRef.current;
    const oldScale = stage.scaleX();
    const pointer = stage.getPointerPosition();
    const mousePointTo = {
      x: (pointer.x - stage.x()) / oldScale,
      y: (pointer.y - stage.y()) / oldScale,
    };
    const direction = e.evt.deltaY > 0 ? 1 : -1;
    const newScale = direction > 0 ? oldScale / scaleBy : oldScale * scaleBy;
    setZoneScale(newScale);
    setZonePos({
      x: pointer.x - mousePointTo.x * newScale,
      y: pointer.y - mousePointTo.y * newScale,
    });
  };

  const saveState = async () => {
    if (!scene) return;
    await fetch("/api/state", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        canvas: scene.canvas,
        regions: scene.regions,
        zone_boundaries: scene.zone_boundaries,
        svg_nodes: nodes,
        svg_segments: segs,
        labels,
        snap,
      }),
    });
  };

  const exportPdf = async () => {
    try {
      setError("");
      setExportMsg("");
      if (!scene?.canvas) {
        throw new Error("canvas missing");
      }
      setExportPdfLoading(true);
      setExportPdfInfo(null);
      const size = { w: scene.canvas.w, h: scene.canvas.h };
      const zoneLabelsSvg = (svgText) =>
        injectSvgLabels(svgText, scene.zone_labels, labelFontFamily, labelFontSize);
      const pages = [
        {
          name: "zone_image",
          svg: zoneLabelsSvg(
            captureStageSvg(zoneRef, size, {
              "zone-image": true,
              "zone-stroke": true,
              "zone-label": true,
              "zone-hit": false,
            })
          ),
        },
        {
          name: "zone_noimage",
          svg: zoneLabelsSvg(
            captureStageSvg(zoneRef, size, {
              "zone-image": false,
              "zone-stroke": true,
              "zone-label": true,
              "zone-hit": false,
            })
          ),
        },
        {
          name: "packed_image_nostroke",
          svg: captureStageSvg(regionRef, size, {
            "packed-image": true,
            "packed-stroke": false,
            "packed-label": true,
            "packed-hit": false,
          }),
        },
        {
          name: "packed_noimage_stroke_nolabel",
          svg: captureStageSvg(regionRef, size, {
            "packed-image": false,
            "packed-stroke": true,
            "packed-label": false,
            "packed-hit": false,
          }),
        },
      ];
      const res = await fetch("/api/export_pdf", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          pages,
          fontName: labelFontFamily,
          fontSize: labelFontSize,
        }),
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.error || `export failed: ${res.status}`);
      }
      const data = await res.json().catch(() => ({}));
      if (data?.name) {
        setExportPdfInfo({ name: data.name });
      }
      setExportMsg("Export PDF Done");
      setTimeout(() => setExportMsg(""), 3000);
    } catch (err) {
      setError(err.message || String(err));
    } finally {
      setExportPdfLoading(false);
    }
  };

  useEffect(() => {
    if (!simPlaying) return;
    let raf = 0;
    let last = performance.now();
    const tick = (now) => {
      const dt = (now - last) / 1000;
      last = now;
      setSimProgress((p) => {
        const next = Math.min(1, p + dt / simTotalSeconds);
        if (next >= 1) setSimPlaying(false);
        return next;
      });
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [simPlaying, simTotalSeconds]);

  useEffect(() => {
    if (!autoPack) return;
    const id = setTimeout(() => {
      loadScene(false);
    }, 500);
    return () => clearTimeout(id);
  }, [packPadding, packMarginX, packMarginY, packBleed, packGrid, packAngle, packMode, autoPack]);

  const parsePackedSvg = (text) => {
    const doc = new DOMParser().parseFromString(text, "image/svg+xml");
    const fill = doc.querySelector('g#fill');
    const bleed = doc.querySelector('g#bleed');
    const parsePaths = (node) =>
      Array.from(node?.querySelectorAll("path") || []).map((p) => ({
        d: p.getAttribute("d") || "",
        fill: p.getAttribute("fill") || "#000000",
      }));
    const fillPaths = parsePaths(fill).filter((p) => p.d);
    const bleedPaths = parsePaths(bleed).filter((p) => p.d);
    return { fillPaths, bleedPaths, hasBleed: !!bleed };
  };

  function lerp(a, b, t) {
    return a + (b - a) * t;
  }

  const escapeXml = (value) => {
    if (value == null) return "";
    return String(value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&apos;");
  };

  const matrixToAttr = (m) => {
    if (!m || m.length < 6) return "";
    const [a, b, c, d, e, f] = m.map((v) => (Number.isFinite(v) ? v : 0));
    return `matrix(${a} ${b} ${c} ${d} ${e} ${f})`;
  };

  const buildSvgFromStage = (stage, exportSize = null) => {
    const width = exportSize?.w || stage.width();
    const height = exportSize?.h || stage.height();
    const parts = [
      `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">`,
    ];

    const pushAttrs = (attrs) => {
      const out = [];
      Object.entries(attrs).forEach(([key, val]) => {
        if (val == null || val === "" || val === false) return;
        out.push(`${key}="${escapeXml(val)}"`);
      });
      return out.join(" ");
    };

    const addShape = (node) => {
      if (!node.isVisible?.() || node.opacity?.() === 0) return;
      const transform = node.getAbsoluteTransform?.();
      const matrix = transform ? matrixToAttr(transform.getMatrix()) : "";
      const strokeScaleEnabled =
        typeof node.strokeScaleEnabled === "function" ? node.strokeScaleEnabled() : true;
      const common = {
        transform: matrix || undefined,
        opacity: node.opacity?.(),
        fill: node.fill?.() ?? undefined,
        "fill-opacity": node.fillOpacity?.(),
        stroke: node.stroke?.() ?? undefined,
        "stroke-opacity": node.strokeOpacity?.(),
        "stroke-width": node.strokeWidth?.(),
        "vector-effect": strokeScaleEnabled ? undefined : "non-scaling-stroke",
      };

      const className = node.getClassName?.();
      if (className === "Line") {
        const pts = node.points?.() || [];
        if (pts.length < 2) return;
        const pairs = [];
        for (let i = 0; i + 1 < pts.length; i += 2) {
          pairs.push(`${pts[i]},${pts[i + 1]}`);
        }
        const closed = node.closed?.();
        const tag = closed ? "polygon" : "polyline";
        const attrs = {
          ...common,
          points: pairs.join(" "),
          fill: closed ? common.fill ?? "none" : "none",
        };
        parts.push(`<${tag} ${pushAttrs(attrs)} />`);
        return;
      }

      if (className === "Path") {
        const d = node.data?.();
        if (!d) return;
        const attrs = { ...common, d };
        parts.push(`<path ${pushAttrs(attrs)} />`);
        return;
      }

      if (className === "Rect") {
        const w = node.width?.();
        const h = node.height?.();
        if (!w || !h) return;
        const attrs = { ...common, x: 0, y: 0, width: w, height: h };
        parts.push(`<rect ${pushAttrs(attrs)} />`);
        return;
      }

      if (className === "Circle") {
        const r = node.radius?.();
        if (!r) return;
        const attrs = { ...common, cx: 0, cy: 0, r };
        parts.push(`<circle ${pushAttrs(attrs)} />`);
        return;
      }

      if (className === "Text") {
        const text = node.text?.();
        if (text == null) return;
        const absPos = node.getAbsolutePosition?.() || { x: 0, y: 0 };
        const attrs = {
          fill: common.fill,
          "fill-opacity": common["fill-opacity"],
          stroke: common.stroke,
          "stroke-opacity": common["stroke-opacity"],
          "stroke-width": common["stroke-width"],
          opacity: common.opacity,
          x: absPos.x,
          y: absPos.y,
          "font-size": node.fontSize?.(),
          "font-family": node.fontFamily?.(),
          "text-anchor": node.align?.() === "center" ? "middle" : undefined,
          "dominant-baseline": "middle",
        };
        parts.push(`<text ${pushAttrs(attrs)}>${escapeXml(text)}</text>`);
      }
    };

    const walk = (node) => {
      const className = node.getClassName?.();
      if (className === "Group" || className === "Layer" || className === "Stage") {
        const children = node.getChildren?.() || [];
        children.forEach((child) => walk(child));
        return;
      }
      addShape(node);
    };

    walk(stage);
    parts.push("</svg>");
    return parts.join("");
  };

  const handleSimVideoDownload = async () => {
    if (simVideoLoading || !scene) return;
    setSimVideoLoading(true);
    try {
      const res = await fetch("/api/export_sim_video", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          scene,
          packedLabels,
          fontName: labelFontFamily,
          fontSize: labelFontSize,
        }),
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.error || `export failed: ${res.status}`);
      }
      const data = await res.json().catch(() => ({}));
      if (data?.name) {
        window.location = `/api/download_sim_video?name=${encodeURIComponent(data.name)}`;
      }
    } catch (err) {
      setError(err.message || String(err));
    } finally {
      setSimVideoLoading(false);
    }
  };

  const handleSimPlayToggle = () => {
    if (!simPlaying && simProgress >= 1) {
      setSimProgress(0);
    }
    setSimPlaying((v) => !v);
  };

  const injectSvgLabels = (svgText, labels, fontFamily, fontSize) => {
    try {
      const doc = new DOMParser().parseFromString(svgText, "image/svg+xml");
      const svg = doc.querySelector("svg");
      if (!svg) return svgText;
      Array.from(svg.querySelectorAll("text")).forEach((n) => n.remove());
      Object.values(labels || {}).forEach((lbl) => {
        const x = Number(lbl.x);
        const y = Number(lbl.y);
        if (!Number.isFinite(x) || !Number.isFinite(y)) return;
        const text = doc.createElementNS("http://www.w3.org/2000/svg", "text");
        text.setAttribute("x", String(x));
        text.setAttribute("y", String(y));
        text.setAttribute("fill", "#ffffff");
        text.setAttribute("font-family", fontFamily || "Arial");
        text.setAttribute("font-size", String(fontSize || 12));
        text.setAttribute("text-anchor", "middle");
        text.setAttribute("dominant-baseline", "middle");
        text.textContent = String(lbl.label ?? "");
        svg.appendChild(text);
      });
      return new XMLSerializer().serializeToString(svg);
    } catch {
      return svgText;
    }
  };

  const captureStageSvg = (ref, exportSize = null, layerVisibility = null) => {
    const stage = ref?.current;
    if (!stage) return "";
    const prevScale = stage.scale();
    const prevPos = stage.position();
    const prevVis = [];
    const applyVis = (name, visible) => {
      let nodes = stage.find(`.${name}`) || [];
      let list = nodes?.toArray ? nodes.toArray() : nodes;
      if (!list || list.length === 0) {
        nodes = stage.find((n) => (n.name && n.name() === name) || false) || [];
        list = nodes?.toArray ? nodes.toArray() : nodes;
      }
      (list || []).forEach((n) => {
        prevVis.push([n, n.visible()]);
        n.visible(visible);
      });
    };
    if (layerVisibility) {
      Object.entries(layerVisibility).forEach(([name, visible]) => applyVis(name, visible));
    }
    stage.scale({ x: 1, y: 1 });
    stage.position({ x: 0, y: 0 });
    stage.draw();
    const svg =
      typeof stage.toSVG === "function"
        ? stage.toSVG()
        : buildSvgFromStage(stage, exportSize);
    prevVis.forEach(([node, vis]) => node.visible(vis));
    stage.scale(prevScale);
    stage.position(prevPos);
    stage.draw();
    return svg;
  };

  const downloadStage = (ref, filename, exportSize = null) => {
    try {
      const svg = captureStageSvg(ref, exportSize);
      try {
        let suffix = "";
        if (showImages) suffix += "_image";
        if (showStroke) suffix += "_stroke";
        suffix += showLabels ? "_label" : "_nolabel";
        const nameWithSuffix = filename.replace(/\.svg$/i, `${suffix}.svg`);
        fetch("/api/save_konva_svg", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ name: nameWithSuffix, svg }),
        });
      } catch {
        // ignore save errors
      }
      // backend save only
    } catch {
      // ignore download errors
    }
  };

  useEffect(() => {
    if (!packedImageSrc) return;
    fetch(packedImageSrc)
      .then((res) => res.text())
      .then((text) => {
        const parsed = parsePackedSvg(text);
        setPackedFillPaths(parsed.fillPaths);
        setPackedBleedPaths(parsed.bleedPaths);
        if (!parsed.hasBleed) {
          setPackedBleedError("packed.svg missing bleed layer");
        } else {
          setPackedBleedError("");
        }
      })
      .catch(() => {
        setPackedFillPaths([]);
        setPackedBleedPaths([]);
        setPackedBleedError("packed.svg failed to load");
      });
  }, [packedImageSrc]);

  useEffect(() => {
    if (!packedImageSrc2) return;
    fetch(packedImageSrc2)
      .then((res) => res.text())
      .then((text) => {
        const parsed = parsePackedSvg(text);
        setPackedFillPaths2(parsed.fillPaths);
        setPackedBleedPaths2(parsed.bleedPaths);
        if (!parsed.hasBleed) {
          setPackedBleedError2("packed_page2.svg missing bleed layer");
        } else {
          setPackedBleedError2("");
        }
      })
      .catch(() => {
        setPackedFillPaths2([]);
        setPackedBleedPaths2([]);
        setPackedBleedError2("packed_page2.svg failed to load");
      });
  }, [packedImageSrc2]);

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
  }, [segs, nodes]);

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
  }, [borderSegments]);

  function offsetPoints(pts, dx, dy) {
    return (pts || []).map((p) => [p[0] + dx, p[1] + dy]);
  }

  const zoneColorMap = useMemo(() => {
    if (!scene?.region_colors || !scene?.zone_id) return {};
    const map = {};
    for (let i = 0; i < scene.zone_id.length; i++) {
      const zid = scene.zone_id[i];
      if (map[zid]) continue;
      const color = scene.region_colors[i];
      if (color) map[zid] = color;
    }
    return map;
  }, [scene]);

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
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scene, regionStageSize, region2StageSize, zoneStageSize, autoFit]);

  return (
    <div className="app">
      <div className="content">
        <div className="column-left">
          <div className="panel toolbar">
            <button onClick={loadScene}>Load</button>
            <button onClick={exportPdf}>Export PDF</button>
            <button
              onClick={() => {
                setSimProgress(0);
                setSimPlaying(false);
                setShowSim(true);
              }}
            >
              Simulate
            </button>
            <button
              className={edgeMode ? "active" : ""}
              onClick={() => {
                setEdgeMode((v) => !v);
                setEdgeCandidate(null);
              }}
            >
              Create Edge
            </button>
            <div className="toolbar-spacer" />
            {exportMsg ? <div className="meta">{exportMsg}</div> : null}
            {error ? <div className="error">{error}</div> : null}
          </div>

          <div className={`left ${sceneLoading ? "is-loading" : ""}`} ref={leftRef}>
          <div className="preview-header">
            <div className="preview-title">Source (Konva)</div>
            <div className="preview-controls">
              <button
                className="icon-button"
                title="Download"
                onClick={() =>
                  downloadStage(stageRef, "source-konva.svg", scene?.canvas || null)
                }
              >
                {"\u2193"}
              </button>
            </div>
          </div>
          <Stage
            width={stageSize.w}
            height={stageSize.h}
            draggable
            scaleX={scale}
            scaleY={scale}
            x={pos.x}
            y={pos.y}
            onWheel={handleWheel}
            onMouseMove={(e) => {
              if (!edgeMode) return;
              const stage = stageRef.current;
              const pointer = stage.getPointerPosition();
              if (!pointer) return;
              const world = {
                x: (pointer.x - pos.x) / scale,
                y: (pointer.y - pos.y) / scale,
              };
              const cand = findEdgeCandidate(world);
              setEdgeCandidate(cand);
            }}
            onMouseLeave={() => {
              if (edgeMode) setEdgeCandidate(null);
            }}
            onMouseDown={() => {
              if (!edgeMode || !edgeCandidate) return;
              const key = edgeKey(edgeCandidate.a, edgeCandidate.b);
              const segSet = new Set(segs.map(([a, b]) => edgeKey(a, b)));
              if (segSet.has(key)) return;
              const nextSegs = [...segs, [edgeCandidate.a, edgeCandidate.b]];
              setSegs(nextSegs);
              fetch("/api/save_svg", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ nodes, segs: nextSegs }),
              }).then(() => loadScene(false));
            }}
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
        {edgeCandidate ? (
          <Layer>
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
          </Layer>
        ) : null}
        <Layer>
          {nodes.map((n) => (
            <Circle
              key={`n-${n.id}`}
              x={n.x}
              y={n.y}
              radius={3 / scale}
              fill="red"
              strokeScaleEnabled={false}
              draggable={!edgeMode}
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
                fetch("/api/save_svg", {
                  method: "POST",
                  headers: { "Content-Type": "application/json" },
                  body: JSON.stringify({ nodes: merged.nodes, segs: merged.segs }),
                }).then(() => loadScene(false));
              }}
            />
          ))}
        </Layer>
          </Stage>
          {sceneLoading ? <div className="loading-overlay">Loading...</div> : null}
          <div className="left-debug">
            <div className="zone-count">
              Zones: {scene?.zone_id ? Math.max(...scene.zone_id) + 1 : 0}
            </div>
            <div className="zone-count">
              Debug:
              {scene?.debug
                ? ` raw=${scene.debug.polygons_raw || 0} kept=${scene.debug.polygons_final || 0} small=${scene.debug.polygons_removed_small || 0} largest=${scene.debug.polygons_removed_largest || 0} tri_keep=${scene.debug.tri_kept || 0} tri_small=${scene.debug.tri_removed_small || 0} tri_out=${scene.debug.tri_removed_outside || 0} packed=${scene.debug.packed_placed || 0}/${scene.debug.zones_total || 0}`
                : " n/a"}
            </div>
            <div className="zone-count">
              ZonePoly:
              {scene?.debug
                ? ` empty=${(scene.debug.zones_empty || []).length} hull=${(scene.debug.zones_convex_hull || []).length}`
                : " n/a"}
            </div>
          </div>
        </div>
        </div>
        <div className="right">
          <div className={`preview tall region-stage ${sceneLoading ? "is-loading" : ""}`} ref={regionWrapRef}>
            <div className="preview-header">
              <div className="preview-title">Packed (Konva)</div>
              <div className="preview-controls">
                <label className="checkbox">
                  <input
                    type="checkbox"
                    checked={showImages}
                    onChange={(e) => {
                      setShowImages(e.target.checked);
                    }}
                  />
                  Image
                </label>
                <label className="checkbox">
                  <input
                    type="checkbox"
                    checked={showStroke}
                    onChange={(e) => {
                      setShowStroke(e.target.checked);
                    }}
                  />
                  Stroke
                </label>
                <label className="checkbox">
                  <input
                    type="checkbox"
                    checked={showLabels}
                    onChange={(e) => {
                      setShowLabels(e.target.checked);
                    }}
                  />
                  Label
                </label>
                <label className="mini-input">
                  Font
                  <select
                    value={labelFontFamily}
                    onChange={(e) => setLabelFontFamily(e.target.value)}
                  >
                    <option value="Arial">Arial</option>
                    <option value="Helvetica">Helvetica</option>
                    <option value="Verdana">Verdana</option>
                    <option value="Tahoma">Tahoma</option>
                    <option value="Georgia">Georgia</option>
                    <option value="Times New Roman">Times New Roman</option>
                    <option value="Courier New">Courier New</option>
                  </select>
                </label>
                <label className="mini-input">
                  Size
                  <input
                    type="number"
                    min="4"
                    max="64"
                    value={labelFontSize}
                    onChange={(e) => setLabelFontSize(parseFloat(e.target.value || "12"))}
                  />
                </label>
                <button
                  className="icon-button"
                  title="Download"
                  onClick={() =>
                    downloadStage(
                      regionRef,
                      "packed-konva.svg",
                      scene?.canvas ? { w: scene.canvas.w, h: scene.canvas.h } : null
                    )
                  }
                >
                  {"\u2193"}
                </button>
              </div>
            </div>
            {scene ? (
              <Stage
                width={regionStageSize.w}
                height={regionStageSize.h}
                  draggable
                  scaleX={regionScale}
                  scaleY={regionScale}
                  x={regionPos.x}
                  y={regionPos.y}
                  onWheel={handleRegionWheel}
                ref={regionRef}
              >
                <Layer>
                  {scene?.canvas ? (
                    <>
                      <Rect
                        x={0}
                        y={0}
                        width={scene.canvas.w}
                        height={scene.canvas.h}
                        stroke="#ffffff"
                        strokeWidth={2 / regionScale}
                        listening={false}
                      />
                    </>
                  ) : null}
                </Layer>
                <Layer name="packed-image" visible={showImages}>
                  <>
                    <Group
                      x={(scene?.canvas?.w || 0) / 2}
                      y={(scene?.canvas?.h || 0) / 2}
                      offsetX={(scene?.canvas?.w || 0) / 2}
                      offsetY={(scene?.canvas?.h || 0) / 2}
                    >
                      {packedFillPaths.map((p, idx) => (
                        <Path
                          key={`fill-path-${idx}`}
                          data={p.d}
                          fill={p.fill}
                          strokeWidth={0}
                          listening={false}
                        />
                      ))}
                      {packedBleedPaths.map((p, idx) => (
                        <Path
                          key={`bleed-path-${idx}`}
                          data={p.d}
                          fill={p.fill}
                          strokeWidth={0}
                          listening={false}
                        />
                      ))}
                    </Group>
                    <Group
                      x={(scene?.canvas?.w || 0) / 2 + (scene?.canvas?.w || 0) + 40}
                      y={(scene?.canvas?.h || 0) / 2}
                      offsetX={(scene?.canvas?.w || 0) / 2}
                      offsetY={(scene?.canvas?.h || 0) / 2}
                    >
                      {packedFillPaths2.map((p, idx) => (
                        <Path
                          key={`fill-path-2-${idx}`}
                          data={p.d}
                          fill={p.fill}
                          strokeWidth={0}
                          listening={false}
                        />
                      ))}
                      {packedBleedPaths2.map((p, idx) => (
                        <Path
                          key={`bleed-path-2-${idx}`}
                          data={p.d}
                          fill={p.fill}
                          strokeWidth={0}
                          listening={false}
                        />
                      ))}
                    </Group>
                  </>
                </Layer>
                <Layer name="packed-stroke" visible={showStroke}>
                  {Object.entries(scene.zone_boundaries || {}).flatMap(([zid, paths]) => {
                    const bin =
                      scene?.placement_bin?.[zid] ?? scene?.placement_bin?.[parseInt(zid, 10)];
                    const page = bin === 1 ? 1 : 0;
                    const xOffset = page === 1 ? (scene?.canvas?.w || 0) + 40 : 0;
                      return (paths || []).map((p, i) => {
                        const shift = scene.zone_shift?.[zid] || scene.zone_shift?.[parseInt(zid, 10)];
                        if (!shift) return null;
                        const rot = scene.zone_rot?.[zid] ?? scene.zone_rot?.[parseInt(zid, 10)] ?? 0;
                        const center =
                          scene.zone_center?.[zid] || scene.zone_center?.[parseInt(zid, 10)] || [0, 0];
                        const tpts = transformPath(p, shift, rot, center);
                        const isSelected = String(zid) === String(selectedZoneId);
                        return (
                          <Line
                            key={`pz-outline-${zid}-${i}`}
                            points={toPoints(offsetPoints(tpts, xOffset, 0))}
                            stroke={isSelected ? "#ff3b30" : "#f5f6ff"}
                            strokeWidth={isSelected ? 3 : 1}
                            strokeScaleEnabled={false}
                            listening={false}
                          />
                        );
                      });
                  })}
                </Layer>
                <Layer name="packed-hit">
                  {Object.entries(scene.zone_boundaries || {}).flatMap(([zid, paths]) => {
                    const bin =
                      scene?.placement_bin?.[zid] ?? scene?.placement_bin?.[parseInt(zid, 10)];
                    const page = bin === 1 ? 1 : 0;
                    const xOffset = page === 1 ? (scene?.canvas?.w || 0) + 40 : 0;
                    return (paths || []).map((p, i) => {
                      const shift = scene.zone_shift?.[zid] || scene.zone_shift?.[parseInt(zid, 10)];
                      if (!shift) return null;
                      const rot = scene.zone_rot?.[zid] ?? scene.zone_rot?.[parseInt(zid, 10)] ?? 0;
                      const center =
                        scene.zone_center?.[zid] || scene.zone_center?.[parseInt(zid, 10)] || [0, 0];
                      const tpts = transformPath(p, shift, rot, center);
                      return (
                        <Line
                          key={`pz-hit-${zid}-${i}`}
                          points={toPoints(offsetPoints(tpts, xOffset, 0))}
                          stroke="rgba(0,0,0,0)"
                          strokeWidth={8 / regionScale}
                          strokeScaleEnabled={false}
                          onClick={() => setSelectedZoneId(String(zid))}
                        />
                      );
                    });
                  })}
                </Layer>
                <Layer name="packed-label" visible={showLabels}>
                  <Group>
                    {packedLabels.map((lbl) => {
                      const bin =
                        scene?.placement_bin?.[lbl.zid] ??
                        scene?.placement_bin?.[parseInt(lbl.zid, 10)];
                      const page = bin === 1 ? 1 : 0;
                      const xOffset = page === 1 ? (scene?.canvas?.w || 0) + 40 : 0;
                      const size = Math.max(labelFontSize / regionScale, 6 / regionScale);
                      const metrics = measureText(lbl.label, size, labelFontFamily);
                      const isSelected = String(lbl.zid) === String(selectedZoneId);
                      return (
                        <Text
                          key={lbl.id}
                          x={lbl.x + xOffset}
                          y={lbl.y}
                          text={lbl.label}
                          fill={isSelected ? "#ff3b30" : "#ffffff"}
                          stroke="rgba(0,0,0,0.5)"
                          strokeWidth={1 / regionScale}
                          fontSize={size}
                          fontFamily={labelFontFamily}
                          align="center"
                          verticalAlign="middle"
                          offsetX={metrics.width / 2}
                          offsetY={metrics.height / 2}
                          draggable
                          onDragEnd={(e) => {
                            const next = packedLabels.map((p) =>
                              p.id === lbl.id ? { ...p, x: e.target.x(), y: e.target.y() } : p
                            );
                            setPackedLabels(next);
                            try {
                              fetch("/api/packed_labels", {
                                method: "POST",
                                headers: { "Content-Type": "application/json" },
                                body: JSON.stringify({
                                  [String(lbl.zid)]: {
                                    x: e.target.x(),
                                    y: e.target.y(),
                                    label: lbl.label,
                                  },
                                }),
                              });
                            } catch {
                              // ignore storage errors
                            }
                          }}
                        />
                      );
                    })}
                  </Group>
                </Layer>
              </Stage>
            ) : null}
            {sceneLoading ? <div className="loading-overlay">Loading...</div> : null}
            {packedBleedError ? <div className="error">{packedBleedError}</div> : null}
            {packedBleedError2 ? <div className="error">{packedBleedError2}</div> : null}
          </div>
          <div className="preview-row">
            <div className={`preview half ${sceneLoading ? "is-loading" : ""}`} ref={region2WrapRef}>
              <div className="preview-header">
                <div className="preview-title">Region (Konva)</div>
                <div className="preview-controls">
                  <button
                    className="icon-button"
                    title="Download"
                    onClick={() =>
                      downloadStage(region2Ref, "region-konva.svg", scene?.canvas || null)
                    }
                  >
                    {"\u2193"}
                  </button>
                </div>
              </div>
              {scene ? (
                <Stage
                  width={region2StageSize.w}
                  height={region2StageSize.h}
                  draggable
                  scaleX={region2Scale}
                  scaleY={region2Scale}
                  x={region2Pos.x}
                  y={region2Pos.y}
                  onWheel={handleRegion2Wheel}
                  ref={region2Ref}
                >
                  <Layer>
                    {scene?.canvas ? (
                      <Rect
                        x={0}
                        y={0}
                        width={scene.canvas.w}
                        height={scene.canvas.h}
                        stroke="#ffffff"
                        strokeWidth={2 / region2Scale}
                        listening={false}
                      />
                    ) : null}
                  </Layer>
                  <Layer>
                    {scene.regions.map((poly, idx) => (
                      <Line
                        key={`r2-${idx}`}
                        points={toPoints(poly)}
                        closed
                        stroke="#f5f6ff"
                        fill="#bbb"
                        strokeWidth={1 / region2Scale}
                        strokeScaleEnabled={false}
                      />
                    ))}
                  </Layer>
                </Stage>
              ) : null}
              {sceneLoading ? <div className="loading-overlay">Loading...</div> : null}
            </div>
            <div className={`preview half ${sceneLoading ? "is-loading" : ""}`} ref={zoneWrapRef}>
              <div className="preview-header">
                <div className="preview-title">Zone (Konva)</div>
                <div className="preview-controls">
                  <button
                    className="icon-button"
                    title="Download"
                    onClick={() =>
                      downloadStage(zoneRef, "zone-konva.svg", scene?.canvas || null)
                    }
                  >
                    {"\u2193"}
                  </button>
                </div>
              </div>
              {scene ? (
                <Stage
                  width={zoneStageSize.w}
                  height={zoneStageSize.h}
                  draggable
                  scaleX={zoneScale}
                  scaleY={zoneScale}
                  x={zonePos.x}
                  y={zonePos.y}
                  onWheel={handleZoneWheel}
                  ref={zoneRef}
                >
                  <Layer>
                    {scene?.canvas ? (
                      <Rect
                        x={0}
                        y={0}
                        width={scene.canvas.w}
                        height={scene.canvas.h}
                        stroke="#ffffff"
                        strokeWidth={2 / zoneScale}
                        listening={false}
                      />
                    ) : null}
                  </Layer>
                  <Layer name="zone-image" visible={showImages}>
                    {scene.region_colors
                      ? scene.regions.map((poly, idx) => (
                          <Line
                            key={`zf-${idx}`}
                            points={toPoints(poly)}
                            closed
                            fill={scene.region_colors[idx]}
                            strokeScaleEnabled={false}
                          />
                        ))
                      : null}
                  </Layer>
                  <Layer name="zone-stroke">
                    {Object.entries(scene.zone_boundaries || {}).flatMap(([zid, paths]) =>
                      paths.map((p, i) => (
                        <Line
                          key={`zb2-${zid}-${i}`}
                          points={toPoints(p)}
                          stroke={String(zid) === String(selectedZoneId) ? "#ff3b30" : "#f5f6ff"}
                          strokeWidth={String(zid) === String(selectedZoneId) ? 3 : 1}
                          strokeScaleEnabled={false}
                        />
                      ))
                    )}
                  </Layer>
                  <Layer name="zone-label" visible={showLabels}>
                    {Object.values(scene.zone_labels || {}).map((lbl) => {
                      const selectedShuffle =
                        scene?.zone_label_map?.[selectedZoneId] ??
                        scene?.zone_label_map?.[parseInt(selectedZoneId, 10)];
                      const targetLabel = selectedShuffle != null ? selectedShuffle : selectedZoneId;
                      const isSelected = String(lbl.label) === String(targetLabel);
                      const size = Math.max(labelFontSize / zoneScale, 6 / zoneScale);
                      const metrics = measureText(lbl.label, size, labelFontFamily);
                      return (
                        <Text
                          key={`zl-${lbl.label}`}
                          x={lbl.x}
                          y={lbl.y}
                          text={`${lbl.label}`}
                          fill={isSelected ? "#ff3b30" : "#ffffff"}
                          fontSize={size}
                          fontFamily={labelFontFamily}
                          align="center"
                          verticalAlign="middle"
                          offsetX={metrics.width / 2}
                          offsetY={metrics.height / 2}
                        />
                      );
                    })}
                  </Layer>
                </Stage>
              ) : null}
              {sceneLoading ? <div className="loading-overlay">Loading...</div> : null}
            </div>
          </div>
        </div>
      </div>
      {exportPdfLoading || exportPdfInfo ? (
        <div className="modal-backdrop">
          <div className="modal">
            <div className="modal-title">
              {exportPdfLoading ? "Creating PDF..." : "Successful created PDF !"}
            </div>
            {!exportPdfLoading && exportPdfInfo ? (
              <div className="modal-actions">
                <button
                  className="btn ghost"
                  onClick={() => setExportPdfInfo(null)}
                >
                  Cancel
                </button>
                <button
                  className="btn"
                  onClick={() => {
                    window.location = `/api/download_pdf?name=${encodeURIComponent(
                      exportPdfInfo.name
                    )}`;
                    setExportPdfInfo(null);
                  }}
                >
                  Download PDF
                </button>
              </div>
            ) : null}
          </div>
        </div>
      ) : null}
      {showSim ? (
        <div className="modal-backdrop">
          <div className="modal sim-modal">
            <button className="modal-close" onClick={() => setShowSim(false)}>
              X
            </button>
            <div className="modal-title">Simulate</div>
            <div className="sim-status">
              {simActiveLabel ? `Moving index: ${simActiveLabel}` : "Moving index: -"}
            </div>
            <div className="sim-body" ref={simWrapRef}>
              {simStage}
            </div>
            <div className="sim-controls">
              <button className="icon-button" onClick={handleSimPlayToggle}>
                {simPlaying ? (
                  <svg viewBox="0 0 20 20" width="16" height="16" aria-hidden="true">
                    <rect x="4" y="3" width="4" height="14" fill="currentColor" />
                    <rect x="12" y="3" width="4" height="14" fill="currentColor" />
                  </svg>
                ) : (
                  <svg viewBox="0 0 20 20" width="16" height="16" aria-hidden="true">
                    <polygon points="6,4 16,10 6,16" fill="currentColor" />
                  </svg>
                )}
              </button>
              <input
                type="range"
                min="0"
                max="1"
                step="0.001"
                value={simProgress}
                onChange={(e) => setSimProgress(parseFloat(e.target.value))}
              />
              <button
                className="btn"
                onClick={handleSimVideoDownload}
                disabled={simVideoLoading}
              >
                {simVideoLoading ? "Creating..." : "Download GIF"}
              </button>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}
