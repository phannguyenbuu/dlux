export const snapNodes = (segments, snap) => {
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

  nodes.forEach((n, i) => {
    if (nodeCnt[i]) {
      n.x = nodeSum[i][0] / nodeCnt[i];
      n.y = nodeSum[i][1] / nodeCnt[i];
    }
  });

  return { nodes, segs };
};

export const calcBounds = (polys) => {
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

export const calcBoundsFromLines = (linesDict) => {
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

export const mergeNodesIfClose = (nodes, segs, movedId, snap) => {
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
      n.id === targetId ? { ...n, x: (n.x + moved.x) / 2, y: (n.y + moved.y) / 2 } : n
    );

  const remap = new Map();
  merged.forEach((n, idx) => remap.set(n.id, idx));
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

export const segmentIntersectParam = (a1, a2, b1, b2) => {
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

export const splitAtIntersections = (segments) => {
  const splits = segments.map(() => [0, 1]);
  for (let i = 0; i < segments.length; i++) {
    for (let j = i + 1; j < segments.length; j++) {
      const a = segments[i];
      const b = segments[j];
      const inter = segmentIntersectParam(a[0], a[1], b[0], b[1]);
      if (!inter) continue;
      splits[i].push(inter.t);
      splits[j].push(inter.u);
    }
  }
  const out = [];
  for (let i = 0; i < segments.length; i++) {
    const ts = Array.from(new Set(splits[i].map((v) => Math.max(0, Math.min(1, v)))))
      .sort((a, b) => a - b);
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

export const segmentWouldIntersect = (nodes, segs, aIdx, bIdx) => {
  if (!nodes[aIdx] || !nodes[bIdx]) return true;
  const a1 = [nodes[aIdx].x, nodes[aIdx].y];
  const a2 = [nodes[bIdx].x, nodes[bIdx].y];
  for (const [s0, s1] of segs) {
    if (s0 === aIdx || s1 === aIdx || s0 === bIdx || s1 === bIdx) continue;
    const b1 = [nodes[s0].x, nodes[s0].y];
    const b2 = [nodes[s1].x, nodes[s1].y];
    const inter = segmentIntersectParam(a1, a2, b1, b2);
    if (!inter) continue;
    if (inter.t > 1e-6 && inter.t < 1 - 1e-6 && inter.u > 1e-6 && inter.u < 1 - 1e-6) {
      return true;
    }
  }
  return false;
};

export const findEdgeCandidate = (nodes, segs, worldPt, edgeKey) => {
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
      if (segmentWouldIntersect(nodes, segs, i, j)) continue;
      bestDist = d;
      best = { a: i, b: j };
    }
  }
  return best;
};

export const findExistingEdgeCandidate = (nodes, segs, worldPt, pointSegDist) => {
  if (!nodes.length || !segs.length) return null;
  const EDGE_HOVER_DIST = 10;
  let best = null;
  let bestDist = Infinity;
  segs.forEach(([a, b], idx) => {
    const na = nodes[a];
    const nb = nodes[b];
    if (!na || !nb) return;
    const d = pointSegDist([worldPt.x, worldPt.y], [na.x, na.y], [nb.x, nb.y]);
    if (d <= EDGE_HOVER_DIST && d < bestDist) {
      bestDist = d;
      best = { a, b, idx };
    }
  });
  return best;
};

export const pruneIsolatedNodes = (nextNodes, nextSegs) => {
  const connected = new Set();
  nextSegs.forEach(([a, b]) => {
    connected.add(a);
    connected.add(b);
  });
  const remap = new Map();
  const kept = [];
  nextNodes.forEach((n, idx) => {
    if (connected.has(idx)) {
      remap.set(idx, kept.length);
      kept.push({ ...n, id: kept.length });
    }
  });
  const remappedSegs = nextSegs
    .map(([a, b]) => [remap.get(a), remap.get(b)])
    .filter(([a, b]) => a != null && b != null && a !== b);
  return { nodes: kept, segs: remappedSegs };
};
