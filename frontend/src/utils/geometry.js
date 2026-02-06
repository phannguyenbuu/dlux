export const toPoints = (pts) => pts.flatMap((p) => [p[0], p[1]]);

export const rotatePt = (pt, angleDeg, cx, cy) => {
  if (!angleDeg) return pt;
  const ang = (angleDeg * Math.PI) / 180;
  const c = Math.cos(ang);
  const s = Math.sin(ang);
  const x = pt[0] - cx;
  const y = pt[1] - cy;
  return [cx + x * c - y * s, cy + x * s + y * c];
};

export const transformPath = (pts, shift, rot, center) => {
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

export const bboxFromPts = (pts) => {
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

export const pointInPoly = (pt, poly) => {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const xi = poly[i][0];
    const yi = poly[i][1];
    const xj = poly[j][0];
    const yj = poly[j][1];
    const intersect =
      yi > pt[1] !== yj > pt[1] &&
      pt[0] < ((xj - xi) * (pt[1] - yi)) / (yj - yi + 0.0) + xi;
    if (intersect) inside = !inside;
  }
  return inside;
};

export const pointSegDist = (pt, a, b) => {
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

export const pointInPolyWithOffset = (pt, poly, offset) => {
  if (pointInPoly(pt, poly)) return true;
  for (let i = 0; i < poly.length; i++) {
    const a = poly[i];
    const b = poly[(i + 1) % poly.length];
    if (pointSegDist(pt, a, b) <= offset) return true;
  }
  return false;
};

export const segmentIntersect = (a1, a2, b1, b2) => {
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
  return [x1 + t * dx12, y1 + t * dy12];
};

export const edgeKey = (a, b) => (a < b ? `${a}-${b}` : `${b}-${a}`);

export const offsetPoints = (pts, dx, dy) => pts.map((p) => [p[0] + dx, p[1] + dy]);

export const lerp = (a, b, t) => a + (b - a) * t;
