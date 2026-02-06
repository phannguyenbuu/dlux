import { bboxFromPts } from "./geometry";

export const parsePoints = (str) => {
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

export const parseSvgSize = (svgText) => {
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
  return { w: Number.isFinite(w) ? w : 1000, h: Number.isFinite(h) ? h : 1000 };
};

export const getSvgHref = (el) =>
  el.getAttribute("href") || el.getAttribute("xlink:href") || "";

export const svgToDataUrl = (svgText) =>
  `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svgText)}`;

export const decodeSvgDataUrl = (src) => {
  const parts = src.split(",");
  if (parts.length < 2) return src;
  return decodeURIComponent(parts.slice(1).join(","));
};

export const applySvgFill = (svgText, color) => {
  try {
    const doc = new DOMParser().parseFromString(svgText, "image/svg+xml");
    const tags = ["path", "rect", "circle", "ellipse", "polygon", "polyline"];
    tags.forEach((tag) => {
      doc.querySelectorAll(tag).forEach((el) => {
        const fill = el.getAttribute("fill");
        if (!fill || fill === "none" || fill === "transparent") return;
        el.setAttribute("fill", color);
      });
    });
    return doc.documentElement.outerHTML;
  } catch {
    return svgText;
  }
};

export const parseOverlayItems = (svgText) => {
  const doc = new DOMParser().parseFromString(svgText, "image/svg+xml");
  const nodes = Array.from(
    doc.querySelectorAll('image[data-role="overlay"],image[data-overlay="true"],image.overlay')
  );
  return nodes.map((node) => {
    const src = getSvgHref(node);
    const x = parseFloat(node.getAttribute("data-x") || node.getAttribute("x") || "0");
    const y = parseFloat(node.getAttribute("data-y") || node.getAttribute("y") || "0");
    const width = parseFloat(node.getAttribute("data-width") || node.getAttribute("width") || "0");
    const height = parseFloat(
      node.getAttribute("data-height") || node.getAttribute("height") || "0"
    );
    const scaleX = parseFloat(node.getAttribute("data-scale-x") || "1");
    const scaleY = parseFloat(node.getAttribute("data-scale-y") || "1");
    const rotation = parseFloat(node.getAttribute("data-rotation") || "0");
    const zid = node.getAttribute("data-zid");
    const raw = node.getAttribute("data-raw") || "";
    return {
      id: node.getAttribute("data-id") || `overlay-${Math.random().toString(36).slice(2)}`,
      src,
      x,
      y,
      width,
      height,
      scaleX,
      scaleY,
      rotation,
      zid: zid != null ? String(zid) : null,
      rawSvg: raw,
    };
  });
};

export const buildSegmentsFromSvg = (svgText) => {
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
    const seg = [[x1, y1], [x2, y2]];
    segments.push(seg);
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

export const parsePackedSvg = (text) => {
  const doc = new DOMParser().parseFromString(text, "image/svg+xml");
  const fill = doc.querySelector("g#fill");
  const bleed = doc.querySelector("g#bleed");
  const parsePaths = (node) =>
    Array.from(node?.querySelectorAll("path") || []).map((p) => ({
      d: p.getAttribute("d") || "",
      fill: p.getAttribute("fill") || "#000000",
    }));
  const fillPaths = parsePaths(fill).filter((p) => p.d);
  const bleedPaths = parsePaths(bleed).filter((p) => p.d);
  return { fillPaths, bleedPaths, hasBleed: bleedPaths.length > 0 };
};

export const calcBoundsFromSegments = (segments) => {
  const pts = segments.flat();
  return bboxFromPts(pts);
};
