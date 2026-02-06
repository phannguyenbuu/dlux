import Konva from "konva";

export const measureText = (text, fontSize, fontFamily) => {
  const size = Number.isFinite(fontSize) ? fontSize : 12;
  const family = fontFamily || "Arial";
  if (Konva?.Util?.getTextWidth) {
    const width = Konva.Util.getTextWidth(text || "", size, family);
    return { width, height: size };
  }
  const width = (text ? text.length : 0) * size * 0.6;
  return { width, height: size };
};

const escapeXml = (value) =>
  String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&apos;");

const matrixToAttr = (m) => {
  const [a, b, c, d, e, f] = m.map((v) => (Number.isFinite(v) ? v : 0));
  return `matrix(${a} ${b} ${c} ${d} ${e} ${f})`;
};

export const buildSvgFromStage = (stage, exportSize = null) => {
  const width = exportSize?.w || stage.width();
  const height = exportSize?.h || stage.height();
  const parts = [
    `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">`,
  ];
  const pushAttrs = (attrs) => {
    const out = [];
    Object.entries(attrs).forEach(([k, v]) => {
      if (v == null || v === "") return;
      out.push(`${k}="${escapeXml(v)}"`);
    });
    return out.join(" ");
  };
  const addShape = (node) => {
    const transform = node.getAbsoluteTransform?.();
    const matrix = transform ? matrixToAttr(transform.getMatrix()) : "";
    const strokeScaleEnabled =
      typeof node.strokeScaleEnabled === "function" ? node.strokeScaleEnabled() : true;
    const common = {
      transform: matrix || undefined,
      opacity: node.opacity?.(),
      stroke: node.stroke?.() || undefined,
      "stroke-width": node.strokeWidth?.(),
      "stroke-linecap": node.lineCap?.(),
      "stroke-linejoin": node.lineJoin?.(),
      fill: node.fill?.() || undefined,
      "fill-rule": node.fillRule?.() || undefined,
      "stroke-scale": strokeScaleEnabled ? "true" : "false",
    };
    const className = node.getClassName?.();
    if (className === "Line") {
      const pts = node.points?.() || [];
      const pairs = [];
      for (let i = 0; i + 1 < pts.length; i += 2) {
        pairs.push(`${pts[i]} ${pts[i + 1]}`);
      }
      const closed = node.closed?.();
      const tag = closed ? "polygon" : "polyline";
      const attrs = {
        ...common,
        points: pairs.join(" "),
      };
      parts.push(`<${tag} ${pushAttrs(attrs)} />`);
      return;
    }
    if (className === "Path") {
      const d = node.data?.();
      const attrs = { ...common, d };
      parts.push(`<path ${pushAttrs(attrs)} />`);
      return;
    }
    if (className === "Rect") {
      const w = node.width?.();
      const h = node.height?.();
      const attrs = { ...common, x: 0, y: 0, width: w, height: h };
      parts.push(`<rect ${pushAttrs(attrs)} />`);
      return;
    }
    if (className === "Circle") {
      const r = node.radius?.();
      const attrs = { ...common, cx: 0, cy: 0, r };
      parts.push(`<circle ${pushAttrs(attrs)} />`);
      return;
    }
    if (className === "Image") {
      const img = node.image?.();
      const src = img?.src;
      const w = node.width?.();
      const h = node.height?.();
      const attrs = { ...common, x: 0, y: 0, width: w, height: h, href: src };
      parts.push(`<image ${pushAttrs(attrs)} />`);
      return;
    }
    if (className === "Text") {
      const text = node.text?.();
      const absPos = node.getAbsolutePosition?.() || { x: 0, y: 0 };
      const attrs = {
        ...common,
        x: 0,
        y: 0,
        "font-size": node.fontSize?.(),
        "font-family": node.fontFamily?.(),
        "text-anchor": "middle",
        "dominant-baseline": "middle",
      };
      parts.push(
        `<text ${pushAttrs(attrs)}>${escapeXml(text || "")}</text>`
      );
      return;
    }
  };
  const walk = (node) => {
    if (!node) return;
    const className = node.getClassName?.();
    if (className && className !== "Stage" && className !== "Layer" && className !== "Group") {
      addShape(node);
    }
    const children = node.getChildren?.() || [];
    children.forEach(walk);
  };
  walk(stage);
  parts.push("</svg>");
  return parts.join("");
};

export const captureStageSvg = (ref, exportSize = null, layerVisibility = null) => {
  const stage = ref?.current;
  if (!stage) return "";
  const prevScale = stage.scale();
  const prevPos = stage.position();
  const prevVis = [];
  const applyVis = (name, visible) => {
    const layer = stage.findOne(`. ${name}`);
    if (layer) {
      prevVis.push([layer, layer.visible()]);
      layer.visible(visible);
    }
  };
  if (layerVisibility) {
    Object.entries(layerVisibility).forEach(([name, vis]) => applyVis(name, vis));
  }
  stage.scale({ x: 1, y: 1 });
  stage.position({ x: 0, y: 0 });
  stage.batchDraw();
  const svg = buildSvgFromStage(stage, exportSize);
  stage.scale(prevScale);
  stage.position(prevPos);
  prevVis.forEach(([layer, vis]) => layer.visible(vis));
  stage.batchDraw();
  return svg;
};
