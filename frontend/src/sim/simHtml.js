export const buildSimulateHtml = (data, packed, fontFamily, fontSize) => {
  const payload = {
    ...data,
    packed_labels: packed || [],
    font_family: fontFamily,
    font_size: fontSize,
  };
  return `<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Simulation</title>
  <script src="https://unpkg.com/konva@9/konva.min.js"></script>
  <style>
    html, body { margin:0; padding:0; width:100%; height:100%; background:#0b0f1e; color:#fff; font-family: Arial, sans-serif; }
    #stageWrap { width:100%; height: calc(100% - 60px); display:flex; align-items:center; justify-content:center; }
    #controls { height:60px; display:flex; gap:12px; align-items:center; padding:0 16px; }
    #movingText { flex:1; text-align:center; font-size:18px; }
    #slider { flex:1; }
    button { background:#222a4b; color:#fff; border:1px solid #3b3f6a; padding:6px 12px; border-radius:8px; cursor:pointer; }
  </style>
</head>
<body>
  <div id="movingText">Moving index: -</div>
  <div id="stageWrap"></div>
  <div id="controls">
    <button id="playBtn">Play</button>
    <input id="slider" type="range" min="0" max="1" step="0.001" value="0"/>
  </div>
  <script>
    const data = ${JSON.stringify(payload)};
    const wrap = document.getElementById('stageWrap');
    const movingText = document.getElementById('movingText');
    const playBtn = document.getElementById('playBtn');
    const slider = document.getElementById('slider');
    const gap = 20;
    const move = 1;
    const hold = 0.2;
    const per = move + hold;
    const zoneIds = Object.keys(data.zone_boundaries || {});
    const getLabel = (zid) => {
      const lbl = data.zone_label_map?.[zid] ?? data.zone_label_map?.[parseInt(zid, 10)] ?? zid;
      const num = Number(lbl);
      return Number.isFinite(num) ? num : Number(zid) || 0;
    };
    zoneIds.sort((a, b) => getLabel(a) - getLabel(b));
    const total = zoneIds.length ? zoneIds.length * per : 1;
    const zoneIndex = {};
    zoneIds.forEach((zid, i) => { zoneIndex[String(zid)] = i; });
    const simLocalFor = (idx, t) => {
      if (idx == null || idx < 0) return 0;
      const dt = t - idx * per;
      if (dt <= 0) return 0;
      if (dt >= per) return 1;
      if (dt >= move) return 1;
      const x = dt / move;
      return 1 - Math.pow(1 - x, 3);
    };
    const rotatePt = (pt, angleDeg, cx, cy) => {
      const ang = angleDeg * Math.PI / 180;
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
    const toPoints = (pts) => pts.flatMap(p => [p[0], p[1]]);
    const offsetPoints = (pts, dx, dy) => pts.map(p => [p[0] + dx, p[1] + dy]);
    const measureText = (text, size, family) => {
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      ctx.font = size + 'px ' + family;
      const metrics = ctx.measureText(text || '');
      return { width: metrics.width, height: size };
    };
    const stage = new Konva.Stage({ container: 'stageWrap', width: 10, height: 10 });
    const layerFrame = new Konva.Layer();
    const layerPacked = new Konva.Layer();
    const layerZoneLabels = new Konva.Layer();
    const layerZoneStroke = new Konva.Layer();
    const layerMove = new Konva.Layer();
    const rectLeft = new Konva.Rect({ x: 0, y: 0, width: data.canvas.w, height: data.canvas.h, stroke: '#ffffff', strokeWidth: 1 });
    const rectRight = new Konva.Rect({ x: data.canvas.w + gap, y: 0, width: data.canvas.w, height: data.canvas.h, stroke: '#ffffff', strokeWidth: 1 });
    layerFrame.add(rectLeft, rectRight);
    const packedShapes = [];
    (data.regions || []).forEach((poly, idx) => {
      const zid = data.zone_id?.[idx];
      const zidKey = String(zid);
      const shift = data.zone_shift?.[zid] || data.zone_shift?.[parseInt(zid, 10)];
      const rot = data.zone_rot?.[zid] ?? data.zone_rot?.[parseInt(zid, 10)] ?? 0;
      const center = data.zone_center?.[zid] || data.zone_center?.[parseInt(zid, 10)] || [0,0];
      const packed = transformPath(poly, shift, rot, center);
      const shape = new Konva.Line({
        points: toPoints(packed),
        closed: true,
        fill: data.region_colors?.[idx] || '#333',
        stroke: 'none',
      });
      shape._zidKey = zidKey;
      layerPacked.add(shape);
      packedShapes.push(shape);
    });
    const packedLabels = [];
    (data.packed_labels || []).forEach((lbl) => {
      const size = Math.max((data.font_size || 12) * 0.5, 6);
      const metrics = measureText(lbl.label, size, data.font_family);
      const text = new Konva.Text({
        x: lbl.x,
        y: lbl.y,
        text: lbl.label,
        fill: '#ffffff',
        fontSize: size,
        fontFamily: data.font_family,
        align: 'center',
        verticalAlign: 'middle',
        offsetX: metrics.width / 2,
        offsetY: metrics.height / 2,
      });
      text._zidKey = String(lbl.zid);
      layerPacked.add(text);
      packedLabels.push(text);
    });
    Object.entries(data.zone_boundaries || {}).forEach(([zid, paths]) => {
      (paths || []).forEach((p) => {
        const shift = data.zone_shift?.[zid] || data.zone_shift?.[parseInt(zid, 10)];
        const rot = data.zone_rot?.[zid] ?? data.zone_rot?.[parseInt(zid, 10)] ?? 0;
        const center = data.zone_center?.[zid] || data.zone_center?.[parseInt(zid, 10)] || [0,0];
        const tpts = transformPath(p, shift, rot, center);
        const shape = new Konva.Line({
          points: toPoints(tpts),
          stroke: '#ffffff',
          strokeWidth: 1,
        });
        shape._zidKey = String(zid);
        layerZoneStroke.add(shape);
      });
    });
    const movingShapes = [];
    (data.regions || []).forEach((poly, idx) => {
      const zid = data.zone_id?.[idx];
      const zidKey = String(zid);
      const shift = data.zone_shift?.[zid] || data.zone_shift?.[parseInt(zid, 10)];
      const rot = data.zone_rot?.[zid] ?? data.zone_rot?.[parseInt(zid, 10)] ?? 0;
      const center = data.zone_center?.[zid] || data.zone_center?.[parseInt(zid, 10)] || [0,0];
      const src = transformPath(poly, shift, rot, center);
      const dst = offsetPoints(poly, data.canvas.w + gap, 0);
      const shape = new Konva.Line({
        points: toPoints(src),
        closed: true,
        fill: data.region_colors?.[idx] || '#333',
        stroke: '#ffffff',
        strokeWidth: 1,
      });
      shape._zidKey = zidKey;
      shape._src = src;
      shape._dst = dst;
      layerMove.add(shape);
      movingShapes.push(shape);
    });
    stage.add(layerFrame, layerPacked, layerZoneLabels, layerZoneStroke, layerMove);
    const resize = () => {
      const rect = wrap.getBoundingClientRect();
      const totalW = (data.canvas.w * 2) + gap;
      const totalH = data.canvas.h;
      const fitScale = Math.min(rect.width / totalW, rect.height / totalH) * 1.06;
      const offsetX = (rect.width - totalW * fitScale) / 2;
      const offsetY = (rect.height - totalH * fitScale) / 2;
      stage.width(rect.width);
      stage.height(rect.height);
      layerFrame.position({ x: offsetX, y: offsetY });
      layerPacked.position({ x: offsetX, y: offsetY });
      layerZoneStroke.position({ x: offsetX, y: offsetY });
      layerMove.position({ x: offsetX, y: offsetY });
      stage.scale({ x: fitScale, y: fitScale });
    };
    window.addEventListener('resize', resize);
    resize();
    let simProgress = 0;
    let lastTime = 0;
    let playing = false;
    const update = () => {
      const activeIdx = zoneIds.length ? Math.min(zoneIds.length - 1, Math.max(0, Math.floor((simProgress * total) / per))) : -1;
      const activeZid = activeIdx >= 0 ? zoneIds[activeIdx] : null;
      const activeLabel = activeZid != null ? (data.zone_label_map?.[activeZid] ?? data.zone_label_map?.[parseInt(activeZid, 10)] ?? activeZid) : '-';
      movingText.textContent = 'Moving index: ' + activeLabel;
      movingShapes.forEach((shape) => {
        const idx = zoneIndex[shape._zidKey] ?? 0;
        const local = simLocalFor(idx, simProgress * total);
        const pts = shape._src.map((sp, k) => {
          const dp = shape._dst[k] || sp;
          return [sp[0] + (dp[0] - sp[0]) * local, sp[1] + (dp[1] - sp[1]) * local];
        });
        shape.points(toPoints(pts));
      });
      stage.draw();
    };
    const tick = (now) => {
      const dt = (now - lastTime) / 1000;
      lastTime = now;
      if (playing) {
        simProgress += dt / total;
        if (simProgress > 1) simProgress = 0;
        slider.value = simProgress;
      }
      update();
      requestAnimationFrame(tick);
    };
    requestAnimationFrame((t) => {
      lastTime = t;
      tick(t);
    });
    playBtn.onclick = () => {
      playing = !playing;
      playBtn.textContent = playing ? 'Pause' : 'Play';
    };
    slider.oninput = (e) => {
      simProgress = parseFloat(e.target.value);
      update();
    };
  </script>
</body>
</html>`;
};
