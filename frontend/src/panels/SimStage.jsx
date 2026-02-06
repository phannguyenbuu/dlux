import React from "react";
import { Layer, Line, Rect, Stage, Text } from "react-konva";

export default function SimStage({
  scene,
  simSize,
  simZoneIds,
  simZoneIndex,
  simLocalFor,
  packedLabels,
  labelFontFamily,
  labelFontSize,
  measureText,
  transformPath,
  toPoints,
  offsetPoints,
  lerp,
}) {
  if (!scene?.canvas) return null;
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
              const shift = scene.zone_shift?.[zid] || scene.zone_shift?.[parseInt(zid, 10)];
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
              const shift = scene.zone_shift?.[zid] || scene.zone_shift?.[parseInt(zid, 10)];
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
}
