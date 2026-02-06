export const findZoneAtPoint = (pt, zoneScene, scene, pointInPoly) => {
  const source = zoneScene || scene;
  const zones = source?.zone_boundaries || {};
  for (const [zid, paths] of Object.entries(zones)) {
    for (const poly of paths || []) {
      if (pointInPoly([pt.x, pt.y], poly)) return String(zid);
    }
  }
  return null;
};

export const findZoneIdByLabel = (label, zoneScene, scene) => {
  const source = zoneScene || scene;
  if (!source) return String(label);
  const target = String(label);
  const map = source.zone_label_map || {};
  for (const [zid, mapped] of Object.entries(map)) {
    if (String(mapped) === target) return String(zid);
  }
  for (const [zid, info] of Object.entries(source.zone_labels || {})) {
    if (String(info?.label) === target) return String(zid);
  }
  return target;
};

export const transformOverlayToPacked = (item, scene, transformPath) => {
  if (!scene || !item || item.zid == null) return item;
  const zid = item.zid;
  const shift = scene.zone_shift?.[zid] || scene.zone_shift?.[parseInt(zid, 10)];
  const rot = scene.zone_rot?.[zid] ?? scene.zone_rot?.[parseInt(zid, 10)] ?? 0;
  const center = scene.zone_center?.[zid] || scene.zone_center?.[parseInt(zid, 10)] || [0, 0];
  const [pt] = transformPath([[item.x, item.y]], shift, rot, center);
  return {
    ...item,
    x: pt[0],
    y: pt[1],
    rotation: (item.rotation || 0) + (rot || 0),
  };
};
