export default function LeftDebug({ scene }) {
  return (
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
  );
}
