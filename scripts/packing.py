from __future__ import annotations

import json
import os
import time
import xml.etree.ElementTree as ET
from math import ceil
from typing import Dict, List, Tuple

import cv2
import numpy as np
from rectpack import newPacker
from shapely.affinity import rotate as _srotate, translate as _stranslate
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

from . import config
from . import geometry
from . import svg_utils
from . import zones

try:
    from shapely.validation import make_valid
except Exception:  # pragma: no cover
    def make_valid(geom):
        return geom.buffer(0)


def _log_step(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


def _rotate_pts(pts: List[Tuple[float, float]], angle_deg: float, cx: float, cy: float) -> List[Tuple[float, float]]:
    if angle_deg == 0:
        return [(float(x), float(y)) for x, y in pts]
    ang = np.deg2rad(angle_deg)
    c = float(np.cos(ang))
    s = float(np.sin(ang))
    out = []
    for x, y in pts:
        rx = x - cx
        ry = y - cy
        out.append((cx + rx * c - ry * s, cy + rx * s + ry * c))
    return out


def _rotate_zone_transforms_180(
    zone_shift: Dict[int, Tuple[float, float]],
    zone_center: Dict[int, Tuple[float, float]],
    zone_rot: Dict[int, float],
    canvas: Tuple[int, int],
) -> None:
    w, h = canvas
    cx_canvas = w / 2.0
    cy_canvas = h / 2.0
    for zid, (dx, dy) in list(zone_shift.items()):
        cx, cy = zone_center.get(zid, (0.0, 0.0))
        ndx = (2 * cx_canvas) - (2 * cx) - float(dx)
        ndy = (2 * cy_canvas) - (2 * cy) - float(dy)
        zone_shift[zid] = (ndx, ndy)
        zone_rot[zid] = float(zone_rot.get(zid, 0.0)) + 180.0


def pack_regions(
    polys: List[List[Tuple[float, float]]],
    canvas: Tuple[int, int],
    allow_rotate: bool = True,
    angle_step: float = 5.0,
    grid_step: float = 5.0,
) -> Tuple[List[Tuple[int, int, int, int, bool]], List[int], List[Dict[str, float]]]:
    w, h = canvas
    pad = float(config.PADDING)
    bleed_offset = float(config.PACK_BLEED + config.PADDING) * float(config.DRAW_SCALE)
    x_min = config.PACK_MARGIN_X
    y_min = config.PACK_MARGIN_Y
    x_max = w - config.PACK_MARGIN_X
    y_max = h - config.PACK_MARGIN_Y

    if config.PACK_MODE == "fast":
        bboxes: List[Tuple[int, float, float, int, int]] = []
        rot_info: List[Dict[str, float]] = []
        for i, pts in enumerate(polys):
            poly = Polygon(pts)
            if poly.is_empty:
                x0 = y0 = 0.0
                x1 = y1 = 1.0
                angle = 0.0
                cx = cy = 0.0
            else:
                if bleed_offset > 0:
                    poly = poly.buffer(bleed_offset)
                cx, cy = float(poly.centroid.x), float(poly.centroid.y)
                angle = 0.0
                best_area = 1e18
                best_bounds = None
                if allow_rotate:
                    ang = 0.0
                    while ang < 180.0:
                        rpts = _rotate_pts(pts, ang, cx, cy)
                        xs = [p[0] for p in rpts]
                        ys = [p[1] for p in rpts]
                        x0t, y0t, x1t, y1t = min(xs), min(ys), max(xs), max(ys)
                        area = (x1t - x0t) * (y1t - y0t)
                        if area < best_area:
                            best_area = area
                            best_bounds = (x0t, y0t, x1t, y1t)
                            angle = ang
                        ang += angle_step
                if best_bounds is None:
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    x0t, y0t, x1t, y1t = min(xs), min(ys), max(xs), max(ys)
                    best_bounds = (x0t, y0t, x1t, y1t)
                x0, y0, x1, y1 = best_bounds
            bw = int(ceil((x1 - x0) + pad * 2))
            bh = int(ceil((y1 - y0) + pad * 2))
            bboxes.append((i, x0, y0, bw, bh))
            rot_info.append({"angle": angle, "cx": cx, "cy": cy, "minx": x0, "miny": y0})

        packer = newPacker(rotation=False)
        packer.add_bin(w - config.PACK_MARGIN_X * 2, h - config.PACK_MARGIN_Y * 2)
        for idx, _, _, bw, bh in bboxes:
            packer.add_rect(bw, bh, rid=idx)
        packer.pack()

        placements: List[Tuple[int, int, int, int, bool]] = [(-1, -1, 0, 0, False)] * len(polys)
        order: List[int] = []
        rects = list(packer.rect_list())
        if not rects:
            return placements, order, rot_info

        min_x = min(r[1] for r in rects)
        min_y = min(r[2] for r in rects)
        max_x = max(r[1] + r[3] for r in rects)
        max_y = max(r[2] + r[4] for r in rects)
        content_w = max_x - min_x
        content_h = max_y - min_y
        offset_x = config.PACK_MARGIN_X + max(0, (w - config.PACK_MARGIN_X * 2 - content_w) // 2)
        offset_y = config.PACK_MARGIN_Y + max(0, (h - config.PACK_MARGIN_Y * 2 - content_h) // 2)

        for _, x, y, pw, ph, rid in rects:
            orig = bboxes[rid]
            x0 = orig[1]
            y0 = orig[2]
            placements[rid] = (x + offset_x + pad - int(x0), y + offset_y + pad - int(y0), pw, ph, False)
            order.append(rid)

        return placements, order, rot_info

    items: List[Tuple[int, Polygon]] = []
    for i, pts in enumerate(polys):
        poly = Polygon(pts)
        if poly.is_empty:
            poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        if bleed_offset > 0:
            poly = poly.buffer(bleed_offset)
            if poly.is_empty:
                poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        poly = make_valid(poly)
        items.append((i, poly))

    items.sort(key=lambda t: t[1].area, reverse=True)

    placements: List[Tuple[int, int, int, int, bool]] = [(-1, -1, 0, 0, False)] * len(polys)
    rot_info: List[Dict[str, float]] = [
        {"angle": 0.0, "cx": 0.0, "cy": 0.0, "minx": 0.0, "miny": 0.0} for _ in polys
    ]
    placed_order: List[int] = []
    placed_polys: List[Polygon] = []

    for rid, poly in items:
        cx, cy = float(poly.centroid.x), float(poly.centroid.y)
        candidates = []
        if allow_rotate:
            ang = 0.0
            while ang < 180.0:
                rpoly = _srotate(poly, ang, origin=(cx, cy), use_radians=False)
                minx, miny, maxx, maxy = rpoly.bounds
                candidates.append((ang, rpoly, minx, miny, maxx, maxy))
                ang += angle_step
        else:
            minx, miny, maxx, maxy = poly.bounds
            candidates.append((0.0, poly, minx, miny, maxx, maxy))

        candidates.sort(key=lambda c: (c[4] - c[2]) * (c[5] - c[3]))

        placed = False
        for ang, rpoly, minx, miny, maxx, maxy in candidates:
            bw = (maxx - minx) + pad * 2
            bh = (maxy - miny) + pad * 2
            if bw > (x_max - x_min) or bh > (y_max - y_min):
                continue
            positions = []
            center_x = (x_min + x_max) / 2.0
            center_y = (y_min + y_max) / 2.0
            y = y_min
            while y + bh <= y_max:
                x = x_min
                while x + bw <= x_max:
                    px = x + bw / 2.0
                    py = y + bh / 2.0
                    positions.append((x, y, (px - center_x) ** 2 + (py - center_y) ** 2))
                    x += grid_step
                y += grid_step
            positions.sort(key=lambda p: p[2])
            for x, y, _ in positions:
                dx = x - minx + pad
                dy = y - miny + pad
                tpoly = _stranslate(rpoly, xoff=dx, yoff=dy)
                tpoly_buf = tpoly.buffer(pad)
                collision = False
                for p in placed_polys:
                    if tpoly_buf.intersects(p) and not tpoly_buf.touches(p):
                        collision = True
                        break
                if not collision:
                    placements[rid] = (int(dx), int(dy), int(bw), int(bh), False)
                    rot_info[rid] = {"angle": float(ang), "cx": cx, "cy": cy, "minx": minx, "miny": miny}
                    placed_order.append(rid)
                    placed_polys.append(tpoly_buf)
                    placed = True
                    break
            if placed:
                break

    return placements, placed_order, rot_info


def write_pack_log(
    polys: List[List[Tuple[float, float]]],
    placements: List[Tuple[int, int, int, int, bool]],
    out_path,
    canvas: Tuple[int, int],
) -> None:
    w, h = canvas
    visible = 0
    overflow = 0
    lines = ["packed_regions"]
    for rid, (dx, dy, bw, bh, rot) in enumerate(placements):
        x0 = dx
        y0 = dy
        x1 = dx + bw
        y1 = dy + bh
        if x1 <= 0 or y1 <= 0 or x0 >= w or y0 >= h:
            overflow += 1
        else:
            visible += 1
        lines.append(f"region={rid} x={dx} y={dy} w={bw} h={bh} rot={rot}")
    lines.append(f"packed_visible={visible} overflow={overflow}")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def build_bleed(group_mask: np.ndarray, canvas_fill: np.ndarray, border_thick: int) -> Tuple[np.ndarray, np.ndarray]:
    if border_thick <= 0:
        return group_mask, canvas_fill
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * border_thick + 1, 2 * border_thick + 1))
    dilated = cv2.dilate(group_mask, kernel)
    bleed = (dilated > 0).astype(np.uint8) * 255

    canvas_mask = (group_mask > 0).astype(np.uint8) * 255
    canvas_fill_masked = cv2.bitwise_and(canvas_fill, canvas_fill, mask=canvas_mask)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * border_thick + 1, 2 * border_thick + 1))
    dilated_final = cv2.dilate(group_mask, kernel2)
    bleed_mask = ((dilated_final > 0) & (group_mask == 0)).astype(np.uint8) * 255

    bleed_color = cv2.dilate(canvas_fill_masked, kernel2)
    return bleed_mask, bleed_color


def write_pack_png(
    polys: List[List[Tuple[float, float]]],
    zone_id: List[int],
    zone_order: List[int],
    zone_polys: List[List[Tuple[float, float]]],
    placements: List[Tuple[int, int, int, int, bool]],
    canvas: Tuple[int, int],
    colors: List[Tuple[int, int, int]],
    zone_geoms: Dict[int, BaseGeometry],
    zone_labels: Dict[int, int],
    region_labels: Dict[int, int],
    rot_info: List[Dict[str, float]],
) -> None:
    w, h = canvas
    hi_scale = config.DRAW_SCALE * 2
    img = np.zeros((int(h * hi_scale), int(w * hi_scale), 3), dtype=np.uint8)
    alpha = np.zeros((int(h * hi_scale), int(w * hi_scale)), dtype=np.uint8)
    zone_shift: Dict[int, Tuple[float, float]] = {}
    zone_rot: Dict[int, float] = {}
    zone_center: Dict[int, Tuple[float, float]] = {}
    for idx, zid in enumerate(zone_order):
        dx, dy, _, _, _ = placements[idx]
        zone_shift[zid] = (dx, dy)
        info = rot_info[idx] if idx < len(rot_info) else {"angle": 0.0, "cx": 0.0, "cy": 0.0}
        zone_rot[zid] = float(info.get("angle", 0.0))
        zone_center[zid] = (float(info.get("cx", 0.0)), float(info.get("cy", 0.0)))

    for rid, pts in enumerate(polys):
        zid = zone_id[rid]
        if zid not in zone_shift:
            continue
        dx, dy = zone_shift[zid]
        ang = zone_rot.get(zid, 0.0)
        cx, cy = zone_center.get(zid, (0.0, 0.0))
        rpts = _rotate_pts(pts, ang, cx, cy)
        pts_shifted = np.array(
            [[(p[0] + dx) * hi_scale, (p[1] + dy) * hi_scale] for p in rpts],
            dtype=np.int32,
        )
        color = colors[rid]
        cv2.fillPoly(img, [pts_shifted], color)
        cv2.fillPoly(alpha, [pts_shifted], 255)

    effective_bleed = int(round(config.PACK_BLEED * config.DRAW_SCALE))
    if effective_bleed > 0:
        base_img = img.copy()
        for zid, (dx, dy) in zone_shift.items():
            mask = np.zeros((int(h * hi_scale), int(w * hi_scale)), dtype=np.uint8)
            for rid, pts in enumerate(polys):
                if zone_id[rid] != zid:
                    continue
                ang = zone_rot.get(zid, 0.0)
                cx, cy = zone_center.get(zid, (0.0, 0.0))
                rpts = _rotate_pts(pts, ang, cx, cy)
                pts_shifted = np.array(
                    [[(p[0] + dx) * hi_scale, (p[1] + dy) * hi_scale] for p in rpts],
                    dtype=np.int32,
                )
                cv2.fillPoly(mask, [pts_shifted], 255)
            bleed_mask, bleed_color_img = build_bleed(mask, base_img, effective_bleed)
            img[bleed_mask > 0] = bleed_color_img[bleed_mask > 0]
            alpha[bleed_mask > 0] = 255

    img_out = cv2.resize(img, (int(w * config.DRAW_SCALE), int(h * config.DRAW_SCALE)), interpolation=cv2.INTER_AREA)
    alpha_out = cv2.resize(alpha, (int(w * config.DRAW_SCALE), int(h * config.DRAW_SCALE)), interpolation=cv2.INTER_AREA)
    img_out = cv2.rotate(img_out, cv2.ROTATE_180)
    alpha_out = cv2.rotate(alpha_out, cv2.ROTATE_180)
    rgba = cv2.merge([img_out[:, :, 0], img_out[:, :, 1], img_out[:, :, 2], alpha_out])
    cv2.imwrite(str(config.OUT_PACK_PNG), rgba)


def write_pack_svg(
    polys: List[List[Tuple[float, float]]],
    zone_id: List[int],
    zone_order: List[int],
    zone_polys: List[List[Tuple[float, float]]],
    placements: List[Tuple[int, int, int, int, bool]],
    canvas: Tuple[int, int],
    colors: List[Tuple[int, int, int]],
    rot_info: List[Dict[str, float]],
) -> None:
    w, h = canvas
    zone_shift: Dict[int, Tuple[float, float]] = {}
    zone_rot: Dict[int, float] = {}
    zone_center: Dict[int, Tuple[float, float]] = {}
    for idx, zid in enumerate(zone_order):
        if idx >= len(placements):
            continue
        dx, dy, _, _, _ = placements[idx]
        zone_shift[zid] = (dx, dy)
        info = rot_info[idx] if idx < len(rot_info) else {"angle": 0.0, "cx": 0.0, "cy": 0.0}
        zone_rot[zid] = float(info.get("angle", 0.0))
        zone_center[zid] = (float(info.get("cx", 0.0)), float(info.get("cy", 0.0)))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
    ]
    parts.append('<g id="fill">')
    for rid, pts in enumerate(polys):
        if rid >= len(zone_id):
            continue
        zid = zone_id[rid]
        if zid not in zone_shift:
            continue
        dx, dy = zone_shift[zid]
        ang = zone_rot.get(zid, 0.0)
        cx, cy = zone_center.get(zid, (0.0, 0.0))
        moved = [(p[0] + dx, p[1] + dy) for p in _rotate_pts(pts, ang, cx, cy)]
        d = "M " + " L ".join(f"{p[0]} {p[1]}" for p in moved) + " Z"
        b, g, r = colors[rid]
        parts.append(f'<path d="{d}" fill="rgb({r},{g},{b})" stroke="none"/>')
    parts.append("</g>")
    parts.append('<g id="outline">')
    for rid, pts in enumerate(polys):
        if rid >= len(zone_id):
            continue
        zid = zone_id[rid]
        if zid not in zone_shift:
            continue
        dx, dy = zone_shift[zid]
        ang = zone_rot.get(zid, 0.0)
        cx, cy = zone_center.get(zid, (0.0, 0.0))
        moved = [(p[0] + dx, p[1] + dy) for p in _rotate_pts(pts, ang, cx, cy)]
        d = "M " + " L ".join(f"{p[0]} {p[1]}" for p in moved) + " Z"
        parts.append(f'<path d="{d}" fill="none" stroke="#000" stroke-width="1"/>')
    parts.append("</g></svg>")
    config.OUT_PACK_SVG.write_text("".join(parts), encoding="utf-8")


def write_pack_outline_png(
    polys: List[List[Tuple[float, float]]],
    zone_id: List[int],
    zone_order: List[int],
    zone_polys: List[List[Tuple[float, float]]],
    placements: List[Tuple[int, int, int, int, bool]],
    canvas: Tuple[int, int],
    zone_geoms: Dict[int, BaseGeometry],
    rot_info: List[Dict[str, float]],
) -> None:
    w, h = canvas
    hi_scale = config.DRAW_SCALE * 2
    img = np.full((int(h * hi_scale), int(w * hi_scale), 3), 255, dtype=np.uint8)
    zone_shift: Dict[int, Tuple[float, float]] = {}
    zone_rot: Dict[int, float] = {}
    zone_center: Dict[int, Tuple[float, float]] = {}
    for idx, zid in enumerate(zone_order):
        dx, dy, _, _, _ = placements[idx]
        zone_shift[zid] = (dx, dy)
        info = rot_info[idx] if idx < len(rot_info) else {"angle": 0.0, "cx": 0.0, "cy": 0.0}
        zone_rot[zid] = float(info.get("angle", 0.0))
        zone_center[zid] = (float(info.get("cx", 0.0)), float(info.get("cy", 0.0)))

    for rid, pts in enumerate(polys):
        zid = zone_id[rid]
        if zid not in zone_shift:
            continue
        dx, dy = zone_shift[zid]
        ang = zone_rot.get(zid, 0.0)
        cx, cy = zone_center.get(zid, (0.0, 0.0))
        rpts = _rotate_pts(pts, ang, cx, cy)
        pts_shifted = np.array(
            [[(p[0] + dx) * hi_scale, (p[1] + dy) * hi_scale] for p in rpts],
            dtype=np.int32,
        )
        cv2.polylines(img, [pts_shifted], True, (0, 0, 0), 1, cv2.LINE_AA)

    img_out = cv2.resize(img, (int(w * config.DRAW_SCALE), int(h * config.DRAW_SCALE)), interpolation=cv2.INTER_AREA)
    img_out = cv2.rotate(img_out, cv2.ROTATE_180)
    cv2.imwrite(str(config.OUT_PACK_OUTLINE_PNG), img_out)


def compute_scene(svg_path, snap: float, render_packed_png: bool = False) -> Dict:
    config._apply_pack_env()
    regions, polys, canvas, debug = geometry.build_regions_from_svg(svg_path, snap_override=snap)
    zone_id = zones.build_zones(polys, config.TARGET_ZONES)
    zone_id, zone_members = zones._remap_zones_by_area(polys, zone_id)
    zone_boundaries = zones.build_zone_boundaries(polys, zone_id)
    zone_geoms = zones.build_zone_geoms(polys, zone_id)
    zone_polys, zone_order, zone_poly_debug = zones.build_zone_polys(polys, zone_id)
    placements, _, rot_info = pack_regions(
        zone_polys,
        canvas,
        allow_rotate=True,
        angle_step=config.PACK_ANGLE_STEP,
        grid_step=config.PACK_GRID_STEP,
    )

    zone_ids = sorted(zone_geoms.keys())
    rng = np.random.default_rng(42)
    shuffled = zone_ids.copy()
    rng.shuffle(shuffled)
    zone_label_map = {z: idx + 1 for idx, z in enumerate(shuffled)}
    zone_labels = {}
    for zid, geom in zone_geoms.items():
        members = zone_members.get(zid, [])
        if members:
            rid0 = members[0]
            c = Polygon(polys[rid0]).centroid
            lx, ly = float(c.x), float(c.y)
        else:
            lx, ly = zones._label_pos_for_zone(geom)
        zone_labels[str(zid)] = {"x": lx, "y": ly, "label": zone_label_map.get(zid, zid)}

    region_labels = {}
    for rid, pts in enumerate(polys):
        poly = Polygon(pts)
        if poly.is_empty:
            continue
        region_labels[str(rid)] = {
            "x": float(poly.centroid.x),
            "y": float(poly.centroid.y),
            "label": rid,
            "zone": zone_id[rid] if rid < len(zone_id) else -1,
        }

    zone_shift: Dict[int, Tuple[float, float]] = {}
    zone_rot: Dict[int, float] = {}
    zone_center: Dict[int, Tuple[float, float]] = {}
    for idx, zid in enumerate(zone_order):
        if idx >= len(placements):
            continue
        dx, dy, _, _, _ = placements[idx]
        zone_shift[zid] = (dx, dy)
        info = rot_info[idx] if idx < len(rot_info) else {"angle": 0.0, "cx": 0.0, "cy": 0.0}
        zone_rot[zid] = float(info.get("angle", 0.0))
        zone_center[zid] = (float(info.get("cx", 0.0)), float(info.get("cy", 0.0)))

    _rotate_zone_transforms_180(zone_shift, zone_center, zone_rot, canvas)

    colors, _ = geometry.compute_region_colors(polys, canvas)
    region_colors = [f"#{r:02x}{g:02x}{b:02x}" for (b, g, r) in colors]

    missing = [z for z in zone_order if z not in zone_shift]
    zone_index = {zid: idx for idx, zid in enumerate(zone_order)}
    lines = [
        f"zones_total={len(zone_order)}",
        f"placed={len(zone_shift)}",
        f"missing={len(missing)}",
    ]
    for idx, zid in enumerate(zone_order):
        label = zone_label_map.get(zid, zid)
        if zid in zone_shift:
            dx, dy = zone_shift[zid]
            ang = zone_rot.get(zid, 0.0)
            cx, cy = zone_center.get(zid, (0.0, 0.0))
            lines.append(
                f"zone_id={zid} shuffle_label={label} dx={dx:.2f} dy={dy:.2f} angle={ang:.2f} cx={cx:.2f} cy={cy:.2f}"
            )
        else:
            if idx < len(placements):
                dx, dy, _, _, _ = placements[idx]
            else:
                dx = dy = -1
            info = rot_info[idx] if idx < len(rot_info) else {"angle": 0.0, "cx": 0.0, "cy": 0.0}
            ang = float(info.get("angle", 0.0))
            cx = float(info.get("cx", 0.0))
            cy = float(info.get("cy", 0.0))
            lines.append(
                f"zone_id={zid} shuffle_label={label} missing=1 dx={dx:.2f} dy={dy:.2f} angle={ang:.2f} cx={cx:.2f} cy={cy:.2f}"
            )
    config.OUT_PACK_RASTER_LOG.write_text("\n".join(lines), encoding="utf-8")

    missing = [z for z in zone_order if z not in zone_shift]
    lines = [
        f"zones_total={len(zone_order)}",
        f"placed={len(zone_shift)}",
        f"missing={len(missing)}",
    ]
    for zid in missing:
        idx = zone_index.get(zid, -1)
        if 0 <= idx < len(placements):
            dx, dy, _, _, _ = placements[idx]
        else:
            dx = dy = -1
        info = rot_info[idx] if 0 <= idx < len(rot_info) else {"angle": 0.0, "cx": 0.0, "cy": 0.0}
        ang = float(info.get("angle", 0.0))
        cx = float(info.get("cx", 0.0))
        cy = float(info.get("cy", 0.0))
        lines.append(
            f"zone_id={zid} shuffle_label={zone_label_map.get(zid, zid)} dx={dx:.2f} dy={dy:.2f} angle={ang:.2f} cx={cx:.2f} cy={cy:.2f}"
        )
    config.OUT_PACK_MISSING_LOG.write_text("\n".join(lines), encoding="utf-8")

    if render_packed_png:
        write_pack_png(
            polys,
            zone_id,
            zone_order,
            zone_polys,
            placements,
            canvas,
            colors,
            zone_geoms,
            zone_label_map,
            region_labels,
            rot_info,
        )

    debug["zones_total"] = float(max(zone_id) + 1) if zone_id else 0.0
    debug["packed_placed"] = float(len(zone_shift))
    debug["zones_empty"] = zone_poly_debug.get("empty", [])
    debug["zones_convex_hull"] = zone_poly_debug.get("convex_hull", [])

    return {
        "canvas": {"w": canvas[0], "h": canvas[1]},
        "draw_scale": config.DRAW_SCALE,
        "regions": polys,
        "zone_boundaries": zone_boundaries,
        "zone_id": zone_id,
        "zone_labels": zone_labels,
        "region_labels": region_labels,
        "zone_order": zone_order,
        "zone_rot": zone_rot,
        "zone_center": zone_center,
        "zone_shift": zone_shift,
        "zone_label_map": zone_label_map,
        "region_colors": region_colors,
        "debug": debug,
        "snap": snap,
    }


def main() -> None:
    if not config.SVG_PATH.exists():
        raise SystemExit(f"Missing {config.SVG_PATH}")
    config._apply_pack_env()

    svg_mtime = os.path.getmtime(config.SVG_PATH)
    cache_ok = False
    if config.USE_ZONE_CACHE and config.OUT_ZONES_JSON.exists():
        try:
            data = json.loads(config.OUT_ZONES_JSON.read_text(encoding="utf-8"))
            cache_ok = float(data.get("svg_mtime", -1)) >= svg_mtime
        except Exception:
            cache_ok = False

    if cache_ok:
        polys, zone_id = zones.load_zones_cache(config.OUT_ZONES_JSON)
        base_canvas = svg_utils._get_canvas_size(ET.parse(config.SVG_PATH).getroot(), 1.0)
        canvas = base_canvas
        regions = [geometry.RegionInfo(i, 0.0, (0, 0, 0, 0), (0.0, 0.0)) for i in range(len(polys))]
    else:
        regions, polys, canvas, _ = geometry.build_regions_from_svg(config.SVG_PATH)
        geometry.write_log(regions, config.OUT_LOG)
        geometry.write_png(polys, regions, canvas)

        zone_id = zones.build_zones(polys, config.TARGET_ZONES)
        zone_id, _ = zones._remap_zones_by_area(polys, zone_id)
        zones.write_zones_log(zone_id, config.OUT_ZONES_LOG)
        zones.save_zones_cache(zone_id, polys, config.OUT_ZONES_JSON)

    colors, _ = geometry.render_color_regions(polys, svg_utils._get_canvas_size(ET.parse(config.SVG_PATH).getroot(), 1.0))
    geometry.write_zones_png(polys, zone_id, canvas, colors)

    zone_polys, zone_order, _ = zones.build_zone_polys(polys, zone_id)
    zone_geoms = zones.build_zone_geoms(polys, zone_id)
    zone_ids = sorted(zone_geoms.keys())
    rng = np.random.default_rng(42)
    shuffled = zone_ids.copy()
    rng.shuffle(shuffled)
    zone_labels = {z: idx + 1 for idx, z in enumerate(shuffled)}
    zone_boundaries = zones.build_zone_boundaries(polys, zone_id)
    zones.write_zone_outline_png(zone_geoms, zone_labels, canvas, zone_boundaries)
    svg_utils.write_zone_svg(polys, zone_boundaries, canvas, colors)
    svg_utils.write_zone_outline_svg(zone_boundaries, canvas)
    svg_utils.write_region_svg(polys, canvas)
    zones.write_zones_log(zone_id, config.OUT_ZONES_LOG, zone_labels)

    region_ids = list(range(len(polys)))
    rng = np.random.default_rng(43)
    rng.shuffle(region_ids)
    region_labels = {rid: idx + 1 for idx, rid in enumerate(region_ids)}
    base_canvas = svg_utils._get_canvas_size(ET.parse(config.SVG_PATH).getroot(), 1.0)
    placements, _, rot_info = pack_regions(zone_polys, base_canvas, allow_rotate=True, angle_step=5.0)
    write_pack_log(zone_polys, placements, config.OUT_PACK_LOG, base_canvas)
    write_pack_png(polys, zone_id, zone_order, zone_polys, placements, base_canvas, colors, zone_geoms, zone_labels, region_labels, rot_info)
    write_pack_svg(polys, zone_id, zone_order, zone_polys, placements, base_canvas, colors, rot_info)
    write_pack_outline_png(polys, zone_id, zone_order, zone_polys, placements, base_canvas, zone_geoms, rot_info)

    total_zones = max(zone_id) + 1 if zone_id else 0
    print(
        f"Wrote {config.OUT_LOG}, {config.OUT_PNG}, {config.OUT_ZONES_LOG}, {config.OUT_ZONES_PNG}, {config.OUT_PACK_LOG}, {config.OUT_PACK_PNG} "
        f"with {len(regions)} regions and {total_zones} zones"
    )


if __name__ == "__main__":
    main()
