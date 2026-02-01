from __future__ import annotations

import json
import os
import subprocess
import math
from pathlib import Path
from typing import Any, Dict

from flask import Flask, jsonify, request, send_from_directory
import cv2
import xml.etree.ElementTree as ET

import new_toy

ROOT = Path(__file__).resolve().parent
WEB_DIR = ROOT / "frontend"
DIST_DIR = WEB_DIR / "dist"
STATE_JSON = ROOT / "ui_state.json"
STATE_SVG = ROOT / "ui_state.svg"
PACKED_LABELS_JSON = ROOT / "packed_labels.json"
ZONE_LABELS_JSON = ROOT / "zone_labels.json"
SCENE_JSON = ROOT / "scene_cache.json"
SVG_PATH = ROOT / "convoi.svg"
SVG_BACKUP = ROOT / "convoi_backup.svg"
EXPORT_DIR = ROOT / "export"

app = Flask(__name__, static_folder=None)


def ensure_outputs() -> None:
    env = os.environ.copy()
    env["INTERSECT_SNAP"] = str(new_toy.INTERSECT_SNAP)
    env["LINE_EXTEND"] = str(new_toy.LINE_EXTEND)
    cmd = [os.fspath(Path(os.environ.get("PYTHON", "python"))), os.fspath(ROOT / "new_toy.py")]
    subprocess.run(cmd, cwd=ROOT, env=env, check=False)


@app.get("/api/scene")
def api_scene():
    snap = request.args.get("snap", type=float) or new_toy.INTERSECT_SNAP
    for key, env_key in (
        ("pack_padding", "PACK_PADDING"),
        ("pack_margin_x", "PACK_MARGIN_X"),
        ("pack_margin_y", "PACK_MARGIN_Y"),
        ("pack_bleed", "PACK_BLEED"),
        ("draw_scale", "DRAW_SCALE"),
        ("pack_grid", "PACK_GRID_STEP"),
        ("pack_angle", "PACK_ANGLE_STEP"),
        ("pack_mode", "PACK_MODE"),
    ):
        val = request.args.get(key)
        if val is not None:
            os.environ[env_key] = str(val)
    data = new_toy.compute_scene(new_toy.SVG_PATH, snap, render_packed_png=True)
    SCENE_JSON.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    zone_labels = data.get("zone_labels")
    if isinstance(zone_labels, dict):
        ZONE_LABELS_JSON.write_text(
            json.dumps(zone_labels, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return jsonify(data)


@app.post("/api/render")
def api_render():
    payload: Dict[str, Any] = request.get_json(silent=True) or {}
    snap = float(payload.get("snap", new_toy.INTERSECT_SNAP))
    env = os.environ.copy()
    env["INTERSECT_SNAP"] = str(snap)
    env["LINE_EXTEND"] = str(new_toy.LINE_EXTEND)
    for key, env_key in (
        ("pack_padding", "PACK_PADDING"),
        ("pack_margin_x", "PACK_MARGIN_X"),
        ("pack_margin_y", "PACK_MARGIN_Y"),
        ("pack_bleed", "PACK_BLEED"),
        ("draw_scale", "DRAW_SCALE"),
        ("pack_grid", "PACK_GRID_STEP"),
        ("pack_angle", "PACK_ANGLE_STEP"),
        ("pack_mode", "PACK_MODE"),
    ):
        if key in payload:
            env[env_key] = str(payload[key])
    cmd = [os.fspath(Path(os.environ.get("PYTHON", "python"))), os.fspath(ROOT / "new_toy.py")]
    proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        return jsonify({"ok": False, "error": proc.stderr.strip()}), 500
    return jsonify({"ok": True})


@app.post("/api/state")
def api_state():
    payload: Dict[str, Any] = request.get_json(silent=True) or {}
    STATE_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # build a lightweight svg for fast restore
    canvas = payload.get("canvas", {})
    w = canvas.get("w", 1000)
    h = canvas.get("h", 1000)
    paths = []
    for region in payload.get("regions", []):
        if not region:
            continue
        d = "M " + " L ".join(f"{p[0]} {p[1]}" for p in region) + " Z"
        paths.append(f'<path d="{d}" fill="none" stroke="#999" stroke-width="0.5"/>')
    labels = []
    for lbl in payload.get("labels", []):
        labels.append(
            f'<text x="{lbl["x"]}" y="{lbl["y"]}" font-size="6" fill="#000">{lbl["label"]}</text>'
        )

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}">'
        + "".join(paths)
        + "".join(labels)
        + "</svg>"
    )
    STATE_SVG.write_text(svg, encoding="utf-8")
    return jsonify({"ok": True})


@app.get("/api/packed_labels")
def api_packed_labels():
    if PACKED_LABELS_JSON.exists():
        try:
            data = json.loads(PACKED_LABELS_JSON.read_text(encoding="utf-8"))
            return jsonify(data)
        except Exception:
            return jsonify({})
    return jsonify({})


@app.post("/api/packed_labels")
def api_save_packed_labels():
    payload: Dict[str, Any] = request.get_json(silent=True) or {}
    data: Dict[str, Any] = {}
    if PACKED_LABELS_JSON.exists():
        try:
            data = json.loads(PACKED_LABELS_JSON.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    for key, val in payload.items():
        data[str(key)] = val
    PACKED_LABELS_JSON.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return jsonify({"ok": True})


@app.post("/api/export")
def api_export():
    try:
        print("[export] 0% start")
        EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        print("[export] 10% use existing outputs")

        target_w_mm = 260.0
        target_h_mm = 190.0
        packed_labels: Dict[str, Any] = {}
        if PACKED_LABELS_JSON.exists():
            try:
                packed_labels = json.loads(PACKED_LABELS_JSON.read_text(encoding="utf-8"))
            except Exception:
                packed_labels = {}

        canvas_w = None
        canvas_h = None
        cached_scene: Dict[str, Any] = {}
        if SCENE_JSON.exists():
            try:
                cached_scene = json.loads(SCENE_JSON.read_text(encoding="utf-8"))
            except Exception:
                cached_scene = {}
        if cached_scene.get("canvas"):
            try:
                canvas_w = float(cached_scene["canvas"].get("w", 0))
                canvas_h = float(cached_scene["canvas"].get("h", 0))
            except Exception:
                canvas_w = None
                canvas_h = None

        export_draw_scale = 10.0
        prefix = new_toy.config.SVG_PATH.stem
        scale_up = 1
        packed_png = ROOT / "packed.png"
        packed_export_png = EXPORT_DIR / f"{prefix}_packed_draw{int(export_draw_scale)}.png"
        img = None
        if (
            cached_scene.get("regions")
            and cached_scene.get("zone_id")
            and cached_scene.get("zone_order")
            and cached_scene.get("placements")
            and cached_scene.get("colors_bgr")
            and cached_scene.get("rot_info")
            and cached_scene.get("canvas")
        ):
            try:
                new_toy.write_pack_png(
                    cached_scene["regions"],
                    cached_scene["zone_id"],
                    cached_scene["zone_order"],
                    [],
                    cached_scene["placements"],
                    (int(canvas_w), int(canvas_h)),
                    cached_scene["colors_bgr"],
                    {},
                    cached_scene.get("zone_label_map", {}),
                    cached_scene.get("region_labels", {}),
                    cached_scene["rot_info"],
                    draw_scale=export_draw_scale,
                    out_path=packed_export_png,
                )
                img = cv2.imread(str(packed_export_png), cv2.IMREAD_UNCHANGED)
            except Exception:
                img = None
        if img is None:
            scale_up = 10
            if not packed_png.exists():
                return jsonify({"ok": False, "error": "packed.png not found"}), 500
            img = cv2.imread(str(packed_png), cv2.IMREAD_UNCHANGED)
            if img is None:
                return jsonify({"ok": False, "error": "failed to read packed.png"}), 500

        if packed_labels and canvas_w and canvas_h:
            scale_x = img.shape[1] / float(canvas_w)
            scale_y = img.shape[0] / float(canvas_h)
            font_px = float(new_toy.config.PACK_LABEL_SCALE) * 20.0 * 0.25
            font_scale = max(0.4, (font_px * min(scale_x, scale_y)) / 30.0)
            thickness = max(1, int(round(font_scale * 2)))
            for _, lbl in packed_labels.items():
                try:
                    x = float(lbl.get("x", 0))
                    y = float(lbl.get("y", 0))
                    text = str(lbl.get("label", ""))
                except Exception:
                    continue
                px = int(round(x * scale_x))
                py = int(round(y * scale_y))
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                cv2.putText(
                    img,
                    text,
                    (px - tw // 2, py + th // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255, 255),
                    thickness,
                    cv2.LINE_AA,
                )

        if scale_up != 1:
            resized = cv2.resize(
                img, (img.shape[1] * scale_up, img.shape[0] * scale_up), interpolation=cv2.INTER_AREA
            )
        else:
            resized = img
        print("[export] 80% write raster")
        out_scale_label = int(export_draw_scale) if scale_up == 1 else scale_up
        out_png = EXPORT_DIR / f"{prefix}_packed_x{out_scale_label}.png"
        cv2.imwrite(str(out_png), resized)

        print("[export] 90% write svgs")
        prefix = new_toy.config.SVG_PATH.stem
        zone_outline_svg = ROOT / "zone_outline.svg"

        if canvas_w and canvas_h and cached_scene.get("zone_boundaries"):
            def rotate_pt(pt, angle_deg, cx, cy):
                if not angle_deg:
                    return pt
                ang = (angle_deg * math.pi) / 180.0
                c = math.cos(ang)
                s = math.sin(ang)
                x = pt[0] - cx
                y = pt[1] - cy
                return [cx + x * c - y * s, cy + x * s + y * c]

            def transform_path(pts, shift, rot, center):
                dx = shift[0] if shift else 0
                dy = shift[1] if shift else 0
                ang = rot if rot else 0
                cx = center[0] if center else 0
                cy = center[1] if center else 0
                out = []
                for p in pts:
                    rp = rotate_pt(p, ang, cx, cy)
                    out.append([rp[0] + dx, rp[1] + dy])
                return out

            parts = [
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{target_w_mm}mm" height="{target_h_mm}mm" viewBox="0 0 {int(canvas_w)} {int(canvas_h)}">'
            ]
            parts.append(
                f'<rect x="0" y="0" width="{int(canvas_w)}" height="{int(canvas_h)}" '
                f'fill="none" stroke="#ffffff" stroke-width="2"/>'
            )
            for zid, paths in cached_scene.get("zone_boundaries", {}).items():
                shift = cached_scene.get("zone_shift", {}).get(str(zid))
                if shift is None:
                    shift = cached_scene.get("zone_shift", {}).get(int(zid)) if str(zid).isdigit() else None
                rot = cached_scene.get("zone_rot", {}).get(str(zid))
                if rot is None:
                    rot = cached_scene.get("zone_rot", {}).get(int(zid)) if str(zid).isdigit() else 0
                center = cached_scene.get("zone_center", {}).get(str(zid))
                if center is None:
                    center = (
                        cached_scene.get("zone_center", {}).get(int(zid))
                        if str(zid).isdigit()
                        else [0, 0]
                    )
                for poly in paths or []:
                    tpts = transform_path(poly, shift, rot or 0, center or [0, 0])
                    if not tpts:
                        continue
                    d = "M " + " L ".join(f"{p[0]} {p[1]}" for p in tpts) + " Z"
                    parts.append(f'<path d="{d}" fill="none" stroke="#ffffff" stroke-width="1"/>')
            packed_label_size = float(new_toy.config.PACK_LABEL_SCALE) * 20.0 * 0.25
            for lbl in packed_labels.values():
                try:
                    x = float(lbl.get("x", 0))
                    y = float(lbl.get("y", 0))
                    text = str(lbl.get("label", ""))
                except Exception:
                    continue
                parts.append(
                    f'<text x="{x}" y="{y}" fill="#ffffff" stroke="rgba(0,0,0,0.5)" '
                    f'stroke-width="1" font-size="{packed_label_size}" text-anchor="middle" '
                    f'dominant-baseline="middle">{text}</text>'
                )
            parts.append("</svg>")
            (EXPORT_DIR / f"{prefix}_packed_260x190.svg").write_text("".join(parts), encoding="utf-8")

        if zone_outline_svg.exists() and canvas_w and canvas_h:
            tree = ET.parse(zone_outline_svg)
            root = tree.getroot()
            parts = [
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{target_w_mm}mm" height="{target_h_mm}mm" viewBox="0 0 {int(canvas_w)} {int(canvas_h)}">'
            ]
            for p in root.findall(".//{http://www.w3.org/2000/svg}path"):
                d = p.attrib.get("d")
                if not d:
                    continue
                parts.append(f'<path d="{d}" fill="none" stroke="#ffffff" stroke-width="1"/>')
            labels = {}
            if ZONE_LABELS_JSON.exists():
                try:
                    labels = json.loads(ZONE_LABELS_JSON.read_text(encoding="utf-8"))
                except Exception:
                    labels = {}
            font_size = str(float(new_toy.config.PACK_LABEL_SCALE) * 20.0 * 0.2)
            for lbl in labels.values():
                try:
                    x = float(lbl.get("x", 0))
                    y = float(lbl.get("y", 0))
                    text = str(lbl.get("label", ""))
                except Exception:
                    continue
                parts.append(
                    f'<text x="{x}" y="{y}" fill="#ffffff" font-size="{font_size}" '
                    f'text-anchor="middle" dominant-baseline="middle">{text}</text>'
                )
            parts.append("</svg>")
            (EXPORT_DIR / f"{prefix}_zone_260x190.svg").write_text("".join(parts), encoding="utf-8")
        print("[export] 100% done")
        return jsonify({"ok": True})
    except Exception as exc:
        print(f"[export] error: {exc}")
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.post("/api/save_svg")
def api_save_svg():
    payload: Dict[str, Any] = request.get_json(silent=True) or {}
    nodes = payload.get("nodes", [])
    segs = payload.get("segs", [])
    if not SVG_PATH.exists():
        return jsonify({"ok": False, "error": "convoi.svg not found"}), 404

    if not SVG_BACKUP.exists():
        SVG_BACKUP.write_bytes(SVG_PATH.read_bytes())

    tree = ET.parse(SVG_PATH)
    root = tree.getroot()

    # remove existing line/polyline/polygon (keep image)
    for parent in list(root.iter()):
        for child in list(parent):
            tag = child.tag.rsplit("}", 1)[-1]
            if tag in {"line", "polyline", "polygon"}:
                parent.remove(child)

    ns = {"svg": "http://www.w3.org/2000/svg"}
    g = ET.Element("g", {"id": "INTERACTIVE"})
    for seg in segs:
        try:
            a = nodes[seg[0]]
            b = nodes[seg[1]]
            line = ET.Element(
                "line",
                {
                    "x1": str(a["x"]),
                    "y1": str(a["y"]),
                    "x2": str(b["x"]),
                    "y2": str(b["y"]),
                    "stroke": "#000",
                    "stroke-width": "1",
                    "fill": "none",
                },
            )
            g.append(line)
        except Exception:
            continue
    root.append(g)

    tree.write(SVG_PATH, encoding="utf-8", xml_declaration=True)
    return jsonify({"ok": True})


@app.get("/")
def index():
    if DIST_DIR.exists():
        return send_from_directory(DIST_DIR, "index.html")
    return send_from_directory(WEB_DIR, "index.html")


@app.get("/<path:path>")
def static_proxy(path: str):
    if DIST_DIR.exists() and (DIST_DIR / path).exists():
        return send_from_directory(DIST_DIR, path)
    return send_from_directory(WEB_DIR, path)


@app.get("/out/<path:path>")
def output_files(path: str):
    return send_from_directory(ROOT, path)


if __name__ == "__main__":
    ensure_outputs()
    app.run(host="127.0.0.1", port=5000, debug=True)
