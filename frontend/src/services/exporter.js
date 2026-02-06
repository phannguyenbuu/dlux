import { captureStageSvg } from "../utils/konva";
import { buildSimulateHtml } from "../sim/simHtml";

const injectSvgLabels = (svgText, labels, fontFamily, fontSize) => {
  try {
    const doc = new DOMParser().parseFromString(svgText, "image/svg+xml");
    const svg = doc.querySelector("svg");
    if (!svg) return svgText;
    Array.from(svg.querySelectorAll("text")).forEach((n) => n.remove());
    Object.values(labels || {}).forEach((lbl) => {
      const x = Number(lbl.x);
      const y = Number(lbl.y);
      if (!Number.isFinite(x) || !Number.isFinite(y)) return;
      const text = doc.createElementNS("http://www.w3.org/2000/svg", "text");
      text.setAttribute("x", String(x));
      text.setAttribute("y", String(y));
      text.setAttribute("fill", "#ffffff");
      text.setAttribute("font-family", fontFamily || "Arial");
      text.setAttribute("font-size", String(fontSize || 12));
      text.setAttribute("text-anchor", "middle");
      text.setAttribute("dominant-baseline", "middle");
      text.textContent = String(lbl.label ?? "");
      svg.appendChild(text);
    });
    return new XMLSerializer().serializeToString(svg);
  } catch {
    return svgText;
  }
};

export default function createExporter(deps) {
  const {
    scene,
    zoneRef,
    regionRef,
    labelFontFamily,
    labelFontSize,
    setExportMsg,
    setExportPdfInfo,
    setExportHtmlInfo,
    setExportPdfLoading,
    setError,
    packedLabels,
    showImages,
    showStroke,
    showLabels,
  } = deps;

  const exportPdf = async () => {
    try {
      setError("");
      setExportMsg("");
      if (!scene?.canvas) {
        throw new Error("canvas missing");
      }
      setExportPdfLoading(true);
      setExportPdfInfo(null);
      const size = { w: scene.canvas.w, h: scene.canvas.h };
      const zoneLabelsSvg = (svgText) =>
        injectSvgLabels(svgText, scene.zone_labels, labelFontFamily, labelFontSize);
      const pages = [
        {
          name: "zone_image",
          svg: zoneLabelsSvg(
            captureStageSvg(zoneRef, size, {
              "zone-image": true,
              "zone-overlay": true,
              "zone-stroke": true,
              "zone-label": true,
              "zone-hit": false,
            })
          ),
        },
        {
          name: "zone_noimage",
          svg: zoneLabelsSvg(
            captureStageSvg(zoneRef, size, {
              "zone-image": false,
              "zone-overlay": true,
              "zone-stroke": true,
              "zone-label": true,
              "zone-hit": false,
            })
          ),
        },
        {
          name: "packed_image_nostroke",
          svg: captureStageSvg(regionRef, size, {
            "packed-image": true,
            "packed-overlay": true,
            "packed-stroke": false,
            "packed-label": true,
            "packed-hit": false,
          }),
        },
        {
          name: "packed_noimage_stroke_nolabel",
          svg: captureStageSvg(regionRef, size, {
            "packed-image": false,
            "packed-overlay": true,
            "packed-stroke": true,
            "packed-label": false,
            "packed-hit": false,
          }),
        },
      ];
      const res = await fetch("/api/export_pdf", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          pages,
          fontName: labelFontFamily,
          fontSize: labelFontSize,
        }),
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.error || `export failed: ${res.status}`);
      }
      const data = await res.json().catch(() => ({}));
      if (data?.name) {
        setExportPdfInfo({ name: data.name });
      }
      const htmlNames = [];
      try {
        const baseName = data?.name ? data.name.replace(/\.pdf$/i, "") : "convoi";
        const html0 = buildSimulateHtml(scene, packedLabels, labelFontFamily, labelFontSize);
        if (html0) {
          const name0 = `${baseName}_simulate.html`;
          await fetch("/api/save_html", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              name: name0,
              html: html0,
            }),
          });
          htmlNames.push(name0);
        }
      } catch {
        // ignore html export errors
      }
      setExportHtmlInfo(htmlNames);
      setExportMsg("Export PDF Done");
      setTimeout(() => setExportMsg(""), 3000);
    } catch (err) {
      setError(err.message || String(err));
    } finally {
      setExportPdfLoading(false);
    }
  };

  const downloadStage = (ref, filename, exportSize = null) => {
    try {
      const svg = captureStageSvg(ref, exportSize);
      try {
        let suffix = "";
        if (showImages) suffix += "_image";
        if (showStroke) suffix += "_stroke";
        suffix += showLabels ? "_label" : "_nolabel";
        const nameWithSuffix = filename.replace(/\.svg$/i, `${suffix}.svg`);
        fetch("/api/save_konva_svg", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ name: nameWithSuffix, svg }),
        });
      } catch {
        // ignore save errors
      }
    } catch {
      // ignore download errors
    }
  };

  return { exportPdf, downloadStage };
}
