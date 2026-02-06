import { useState } from "react";

export default function useSceneState() {
  const [snap, setSnap] = useState(1);
  const [scene, setScene] = useState(null);
  const [zoneScene, setZoneScene] = useState(null);
  const [error, setError] = useState("");
  const [labels, setLabels] = useState([]);
  const [packedLabels, setPackedLabels] = useState([]);
  const [exportMsg, setExportMsg] = useState("");
  const [exportPdfInfo, setExportPdfInfo] = useState(null);
  const [exportHtmlInfo, setExportHtmlInfo] = useState([]);
  const [exportPdfLoading, setExportPdfLoading] = useState(false);
  const [selectedZoneId, setSelectedZoneId] = useState(null);
  const [rawSegments, setRawSegments] = useState([]);
  const [borderSegments, setBorderSegments] = useState([]);
  const [nodes, setNodes] = useState([]);
  const [segs, setSegs] = useState([]);
  const [svgImage, setSvgImage] = useState(null);
  const [svgFallback, setSvgFallback] = useState([]);
  const [svgSize, setSvgSize] = useState({ w: 1000, h: 1000 });
  const [sceneLoading, setSceneLoading] = useState(true);
  const [showImages, setShowImages] = useState(false);
  const [showStroke, setShowStroke] = useState(true);
  const [showLabels, setShowLabels] = useState(true);
  const [labelFontFamily, setLabelFontFamily] = useState("Arial");
  const [labelFontSize, setLabelFontSize] = useState(12);

  return {
    snap,
    setSnap,
    scene,
    setScene,
    zoneScene,
    setZoneScene,
    error,
    setError,
    labels,
    setLabels,
    packedLabels,
    setPackedLabels,
    exportMsg,
    setExportMsg,
    exportPdfInfo,
    setExportPdfInfo,
    exportHtmlInfo,
    setExportHtmlInfo,
    exportPdfLoading,
    setExportPdfLoading,
    selectedZoneId,
    setSelectedZoneId,
    rawSegments,
    setRawSegments,
    borderSegments,
    setBorderSegments,
    nodes,
    setNodes,
    segs,
    setSegs,
    svgImage,
    setSvgImage,
    svgFallback,
    setSvgFallback,
    svgSize,
    setSvgSize,
    sceneLoading,
    setSceneLoading,
    showImages,
    setShowImages,
    showStroke,
    setShowStroke,
    showLabels,
    setShowLabels,
    labelFontFamily,
    setLabelFontFamily,
    labelFontSize,
    setLabelFontSize,
  };
}
