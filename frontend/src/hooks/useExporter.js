import { useMemo } from "react";
import createExporter from "../services/exporter";

export default function useExporter({
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
}) {
  return useMemo(
    () =>
      createExporter({
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
      }),
    [
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
    ]
  );
}
