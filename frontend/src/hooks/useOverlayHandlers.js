import { useMemo } from "react";
import createOverlayHandlers from "../services/overlayHandlers";

export default function useOverlayHandlers({
  overlayInputRef,
  overlayFill,
  setOverlayItems,
  setSelectedOverlayId,
  svgSize,
  overlayItems,
  findZoneAtPoint,
}) {
  return useMemo(
    () =>
      createOverlayHandlers({
        overlayInputRef,
        overlayFill,
        setOverlayItems,
        setSelectedOverlayId,
        svgSize,
        overlayItems,
        findZoneAtPoint,
      }),
    [
      overlayInputRef,
      overlayFill,
      setOverlayItems,
      setSelectedOverlayId,
      svgSize,
      overlayItems,
      findZoneAtPoint,
    ]
  );
}
