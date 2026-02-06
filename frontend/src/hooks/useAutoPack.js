import { useEffect } from "react";

export default function useAutoPack(
  autoPack,
  packPadding,
  packMarginX,
  packMarginY,
  packBleed,
  packGrid,
  packAngle,
  packMode,
  loadScene
) {
  useEffect(() => {
    if (!autoPack) return;
    const id = setTimeout(() => {
      loadScene(false);
    }, 500);
    return () => clearTimeout(id);
  }, [packPadding, packMarginX, packMarginY, packBleed, packGrid, packAngle, packMode, autoPack, loadScene]);
}
