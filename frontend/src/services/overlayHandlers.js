import { applySvgFill, decodeSvgDataUrl, parseSvgSize, svgToDataUrl } from "../utils/svg";
import { loadImageFromSrc } from "../utils/overlay";

export default function createOverlayHandlers({
  overlayInputRef,
  overlayFill,
  setOverlayItems,
  setSelectedOverlayId,
  svgSize,
  overlayItems,
  findZoneAtPoint,
}) {
  const updateOverlayItem = (id, patch) => {
    setOverlayItems((items) =>
      items.map((item) => (item.id === id ? { ...item, ...patch } : item))
    );
  };

  const handleOverlayPick = () => {
    overlayInputRef.current?.click();
  };

  const handleOverlayFileChange = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const raw = await file.text();
    const filled = applySvgFill(raw, overlayFill);
    const src = svgToDataUrl(filled);
    const img = await loadImageFromSrc(src);
    const width = img?.width || parseSvgSize(raw).w || 1;
    const height = img?.height || parseSvgSize(raw).h || 1;
    const id = `overlay-${Date.now()}`;
    const item = {
      id,
      src,
      rawSvg: raw,
      x: svgSize.w * 0.5,
      y: svgSize.h * 0.5,
      width,
      height,
      scaleX: 1,
      scaleY: 1,
      rotation: 0,
      img,
      zid: findZoneAtPoint({ x: svgSize.w * 0.5, y: svgSize.h * 0.5 }),
    };
    const next = [...overlayItems, item];
    setOverlayItems(next);
    setSelectedOverlayId(id);
    e.target.value = "";
  };

  const updateOverlayColor = async (id, color) => {
    const item = overlayItems.find((i) => i.id === id);
    if (!item) return;
    const raw = item.rawSvg || decodeSvgDataUrl(item.src);
    if (!raw) return;
    const filled = applySvgFill(raw, color);
    const src = svgToDataUrl(filled);
    const img = await loadImageFromSrc(src);
    const next = overlayItems.map((i) => (i.id === id ? { ...i, src, img, rawSvg: raw } : i));
    setOverlayItems(next);
  };

  return {
    updateOverlayItem,
    handleOverlayPick,
    handleOverlayFileChange,
    updateOverlayColor,
  };
}
