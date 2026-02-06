import { useRef, useState } from "react";

export default function useOverlayState() {
  const overlayInputRef = useRef(null);
  const overlayTransformerRef = useRef(null);
  const overlayNodeRefs = useRef({});
  const [overlayItems, setOverlayItems] = useState([]);
  const [selectedOverlayId, setSelectedOverlayId] = useState(null);
  const [overlayFill, setOverlayFill] = useState("#000000");

  return {
    overlayInputRef,
    overlayTransformerRef,
    overlayNodeRefs,
    overlayItems,
    setOverlayItems,
    selectedOverlayId,
    setSelectedOverlayId,
    overlayFill,
    setOverlayFill,
  };
}
