import { useEffect } from "react";
import { parsePackedSvg } from "../utils/svg";

export default function usePackedSvg(packedImageSrc, setPackedFillPaths, setPackedBleedPaths, setPackedBleedError) {
  useEffect(() => {
    if (!packedImageSrc) return;
    fetch(packedImageSrc)
      .then((res) => res.text())
      .then((text) => {
        const parsed = parsePackedSvg(text);
        setPackedFillPaths(parsed.fillPaths);
        setPackedBleedPaths(parsed.bleedPaths);
        if (!parsed.hasBleed) {
          setPackedBleedError("packed.svg missing bleed layer");
        } else {
          setPackedBleedError("");
        }
      })
      .catch(() => {
        setPackedFillPaths([]);
        setPackedBleedPaths([]);
        setPackedBleedError("packed.svg failed to load");
      });
  }, [packedImageSrc, setPackedFillPaths, setPackedBleedPaths, setPackedBleedError]);
}
