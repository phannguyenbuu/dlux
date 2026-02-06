import { useState } from "react";

export default function usePackState() {
  const [packPadding, setPackPadding] = useState(4);
  const [packMarginX, setPackMarginX] = useState(30);
  const [packMarginY, setPackMarginY] = useState(30);
  const [packBleed, setPackBleed] = useState(10);
  const [drawScale, setDrawScale] = useState(0.5);
  const [packGrid, setPackGrid] = useState(5);
  const [packAngle, setPackAngle] = useState(5);
  const [packMode, setPackMode] = useState("fast");
  const [autoPack, setAutoPack] = useState(false);
  const [packedImageSrc, setPackedImageSrc] = useState("/out/packed.svg");
  const [packedImageSrc2, setPackedImageSrc2] = useState("/out/packed_page2.svg");
  const [packedFillPaths, setPackedFillPaths] = useState([]);
  const [packedBleedPaths, setPackedBleedPaths] = useState([]);
  const [packedBleedError, setPackedBleedError] = useState("");
  const [packedFillPaths2, setPackedFillPaths2] = useState([]);
  const [packedBleedPaths2, setPackedBleedPaths2] = useState([]);
  const [packedBleedError2, setPackedBleedError2] = useState("");

  return {
    packPadding,
    setPackPadding,
    packMarginX,
    setPackMarginX,
    packMarginY,
    setPackMarginY,
    packBleed,
    setPackBleed,
    drawScale,
    setDrawScale,
    packGrid,
    setPackGrid,
    packAngle,
    setPackAngle,
    packMode,
    setPackMode,
    autoPack,
    setAutoPack,
    packedImageSrc,
    setPackedImageSrc,
    packedImageSrc2,
    setPackedImageSrc2,
    packedFillPaths,
    setPackedFillPaths,
    packedBleedPaths,
    setPackedBleedPaths,
    packedBleedError,
    setPackedBleedError,
    packedFillPaths2,
    setPackedFillPaths2,
    packedBleedPaths2,
    setPackedBleedPaths2,
    packedBleedError2,
    setPackedBleedError2,
  };
}
