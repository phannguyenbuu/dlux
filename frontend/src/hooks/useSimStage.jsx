import { useMemo } from "react";
import SimStage from "../panels/SimStage";

export default function useSimStage({
  scene,
  simSize,
  simZoneIds,
  simZoneIndex,
  simLocalFor,
  packedLabels,
  labelFontFamily,
  labelFontSize,
  measureText,
  transformPath,
  toPoints,
  offsetPoints,
  lerp,
}) {
  return useMemo(
    () => (
      <SimStage
        scene={scene}
        simSize={simSize}
        simZoneIds={simZoneIds}
        simZoneIndex={simZoneIndex}
        simLocalFor={simLocalFor}
        packedLabels={packedLabels}
        labelFontFamily={labelFontFamily}
        labelFontSize={labelFontSize}
        measureText={measureText}
        transformPath={transformPath}
        toPoints={toPoints}
        offsetPoints={offsetPoints}
        lerp={lerp}
      />
    ),
    [
      scene,
      simSize,
      simZoneIds,
      simZoneIndex,
      simLocalFor,
      packedLabels,
      labelFontFamily,
      labelFontSize,
      measureText,
      transformPath,
      toPoints,
      offsetPoints,
      lerp,
    ]
  );
}
