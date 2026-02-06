import { useMemo } from "react";
import createSimHandlers from "../services/simHandlers";

export default function useSimHandlers({
  scene,
  packedLabels,
  labelFontFamily,
  labelFontSize,
  simVideoLoading,
  setSimVideoLoading,
  setError,
  simPlaying,
  simProgress,
  setSimProgress,
  setSimPlaying,
}) {
  return useMemo(
    () =>
      createSimHandlers({
        scene,
        packedLabels,
        labelFontFamily,
        labelFontSize,
        simVideoLoading,
        setSimVideoLoading,
        setError,
        simPlaying,
        simProgress,
        setSimProgress,
        setSimPlaying,
      }),
    [
      scene,
      packedLabels,
      labelFontFamily,
      labelFontSize,
      simVideoLoading,
      setSimVideoLoading,
      setError,
      simPlaying,
      simProgress,
      setSimProgress,
      setSimPlaying,
    ]
  );
}
