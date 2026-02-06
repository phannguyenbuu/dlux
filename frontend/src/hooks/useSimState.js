import { useEffect, useMemo, useRef, useState } from "react";

export default function useSimState(scene) {
  const [showSim, setShowSim] = useState(false);
  const [simPlaying, setSimPlaying] = useState(false);
  const [simProgress, setSimProgress] = useState(0);
  const [simSize, setSimSize] = useState({ w: 800, h: 500 });
  const [simVideoLoading, setSimVideoLoading] = useState(false);
  const simWrapRef = useRef(null);

  const simZoneIds = useMemo(() => {
    const ids = Object.keys(scene?.zone_boundaries || {});
    const getLabel = (zid) => {
      const lbl =
        scene?.zone_label_map?.[zid] ??
        scene?.zone_label_map?.[parseInt(zid, 10)] ??
        zid;
      const num = Number(lbl);
      return Number.isFinite(num) ? num : Number(zid) || 0;
    };
    return ids.sort((a, b) => getLabel(a) - getLabel(b));
  }, [scene]);

  const simZoneIndex = useMemo(() => {
    const map = {};
    simZoneIds.forEach((zid, idx) => {
      map[String(zid)] = idx;
    });
    return map;
  }, [simZoneIds]);

  const simTiming = useMemo(() => {
    const move = 1;
    const hold = 0.2;
    const per = move + hold;
    const total = simZoneIds.length ? simZoneIds.length * per : 1;
    return { move, hold, per, total };
  }, [simZoneIds]);

  const simMoveSeconds = simTiming.move;
  const simHoldSeconds = simTiming.hold;
  const simPerZone = simTiming.per;
  const simTotalSeconds = simTiming.total;

  const simActiveIdx = simZoneIds.length
    ? Math.min(
        simZoneIds.length - 1,
        Math.max(0, Math.floor((simProgress * simTotalSeconds) / simPerZone))
      )
    : -1;
  const simActiveZid = simActiveIdx >= 0 ? simZoneIds[simActiveIdx] : null;
  const simActiveLabel =
    simActiveZid != null
      ? scene?.zone_label_map?.[simActiveZid] ??
        scene?.zone_label_map?.[parseInt(simActiveZid, 10)] ??
        simActiveZid
      : "";

  const simLocalFor = (idx) => {
    if (idx == null || idx < 0) return 0;
    const t = simProgress * simTotalSeconds - idx * simPerZone;
    if (t <= 0) return 0;
    if (t >= simPerZone) return 1;
    if (t >= simMoveSeconds) return 1;
    const x = t / simMoveSeconds;
    return 1 - Math.pow(1 - x, 3);
  };

  useEffect(() => {
    const updateSimSize = () => {
      if (!simWrapRef.current) return;
      const rect = simWrapRef.current.getBoundingClientRect();
      setSimSize({ w: Math.max(300, rect.width), h: Math.max(200, rect.height) });
    };
    updateSimSize();
    window.addEventListener("resize", updateSimSize);
    return () => window.removeEventListener("resize", updateSimSize);
  }, []);

  return {
    showSim,
    setShowSim,
    simPlaying,
    setSimPlaying,
    simProgress,
    setSimProgress,
    simSize,
    simVideoLoading,
    setSimVideoLoading,
    simWrapRef,
    simZoneIds,
    simZoneIndex,
    simTiming,
    simMoveSeconds,
    simHoldSeconds,
    simPerZone,
    simTotalSeconds,
    simActiveLabel,
    simLocalFor,
  };
}
