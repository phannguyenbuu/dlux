import { useEffect, useRef, useState } from "react";

export default function useViewState() {
  const stageRef = useRef(null);
  const leftRef = useRef(null);
  const [scale, setScale] = useState(1);
  const [pos, setPos] = useState({ x: 0, y: 0 });
  const [stageSize, setStageSize] = useState({ w: 800, h: 600 });
  const [autoFit, setAutoFit] = useState(true);

  const regionRef = useRef(null);
  const regionWrapRef = useRef(null);
  const [regionScale, setRegionScale] = useState(1);
  const [regionPos, setRegionPos] = useState({ x: 0, y: 0 });
  const [regionStageSize, setRegionStageSize] = useState({ w: 400, h: 400 });

  const region2Ref = useRef(null);
  const region2WrapRef = useRef(null);
  const [region2Scale, setRegion2Scale] = useState(1);
  const [region2Pos, setRegion2Pos] = useState({ x: 0, y: 0 });
  const [region2StageSize, setRegion2StageSize] = useState({ w: 300, h: 200 });

  const zoneRef = useRef(null);
  const zoneWrapRef = useRef(null);
  const [zoneScale, setZoneScale] = useState(1);
  const [zonePos, setZonePos] = useState({ x: 0, y: 0 });
  const [zoneStageSize, setZoneStageSize] = useState({ w: 300, h: 200 });

  useEffect(() => {
    const updateSize = () => {
      if (!leftRef.current) return;
      const rect = leftRef.current.getBoundingClientRect();
      setStageSize({ w: Math.max(300, rect.width), h: Math.max(300, rect.height) });
    };
    updateSize();
    window.addEventListener("resize", updateSize);
    return () => window.removeEventListener("resize", updateSize);
  }, []);

  useEffect(() => {
    const updateRegionSize = () => {
      if (!regionWrapRef.current) return;
      const rect = regionWrapRef.current.getBoundingClientRect();
      setRegionStageSize({ w: Math.max(200, rect.width), h: Math.max(200, rect.height) });
    };
    updateRegionSize();
    window.addEventListener("resize", updateRegionSize);
    return () => window.removeEventListener("resize", updateRegionSize);
  }, []);

  useEffect(() => {
    const updateRegion2Size = () => {
      if (!region2WrapRef.current) return;
      const rect = region2WrapRef.current.getBoundingClientRect();
      setRegion2StageSize({ w: Math.max(200, rect.width), h: Math.max(200, rect.height) });
    };
    updateRegion2Size();
    window.addEventListener("resize", updateRegion2Size);
    return () => window.removeEventListener("resize", updateRegion2Size);
  }, []);

  useEffect(() => {
    const updateZoneSize = () => {
      if (!zoneWrapRef.current) return;
      const rect = zoneWrapRef.current.getBoundingClientRect();
      setZoneStageSize({ w: Math.max(200, rect.width), h: Math.max(200, rect.height) });
    };
    updateZoneSize();
    window.addEventListener("resize", updateZoneSize);
    return () => window.removeEventListener("resize", updateZoneSize);
  }, []);

  const fitToView = (w, h) => {
    const viewW = stageSize.w;
    const viewH = stageSize.h;
    const fitScale = Math.min(viewW / w, viewH / h) * 0.95;
    setScale(fitScale);
    setPos({
      x: (viewW - w * fitScale) / 2,
      y: (viewH - h * fitScale) / 2,
    });
  };

  const fitRegionToView = (bounds) => {
    const rect = regionWrapRef.current?.getBoundingClientRect();
    const viewW = rect?.width || regionStageSize.w;
    const viewH = rect?.height || regionStageSize.h;
    const w = bounds.maxx - bounds.minx;
    const h = bounds.maxy - bounds.miny;
    const fitScale = Math.min(viewW / w, viewH / h) * 0.95;
    setRegionScale(fitScale);
    setRegionPos({
      x: (viewW - w * fitScale) / 2 - bounds.minx * fitScale,
      y: (viewH - h * fitScale) / 2 - bounds.miny * fitScale,
    });
  };

  const fitRegion2ToView = (bounds) => {
    const rect = region2WrapRef.current?.getBoundingClientRect();
    const viewW = rect?.width || region2StageSize.w;
    const viewH = rect?.height || region2StageSize.h;
    const w = bounds.maxx - bounds.minx;
    const h = bounds.maxy - bounds.miny;
    const fitScale = Math.min(viewW / w, viewH / h) * 0.95;
    setRegion2Scale(fitScale);
    setRegion2Pos({
      x: (viewW - w * fitScale) / 2 - bounds.minx * fitScale,
      y: (viewH - h * fitScale) / 2 - bounds.miny * fitScale,
    });
  };

  const fitZoneToView = (bounds) => {
    const rect = zoneWrapRef.current?.getBoundingClientRect();
    const viewW = rect?.width || zoneStageSize.w;
    const viewH = rect?.height || zoneStageSize.h;
    const w = bounds.maxx - bounds.minx;
    const h = bounds.maxy - bounds.miny;
    const fitScale = Math.min(viewW / w, viewH / h) * 0.95;
    setZoneScale(fitScale);
    setZonePos({
      x: (viewW - w * fitScale) / 2 - bounds.minx * fitScale,
      y: (viewH - h * fitScale) / 2 - bounds.miny * fitScale,
    });
  };

  return {
    stageRef,
    leftRef,
    scale,
    setScale,
    pos,
    setPos,
    stageSize,
    setStageSize,
    autoFit,
    setAutoFit,
    regionRef,
    regionWrapRef,
    regionScale,
    setRegionScale,
    regionPos,
    setRegionPos,
    regionStageSize,
    setRegionStageSize,
    region2Ref,
    region2WrapRef,
    region2Scale,
    setRegion2Scale,
    region2Pos,
    setRegion2Pos,
    region2StageSize,
    setRegion2StageSize,
    zoneRef,
    zoneWrapRef,
    zoneScale,
    setZoneScale,
    zonePos,
    setZonePos,
    zoneStageSize,
    setZoneStageSize,
    fitToView,
    fitRegionToView,
    fitRegion2ToView,
    fitZoneToView,
  };
}
