const makeWheelHandler = (ref, setScale, setPos) => (e) => {
  e.evt.preventDefault();
  const scaleBy = 1.05;
  const stage = ref.current;
  if (!stage) return;
  const oldScale = stage.scaleX();
  const pointer = stage.getPointerPosition();
  if (!pointer) return;
  const mousePointTo = {
    x: (pointer.x - stage.x()) / oldScale,
    y: (pointer.y - stage.y()) / oldScale,
  };
  const direction = e.evt.deltaY > 0 ? 1 : -1;
  const newScale = direction > 0 ? oldScale / scaleBy : oldScale * scaleBy;
  setScale(newScale);
  setPos({
    x: pointer.x - mousePointTo.x * newScale,
    y: pointer.y - mousePointTo.y * newScale,
  });
};

export default function useWheelHandlers({
  stageRef,
  setScale,
  setPos,
  regionRef,
  setRegionScale,
  setRegionPos,
  region2Ref,
  setRegion2Scale,
  setRegion2Pos,
  zoneRef,
  setZoneScale,
  setZonePos,
}) {
  return {
    handleWheel: makeWheelHandler(stageRef, setScale, setPos),
    handleRegionWheel: makeWheelHandler(regionRef, setRegionScale, setRegionPos),
    handleRegion2Wheel: makeWheelHandler(region2Ref, setRegion2Scale, setRegion2Pos),
    handleZoneWheel: makeWheelHandler(zoneRef, setZoneScale, setZonePos),
  };
}
