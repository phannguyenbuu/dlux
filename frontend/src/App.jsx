import React, { useEffect, useCallback } from "react";
import {
  toPoints,
  transformPath,
  pointInPoly,
  pointSegDist,
  lerp,
  offsetPoints,
} from "./utils/geometry";
import { measureText } from "./utils/konva";
import { mergeNodesIfClose, calcBounds, calcBoundsFromLines } from "./utils/sourceGraph";
import { findZoneAtPoint as findZoneAtPointCore, transformOverlayToPacked } from "./utils/zone";
import useSceneState from "./hooks/useSceneState";
import usePackState from "./hooks/usePackState";
import useOverlayState from "./hooks/useOverlayState";
import useEditState from "./hooks/useEditState";
import useViewState from "./hooks/useViewState";
import useSimState from "./hooks/useSimState";
import useSceneEffects from "./hooks/useSceneEffects";
import useSimTicker from "./hooks/useSimTicker";
import useAutoPack from "./hooks/useAutoPack";
import usePackedSvg from "./hooks/usePackedSvg";
import useWheelHandlers from "./hooks/useWheelHandlers";
import useSceneLoader from "./hooks/useSceneLoader";
import useOverlayHandlers from "./hooks/useOverlayHandlers";
import useSourceHandlers from "./hooks/useSourceHandlers";
import useExporter from "./hooks/useExporter";
import useSimHandlers from "./hooks/useSimHandlers";
import useSourceLayers from "./hooks/useSourceLayers.jsx";
import useSimStage from "./hooks/useSimStage.jsx";
import useSaveState from "./hooks/useSaveState";
import AppLayout from "./panels/AppLayout";

export default function App() {
  const sceneState = useSceneState();
  const packState = usePackState();
  const overlayState = useOverlayState();
  const editState = useEditState();
  const viewState = useViewState();
  const simState = useSimState(sceneState.scene);

  const findZoneAtPoint = useCallback(
    (pt) => findZoneAtPointCore(pt, sceneState.zoneScene, sceneState.scene, pointInPoly),
    [sceneState.zoneScene, sceneState.scene]
  );

  const sceneLoader = useSceneLoader({
    snap: sceneState.snap,
    packPadding: packState.packPadding,
    packMarginX: packState.packMarginX,
    packMarginY: packState.packMarginY,
    drawScale: packState.drawScale,
    packGrid: packState.packGrid,
    packAngle: packState.packAngle,
    packMode: packState.packMode,
    setError: sceneState.setError,
    setAutoFit: viewState.setAutoFit,
    setSceneLoading: sceneState.setSceneLoading,
    setSvgSize: sceneState.setSvgSize,
    setOverlayItems: overlayState.setOverlayItems,
    setSelectedOverlayId: overlayState.setSelectedOverlayId,
    setSvgFallback: sceneState.setSvgFallback,
    setBorderSegments: sceneState.setBorderSegments,
    setRawSegments: sceneState.setRawSegments,
    setNodes: sceneState.setNodes,
    setSegs: sceneState.setSegs,
    setScene: sceneState.setScene,
    setZoneScene: sceneState.setZoneScene,
    setDrawScale: packState.setDrawScale,
    setLabels: sceneState.setLabels,
    setPackedImageSrc: packState.setPackedImageSrc,
    setPackedImageSrc2: packState.setPackedImageSrc2,
    setPackedLabels: sceneState.setPackedLabels,
    setScale: viewState.setScale,
    setPos: viewState.setPos,
    setRegionScale: viewState.setRegionScale,
    setRegionPos: viewState.setRegionPos,
    setRegion2Scale: viewState.setRegion2Scale,
    setRegion2Pos: viewState.setRegion2Pos,
    setZoneScale: viewState.setZoneScale,
    setZonePos: viewState.setZonePos,
    stageSize: viewState.stageSize,
    fitRegionToView: viewState.fitRegionToView,
    fitRegion2ToView: viewState.fitRegion2ToView,
    fitZoneToView: viewState.fitZoneToView,
  });

  const overlayHandlers = useOverlayHandlers({
    overlayInputRef: overlayState.overlayInputRef,
    overlayFill: overlayState.overlayFill,
    setOverlayItems: overlayState.setOverlayItems,
    setSelectedOverlayId: overlayState.setSelectedOverlayId,
    svgSize: sceneState.svgSize,
    overlayItems: overlayState.overlayItems,
    findZoneAtPoint,
  });

  const sourceHandlers = useSourceHandlers({
    stageRef: viewState.stageRef,
    pos: viewState.pos,
    scale: viewState.scale,
    edgeMode: editState.edgeMode,
    deleteEdgeMode: editState.deleteEdgeMode,
    addNodeMode: editState.addNodeMode,
    edgeCandidate: editState.edgeCandidate,
    deleteEdgeCandidate: editState.deleteEdgeCandidate,
    nodes: sceneState.nodes,
    segs: sceneState.segs,
    setEdgeCandidate: editState.setEdgeCandidate,
    setDeleteEdgeCandidate: editState.setDeleteEdgeCandidate,
    setNodes: sceneState.setNodes,
    setSegs: sceneState.setSegs,
    pointSegDist,
  });

  const exporter = useExporter({
    scene: sceneState.scene,
    zoneRef: viewState.zoneRef,
    regionRef: viewState.regionRef,
    labelFontFamily: sceneState.labelFontFamily,
    labelFontSize: sceneState.labelFontSize,
    setExportMsg: sceneState.setExportMsg,
    setExportPdfInfo: sceneState.setExportPdfInfo,
    setExportHtmlInfo: sceneState.setExportHtmlInfo,
    setExportPdfLoading: sceneState.setExportPdfLoading,
    setError: sceneState.setError,
    packedLabels: sceneState.packedLabels,
    showImages: sceneState.showImages,
    showStroke: sceneState.showStroke,
    showLabels: sceneState.showLabels,
  });

  const simHandlers = useSimHandlers({
    scene: sceneState.scene,
    packedLabels: sceneState.packedLabels,
    labelFontFamily: sceneState.labelFontFamily,
    labelFontSize: sceneState.labelFontSize,
    simVideoLoading: simState.simVideoLoading,
    setSimVideoLoading: simState.setSimVideoLoading,
    setError: sceneState.setError,
    simPlaying: simState.simPlaying,
    simProgress: simState.simProgress,
    setSimProgress: simState.setSimProgress,
    setSimPlaying: simState.setSimPlaying,
  });

  const wheelHandlers = useWheelHandlers({
    stageRef: viewState.stageRef,
    setScale: viewState.setScale,
    setPos: viewState.setPos,
    regionRef: viewState.regionRef,
    setRegionScale: viewState.setRegionScale,
    setRegionPos: viewState.setRegionPos,
    region2Ref: viewState.region2Ref,
    setRegion2Scale: viewState.setRegion2Scale,
    setRegion2Pos: viewState.setRegion2Pos,
    zoneRef: viewState.zoneRef,
    setZoneScale: viewState.setZoneScale,
    setZonePos: viewState.setZonePos,
  });

  useEffect(() => {
    sceneLoader.loadScene();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useSceneEffects({
    autoFit: viewState.autoFit,
    svgSize: sceneState.svgSize,
    stageSize: viewState.stageSize,
    fitToView: viewState.fitToView,
    scene: sceneState.scene,
    overlayItems: overlayState.overlayItems,
    setOverlayItems: overlayState.setOverlayItems,
    overlayTransformerRef: overlayState.overlayTransformerRef,
    overlayNodeRefs: overlayState.overlayNodeRefs,
    selectedOverlayId: overlayState.selectedOverlayId,
    setSelectedOverlayId: overlayState.setSelectedOverlayId,
    findZoneAtPoint,
    regionStageSize: viewState.regionStageSize,
    region2StageSize: viewState.region2StageSize,
    zoneStageSize: viewState.zoneStageSize,
    fitRegionToView: viewState.fitRegionToView,
    fitRegion2ToView: viewState.fitRegion2ToView,
    fitZoneToView: viewState.fitZoneToView,
    calcBounds,
    calcBoundsFromLines,
  });

  useSimTicker(
    simState.simPlaying,
    simState.setSimPlaying,
    simState.simTotalSeconds,
    simState.setSimProgress
  );

  useAutoPack(
    packState.autoPack,
    packState.packPadding,
    packState.packMarginX,
    packState.packMarginY,
    packState.packBleed,
    packState.packGrid,
    packState.packAngle,
    packState.packMode,
    sceneLoader.loadScene
  );

  usePackedSvg(
    packState.packedImageSrc,
    packState.setPackedFillPaths,
    packState.setPackedBleedPaths,
    packState.setPackedBleedError
  );
  usePackedSvg(
    packState.packedImageSrc2,
    packState.setPackedFillPaths2,
    packState.setPackedBleedPaths2,
    packState.setPackedBleedError2
  );

  const { nodeLayer, borderLayer } = useSourceLayers(
    sceneState.segs,
    sceneState.nodes,
    sceneState.borderSegments,
    viewState.scale
  );

  const simStage = useSimStage({
    scene: sceneState.scene,
    simSize: simState.simSize,
    simZoneIds: simState.simZoneIds,
    simZoneIndex: simState.simZoneIndex,
    simLocalFor: simState.simLocalFor,
    packedLabels: sceneState.packedLabels,
    labelFontFamily: sceneState.labelFontFamily,
    labelFontSize: sceneState.labelFontSize,
    measureText,
    transformPath,
    toPoints,
    offsetPoints,
    lerp,
  });

  const saveState = useSaveState({
    scene: sceneState.scene,
    nodes: sceneState.nodes,
    segs: sceneState.segs,
    labels: sceneState.labels,
    snap: sceneState.snap,
    view: {
      source: { scale: viewState.scale, pos: viewState.pos },
      region: { scale: viewState.regionScale, pos: viewState.regionPos },
      region2: { scale: viewState.region2Scale, pos: viewState.region2Pos },
      zone: { scale: viewState.zoneScale, pos: viewState.zonePos },
    },
    saveState: sceneLoader.saveState,
  });

  const saveSvg = (
    nextNodes = sceneState.nodes,
    nextSegs = sceneState.segs,
    nextOverlays = overlayState.overlayItems
  ) => sceneLoader.saveSvg(nextNodes, nextSegs, nextOverlays);

  const transformPacked = useCallback(
    (item) => transformOverlayToPacked(item, sceneState.scene, transformPath),
    [sceneState.scene]
  );

  return (
    <AppLayout
      sceneState={sceneState}
      packState={packState}
      overlayState={overlayState}
      editState={editState}
      viewState={viewState}
      simState={simState}
      sceneLoader={sceneLoader}
      overlayHandlers={overlayHandlers}
      sourceHandlers={sourceHandlers}
      exporter={exporter}
      simHandlers={simHandlers}
      wheelHandlers={wheelHandlers}
      nodeLayer={nodeLayer}
      borderLayer={borderLayer}
      simStage={simStage}
      saveSvg={saveSvg}
      saveState={saveState}
      findZoneAtPoint={findZoneAtPoint}
      mergeNodesIfClose={mergeNodesIfClose}
      measureText={measureText}
      toPoints={toPoints}
      transformPath={transformPath}
      offsetPoints={offsetPoints}
      transformOverlayToPacked={transformPacked}
    />
  );
}
