import { Image, Transformer } from "react-konva";

export default function SourceOverlayLayer({
  overlayItems,
  edgeMode,
  deleteEdgeMode,
  addNodeMode,
  setSelectedOverlayId,
  findZoneAtPoint,
  overlayNodeRefs,
  overlayTransformerRef,
  setOverlayItems,
}) {
  return (
    <>
      {overlayItems.map((item) =>
        item.img ? (
          <Image
            key={item.id}
            image={item.img}
            x={item.x}
            y={item.y}
            width={item.width}
            height={item.height}
            offsetX={item.width / 2}
            offsetY={item.height / 2}
            scaleX={item.scaleX}
            scaleY={item.scaleY}
            rotation={item.rotation}
            draggable={!edgeMode && !deleteEdgeMode && !addNodeMode}
            onClick={() => setSelectedOverlayId(item.id)}
            onTap={() => setSelectedOverlayId(item.id)}
            onDragEnd={(e) => {
              const nx = e.target.x();
              const ny = e.target.y();
              const zid = findZoneAtPoint({ x: nx, y: ny });
              const next = overlayItems.map((o) =>
                o.id === item.id ? { ...o, x: nx, y: ny, zid } : o
              );
              setOverlayItems(next);
            }}
            onTransformEnd={() => {
              const node = overlayNodeRefs.current[item.id];
              if (!node) return;
              const scaleX = node.scaleX();
              const scaleY = node.scaleY();
              const rotation = node.rotation();
              const nx = node.x();
              const ny = node.y();
              node.scaleX(1);
              node.scaleY(1);
              const next = overlayItems.map((o) =>
                o.id === item.id
                  ? {
                      ...o,
                      x: nx,
                      y: ny,
                      rotation,
                      scaleX: o.scaleX * scaleX,
                      scaleY: o.scaleY * scaleY,
                    }
                  : o
              );
              setOverlayItems(next);
            }}
            ref={(node) => {
              if (node) overlayNodeRefs.current[item.id] = node;
            }}
          />
        ) : null
      )}
      <Transformer
        ref={overlayTransformerRef}
        rotateEnabled
        enabledAnchors={["top-left", "top-right", "bottom-left", "bottom-right"]}
        boundBoxFunc={(oldBox, newBox) => {
          if (newBox.width < 10 || newBox.height < 10) return oldBox;
          return newBox;
        }}
      />
    </>
  );
}
