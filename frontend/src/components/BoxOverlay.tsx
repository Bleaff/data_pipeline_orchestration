"use client";

import { useEffect, useRef, useState } from "react";
import type { BoxSpec } from "@/lib/api";
import { boxFromDrag, boxToCssRect, moveBox, resizeBox } from "@/lib/boxGeometry";
import styles from "./BoxOverlay.module.css";

type DragMode = { kind: "move" | "resize"; boxIndex: number } | { kind: "draw" };

interface DragState {
  mode: DragMode;
  startFrac: { x: number; y: number };
}

function fractionFromEvent(container: HTMLElement, e: { clientX: number; clientY: number }) {
  const rect = container.getBoundingClientRect();
  return {
    x: Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width)),
    y: Math.min(1, Math.max(0, (e.clientY - rect.top) / rect.height)),
  };
}

export function BoxOverlay({
  imageUrl,
  boxes,
  selectedIndex,
  onSelect,
  onBoxChange,
  onAddBox,
}: {
  imageUrl: string;
  boxes: BoxSpec[];
  selectedIndex: number | null;
  onSelect: (index: number | null) => void;
  onBoxChange: (index: number, box: BoxSpec) => void;
  onAddBox: (box: BoxSpec) => void;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [drag, setDrag] = useState<DragState | null>(null);
  const [drawPreview, setDrawPreview] = useState<{ x: number; y: number } | null>(null);

  useEffect(() => {
    if (!drag) return;

    function handleMove(e: MouseEvent) {
      const container = containerRef.current;
      if (!container || !drag) return;
      const current = fractionFromEvent(container, e);

      if (drag.mode.kind === "draw") {
        setDrawPreview(current);
        return;
      }

      const box = boxes[drag.mode.boxIndex];
      if (!box) return;
      const dx = current.x - drag.startFrac.x;
      const dy = current.y - drag.startFrac.y;
      const updated =
        drag.mode.kind === "move"
          ? { ...moveBox(box, dx, dy), class_id: box.class_id }
          : { ...resizeBox(box, dx, dy), class_id: box.class_id };
      onBoxChange(drag.mode.boxIndex, updated);
      setDrag({ ...drag, startFrac: current });
    }

    function handleUp(e: MouseEvent) {
      const container = containerRef.current;
      if (container && drag?.mode.kind === "draw") {
        const end = fractionFromEvent(container, e);
        const size = Math.abs(end.x - drag.startFrac.x) + Math.abs(end.y - drag.startFrac.y);
        if (size > 0.01) {
          onAddBox(boxFromDrag(drag.startFrac, end, 0));
        }
      }
      setDrag(null);
      setDrawPreview(null);
    }

    window.addEventListener("mousemove", handleMove);
    window.addEventListener("mouseup", handleUp);
    return () => {
      window.removeEventListener("mousemove", handleMove);
      window.removeEventListener("mouseup", handleUp);
    };
     
  }, [drag, boxes, onBoxChange, onAddBox]);

  function startDraw(e: React.MouseEvent) {
    const container = containerRef.current;
    if (!container) return;
    onSelect(null);
    setDrag({ mode: { kind: "draw" }, startFrac: fractionFromEvent(container, e.nativeEvent) });
  }

  function startMove(index: number, e: React.MouseEvent) {
    e.stopPropagation();
    const container = containerRef.current;
    if (!container) return;
    onSelect(index);
    setDrag({ mode: { kind: "move", boxIndex: index }, startFrac: fractionFromEvent(container, e.nativeEvent) });
  }

  function startResize(index: number, e: React.MouseEvent) {
    e.stopPropagation();
    const container = containerRef.current;
    if (!container) return;
    setDrag({ mode: { kind: "resize", boxIndex: index }, startFrac: fractionFromEvent(container, e.nativeEvent) });
  }

  const previewBox = drag?.mode.kind === "draw" && drawPreview ? boxFromDrag(drag.startFrac, drawPreview, 0) : null;

  return (
    <div ref={containerRef} className={styles.container} onMouseDown={startDraw}>
      {/* eslint-disable-next-line @next/next/no-img-element -- served from the control-plane API, not next/image-optimizable */}
      <img src={imageUrl} alt="Frame under review" className={styles.image} draggable={false} />
      {boxes.map((box, i) => (
        <div
          key={i}
          className={`${styles.box} ${i === selectedIndex ? styles.selected : ""}`}
          style={boxToCssRect(box)}
          onMouseDown={(e) => startMove(i, e)}
        >
          <span className={styles.classLabel}>{box.class_id}</span>
          {i === selectedIndex && <div className={styles.resizeHandle} onMouseDown={(e) => startResize(i, e)} />}
        </div>
      ))}
      {previewBox && <div className={styles.drawPreview} style={boxToCssRect(previewBox)} />}
    </div>
  );
}
