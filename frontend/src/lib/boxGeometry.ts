/** Pure geometry for editing YOLO-normalized (0-1) boxes over an image (#24). */

export interface NormalizedBox {
  x_center: number;
  y_center: number;
  width: number;
  height: number;
}

const MIN_SIZE = 0.01;

/** Keep a box's center/size within [0,1] and above a sane minimum size. */
export function clampBox(box: NormalizedBox): NormalizedBox {
  const width = Math.min(1, Math.max(MIN_SIZE, box.width));
  const height = Math.min(1, Math.max(MIN_SIZE, box.height));
  const x_center = Math.min(1 - width / 2, Math.max(width / 2, box.x_center));
  const y_center = Math.min(1 - height / 2, Math.max(height / 2, box.y_center));
  return { x_center, y_center, width, height };
}

/** CSS percentage rect (top-left + size) for a normalized box, for absolute positioning over an image. */
export function boxToCssRect(box: NormalizedBox): { left: string; top: string; width: string; height: string } {
  const left = box.x_center - box.width / 2;
  const top = box.y_center - box.height / 2;
  return {
    left: `${left * 100}%`,
    top: `${top * 100}%`,
    width: `${box.width * 100}%`,
    height: `${box.height * 100}%`,
  };
}

/** Translate a box by a fraction of the container's width/height (e.g. from a drag delta). */
export function moveBox(box: NormalizedBox, dxFrac: number, dyFrac: number): NormalizedBox {
  return clampBox({ ...box, x_center: box.x_center + dxFrac, y_center: box.y_center + dyFrac });
}

/** Resize a box by a fraction of the container's width/height, anchored at its top-left corner. */
export function resizeBox(box: NormalizedBox, dwFrac: number, dhFrac: number): NormalizedBox {
  const width = Math.max(MIN_SIZE, box.width + dwFrac);
  const height = Math.max(MIN_SIZE, box.height + dhFrac);
  // Growing/shrinking from the top-left corner keeps that corner fixed, so the center shifts by half the delta.
  return clampBox({
    x_center: box.x_center + (width - box.width) / 2,
    y_center: box.y_center + (height - box.height) / 2,
    width,
    height,
  });
}

/** Build a new box from a click-drag gesture, both points as fractions [0,1] of the container. */
export function boxFromDrag(
  start: { x: number; y: number },
  end: { x: number; y: number },
  classId: number,
): NormalizedBox & { class_id: number } {
  const x1 = Math.min(start.x, end.x);
  const x2 = Math.max(start.x, end.x);
  const y1 = Math.min(start.y, end.y);
  const y2 = Math.max(start.y, end.y);
  return { class_id: classId, ...clampBox({ x_center: (x1 + x2) / 2, y_center: (y1 + y2) / 2, width: x2 - x1, height: y2 - y1 }) };
}
