import { describe, expect, it } from "vitest";
import { boxFromDrag, boxToCssRect, clampBox, moveBox, resizeBox } from "../boxGeometry";

describe("clampBox", () => {
  it("leaves a well-formed centered box untouched", () => {
    expect(clampBox({ x_center: 0.5, y_center: 0.5, width: 0.2, height: 0.2 })).toEqual({
      x_center: 0.5,
      y_center: 0.5,
      width: 0.2,
      height: 0.2,
    });
  });

  it("pulls a box back inside [0,1] when its center would push it out of bounds", () => {
    const box = clampBox({ x_center: -0.1, y_center: 1.2, width: 0.2, height: 0.2 });

    expect(box.x_center).toBeCloseTo(0.1);
    expect(box.y_center).toBeCloseTo(0.9);
  });

  it("floors width/height at the minimum size instead of collapsing to zero", () => {
    const box = clampBox({ x_center: 0.5, y_center: 0.5, width: 0, height: -1 });

    expect(box.width).toBeGreaterThan(0);
    expect(box.height).toBeGreaterThan(0);
  });

  it("caps width/height at 1", () => {
    const box = clampBox({ x_center: 0.5, y_center: 0.5, width: 5, height: 5 });

    expect(box.width).toBe(1);
    expect(box.height).toBe(1);
  });
});

describe("boxToCssRect", () => {
  it("converts center/size to a top-left percentage rect", () => {
    expect(boxToCssRect({ x_center: 0.5, y_center: 0.5, width: 0.2, height: 0.4 })).toEqual({
      left: "40%",
      top: "30%",
      width: "20%",
      height: "40%",
    });
  });
});

describe("moveBox", () => {
  it("translates the center by the given fraction", () => {
    const box = moveBox({ x_center: 0.5, y_center: 0.5, width: 0.2, height: 0.2 }, 0.1, -0.05);

    expect(box.x_center).toBeCloseTo(0.6);
    expect(box.y_center).toBeCloseTo(0.45);
    expect(box.width).toBeCloseTo(0.2);
  });

  it("stops at the edge instead of moving the box off-canvas", () => {
    const box = moveBox({ x_center: 0.9, y_center: 0.5, width: 0.2, height: 0.2 }, 0.5, 0);

    expect(box.x_center).toBeCloseTo(0.9); // already touching the right edge (0.9 + 0.1 = 1.0)
  });
});

describe("resizeBox", () => {
  it("grows from the top-left corner, keeping that corner fixed", () => {
    const original = { x_center: 0.5, y_center: 0.5, width: 0.2, height: 0.2 };
    const originalLeft = original.x_center - original.width / 2;
    const originalTop = original.y_center - original.height / 2;

    const resized = resizeBox(original, 0.1, 0.1);

    expect(resized.width).toBeCloseTo(0.3);
    expect(resized.height).toBeCloseTo(0.3);
    expect(resized.x_center - resized.width / 2).toBeCloseTo(originalLeft);
    expect(resized.y_center - resized.height / 2).toBeCloseTo(originalTop);
  });

  it("never shrinks below the minimum size", () => {
    const resized = resizeBox({ x_center: 0.5, y_center: 0.5, width: 0.05, height: 0.05 }, -1, -1);

    expect(resized.width).toBeGreaterThan(0);
    expect(resized.height).toBeGreaterThan(0);
  });
});

describe("boxFromDrag", () => {
  it("normalizes a drag regardless of drag direction", () => {
    const forward = boxFromDrag({ x: 0.2, y: 0.2 }, { x: 0.6, y: 0.5 }, 3);
    const backward = boxFromDrag({ x: 0.6, y: 0.5 }, { x: 0.2, y: 0.2 }, 3);

    expect(forward.class_id).toBe(3);
    expect(forward.x_center).toBeCloseTo(backward.x_center);
    expect(forward.y_center).toBeCloseTo(backward.y_center);
    expect(forward.width).toBeCloseTo(backward.width);
    expect(forward.height).toBeCloseTo(backward.height);
    expect(forward.x_center).toBeCloseTo(0.4);
    expect(forward.y_center).toBeCloseTo(0.35);
    expect(forward.width).toBeCloseTo(0.4);
    expect(forward.height).toBeCloseTo(0.3);
  });
});
