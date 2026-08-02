import { describe, expect, it } from "vitest";
import { layoutNodes } from "../graphLayout";
import type { NodeConfig } from "../api";

function node(id: string, outputs: string[] = []): NodeConfig {
  return { id, type: "Test", outputs };
}

describe("layoutNodes", () => {
  it("places a linear chain in increasing layers", () => {
    const nodes = [node("a", ["b"]), node("b", ["c"]), node("c", [])];

    const positions = layoutNodes(nodes);

    const byId = Object.fromEntries(positions.map((p) => [p.id, p]));
    expect(byId.a.x).toBeLessThan(byId.b.x);
    expect(byId.b.x).toBeLessThan(byId.c.x);
  });

  it("gives siblings the same layer but different rows", () => {
    const nodes = [node("reader", ["left", "right"]), node("left", []), node("right", [])];

    const positions = layoutNodes(nodes);
    const byId = Object.fromEntries(positions.map((p) => [p.id, p]));

    expect(byId.left.x).toBe(byId.right.x);
    expect(byId.left.y).not.toBe(byId.right.y);
  });

  it("places a diamond's join node after both of its parents", () => {
    const nodes = [node("a", ["b", "c"]), node("b", ["d"]), node("c", ["d"]), node("d", [])];

    const positions = layoutNodes(nodes);
    const byId = Object.fromEntries(positions.map((p) => [p.id, p]));

    expect(byId.d.x).toBeGreaterThan(byId.b.x);
    expect(byId.d.x).toBeGreaterThan(byId.c.x);
  });

  it("ignores an output pointing at an id not in the node list", () => {
    const nodes = [node("a", ["ghost"])];

    expect(() => layoutNodes(nodes)).not.toThrow();
    expect(layoutNodes(nodes).map((p) => p.id)).toEqual(["a"]);
  });

  it("terminates and positions every node even when the graph has a cycle", () => {
    const nodes = [node("a", ["b"]), node("b", ["a"])];

    const positions = layoutNodes(nodes);

    expect(positions.map((p) => p.id).sort()).toEqual(["a", "b"]);
  });

  it("returns an empty layout for an empty node list", () => {
    expect(layoutNodes([])).toEqual([]);
  });
});
