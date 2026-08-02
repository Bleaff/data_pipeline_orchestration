/** Pure left-to-right layered layout for the pipeline graph view, by longest path from a root. */

import type { NodeConfig } from "./api";

export interface LayoutPosition {
  id: string;
  x: number;
  y: number;
}

const LAYER_WIDTH = 220;
const ROW_HEIGHT = 100;

/**
 * Assign each node a layer (its longest distance from any root) and a row within that
 * layer, then convert to pixel coordinates. Nodes not reachable from a root (e.g. stray
 * ids, or a cycle the config validator should have already rejected) fall back to layer 0.
 */
export function layoutNodes(nodes: NodeConfig[]): LayoutPosition[] {
  const ids = nodes.map((n) => n.id);
  const idSet = new Set(ids);
  const outputsOf = new Map(nodes.map((n) => [n.id, (n.outputs ?? []).filter((o) => idSet.has(o))]));
  const hasIncoming = new Set<string>();
  for (const outs of outputsOf.values()) {
    for (const target of outs) hasIncoming.add(target);
  }

  const layer = new Map<string, number>();
  const queue: string[] = ids.filter((id) => !hasIncoming.has(id));
  for (const id of queue) layer.set(id, 0);

  // Bounded so a cyclic config (which the graph shouldn't have, but this must not
  // trust) can't grow `queue` forever instead of just producing an imperfect layout.
  const maxSteps = ids.length * ids.length + ids.length;
  let head = 0;
  let steps = 0;
  while (head < queue.length && steps < maxSteps) {
    const id = queue[head++];
    const currentLayer = layer.get(id) ?? 0;
    for (const target of outputsOf.get(id) ?? []) {
      const candidate = currentLayer + 1;
      if (candidate > (layer.get(target) ?? -1)) {
        layer.set(target, candidate);
        queue.push(target);
      }
      steps++;
    }
  }
  // Any node never reached (e.g. purely cyclic) still needs a position.
  for (const id of ids) {
    if (!layer.has(id)) layer.set(id, 0);
  }

  const rowInLayer = new Map<number, number>();
  return ids.map((id) => {
    const l = layer.get(id) ?? 0;
    const row = rowInLayer.get(l) ?? 0;
    rowInLayer.set(l, row + 1);
    return { id, x: l * LAYER_WIDTH, y: row * ROW_HEIGHT };
  });
}
