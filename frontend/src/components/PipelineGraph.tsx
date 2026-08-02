"use client";

import { useMemo } from "react";
import { Background, Controls, ReactFlow, type Edge, type Node } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import type { MetricsSnapshot, NodeConfig } from "@/lib/api";
import { layoutNodes } from "@/lib/graphLayout";
import { getNodeStats, healthState } from "@/lib/metrics";

const HEALTH_COLORS: Record<string, string> = {
  healthy: "#2e9e5b",
  unhealthy: "#d1453b",
  unknown: "#8a8a8a",
};

function nodeLabel(node: NodeConfig, snapshot: MetricsSnapshot): string {
  const stats = getNodeStats(snapshot, node.id);
  const lines = [node.id, node.type];
  if (stats.queueDepth !== undefined) lines.push(`queue: ${stats.queueDepth}`);
  if (stats.messagesProcessed !== undefined) lines.push(`processed: ${stats.messagesProcessed}`);
  return lines.join("\n");
}

export function PipelineGraph({ nodes: nodeConfigs, metrics }: { nodes: NodeConfig[]; metrics: MetricsSnapshot }) {
  const { nodes, edges } = useMemo(() => {
    const positions = new Map(layoutNodes(nodeConfigs).map((p) => [p.id, p]));

    const flowNodes: Node[] = nodeConfigs.map((config) => {
      const pos = positions.get(config.id) ?? { x: 0, y: 0 };
      const state = healthState(getNodeStats(metrics, config.id).health);
      return {
        id: config.id,
        position: { x: pos.x, y: pos.y },
        data: { label: nodeLabel(config, metrics) },
        style: {
          whiteSpace: "pre-line",
          fontSize: 12,
          fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
          border: `2px solid ${HEALTH_COLORS[state]}`,
          borderRadius: 8,
          padding: 8,
        },
      };
    });

    const idSet = new Set(nodeConfigs.map((n) => n.id));
    const flowEdges: Edge[] = nodeConfigs.flatMap((config) =>
      (config.outputs ?? [])
        .filter((target) => idSet.has(target))
        .map((target) => ({
          id: `${config.id}->${target}`,
          source: config.id,
          target,
        })),
    );

    return { nodes: flowNodes, edges: flowEdges };
  }, [nodeConfigs, metrics]);

  return (
    <div style={{ height: 420, border: "1px solid var(--border)", borderRadius: 8 }}>
      <ReactFlow nodes={nodes} edges={edges} fitView proOptions={{ hideAttribution: true }}>
        <Background />
        <Controls showInteractive={false} />
      </ReactFlow>
    </div>
  );
}
