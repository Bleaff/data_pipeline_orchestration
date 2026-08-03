"use client";

import { useMemo } from "react";
import { Background, Controls, ReactFlow, type Connection, type Edge, type Node, type NodeChange } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import type { BuilderGraph } from "@/lib/builderGraph";

const NODE_STYLE = {
  fontSize: 12,
  fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
  borderRadius: 8,
  padding: 8,
};

export function BuilderCanvas({
  graph,
  selectedNodeId,
  onSelectNode,
  onConnect,
  onMoveNode,
  onDeleteNode,
}: {
  graph: BuilderGraph;
  selectedNodeId: string | null;
  onSelectNode: (nodeId: string | null) => void;
  onConnect: (source: string, target: string) => void;
  onMoveNode: (nodeId: string, position: { x: number; y: number }) => void;
  onDeleteNode: (nodeId: string) => void;
}) {
  const { nodes, edges } = useMemo(() => {
    const flowNodes: Node[] = graph.nodes.map((n) => ({
      id: n.id,
      position: n.position,
      data: { label: `${n.id}\n${n.type}` },
      style: {
        ...NODE_STYLE,
        whiteSpace: "pre-line" as const,
        border: n.id === selectedNodeId ? "2px solid #2e7bd1" : "1px solid var(--border)",
      },
    }));
    const flowEdges: Edge[] = graph.edges.map((e) => ({
      id: `${e.source}->${e.target}`,
      source: e.source,
      target: e.target,
    }));
    return { nodes: flowNodes, edges: flowEdges };
  }, [graph, selectedNodeId]);

  function handleConnect(connection: Connection) {
    if (connection.source && connection.target) {
      onConnect(connection.source, connection.target);
    }
  }

  function handleNodesChange(changes: NodeChange[]) {
    for (const change of changes) {
      if (change.type === "position" && change.position) {
        onMoveNode(change.id, change.position);
      } else if (change.type === "remove") {
        onDeleteNode(change.id);
      } else if (change.type === "select" && change.selected) {
        onSelectNode(change.id);
      }
    }
  }

  return (
    <div style={{ height: "100%", border: "1px solid var(--border)", borderRadius: 8 }}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onConnect={handleConnect}
        onNodesChange={handleNodesChange}
        onPaneClick={() => onSelectNode(null)}
        fitView
        proOptions={{ hideAttribution: true }}
      >
        <Background />
        <Controls showInteractive={false} />
      </ReactFlow>
    </div>
  );
}
