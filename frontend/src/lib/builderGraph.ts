/** Pure state management for the visual config builder's editable graph (#24). */

import type { NodeConfig } from "./api";

export interface BuilderNode {
  id: string;
  type: string;
  position: { x: number; y: number };
  /** Field values keyed by field name (e.g. `folder_path`, or `model_config` as a nested object). */
  config: Record<string, unknown>;
}

export interface BuilderEdge {
  source: string;
  target: string;
}

export interface BuilderGraph {
  nodes: BuilderNode[];
  edges: BuilderEdge[];
}

export const EMPTY_GRAPH: BuilderGraph = { nodes: [], edges: [] };

/** `"FolderImageNode"` -> `"reader_1"`, `"reader_2"`, ... — short, readable, unique node ids. */
function nextNodeId(graph: BuilderGraph, nodeType: string): string {
  const prefix = nodeType.replace(/Node$/, "").replace(/([a-z])([A-Z])/g, "$1_$2").toLowerCase();
  const existing = new Set(graph.nodes.map((n) => n.id));
  let i = 1;
  let id = `${prefix}_${i}`;
  while (existing.has(id)) {
    i += 1;
    id = `${prefix}_${i}`;
  }
  return id;
}

export function addNode(graph: BuilderGraph, nodeType: string, position: { x: number; y: number }): BuilderGraph {
  const id = nextNodeId(graph, nodeType);
  const node: BuilderNode = { id, type: nodeType, position, config: {} };
  return { ...graph, nodes: [...graph.nodes, node] };
}

export function removeNode(graph: BuilderGraph, nodeId: string): BuilderGraph {
  return {
    nodes: graph.nodes.filter((n) => n.id !== nodeId),
    edges: graph.edges.filter((e) => e.source !== nodeId && e.target !== nodeId),
  };
}

export function moveNode(graph: BuilderGraph, nodeId: string, position: { x: number; y: number }): BuilderGraph {
  return { ...graph, nodes: graph.nodes.map((n) => (n.id === nodeId ? { ...n, position } : n)) };
}

export function addEdge(graph: BuilderGraph, source: string, target: string): BuilderGraph {
  if (source === target) return graph;
  const exists = graph.edges.some((e) => e.source === source && e.target === target);
  if (exists) return graph;
  return { ...graph, edges: [...graph.edges, { source, target }] };
}

export function removeEdge(graph: BuilderGraph, source: string, target: string): BuilderGraph {
  return { ...graph, edges: graph.edges.filter((e) => !(e.source === source && e.target === target)) };
}

export function updateNodeConfig(graph: BuilderGraph, nodeId: string, key: string, value: unknown): BuilderGraph {
  return {
    ...graph,
    nodes: graph.nodes.map((n) => (n.id === nodeId ? { ...n, config: { ...n.config, [key]: value } } : n)),
  };
}

/** Convert the builder's internal graph into the flat node-config array the API expects. */
export function toNodesConfig(graph: BuilderGraph): NodeConfig[] {
  return graph.nodes.map((node) => ({
    id: node.id,
    type: node.type,
    outputs: graph.edges.filter((e) => e.source === node.id).map((e) => e.target),
    ...node.config,
  }));
}
