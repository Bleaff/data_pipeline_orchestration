import { describe, expect, it } from "vitest";
import {
  EMPTY_GRAPH,
  addEdge,
  addNode,
  moveNode,
  removeEdge,
  removeNode,
  toNodesConfig,
  updateNodeConfig,
} from "../builderGraph";

describe("addNode", () => {
  it("generates a short, readable id from the node type", () => {
    const graph = addNode(EMPTY_GRAPH, "FolderImageNode", { x: 0, y: 0 });

    expect(graph.nodes).toHaveLength(1);
    expect(graph.nodes[0].id).toBe("folder_image_1");
    expect(graph.nodes[0].type).toBe("FolderImageNode");
    expect(graph.nodes[0].config).toEqual({});
  });

  it("increments the suffix for repeated additions of the same type", () => {
    let graph = addNode(EMPTY_GRAPH, "SaveImageNode", { x: 0, y: 0 });
    graph = addNode(graph, "SaveImageNode", { x: 10, y: 10 });

    expect(graph.nodes.map((n) => n.id)).toEqual(["save_image_1", "save_image_2"]);
  });

  it("does not collide with an id after the node that held it is removed and re-added", () => {
    let graph = addNode(EMPTY_GRAPH, "DrawNode", { x: 0, y: 0 });
    graph = addNode(graph, "DrawNode", { x: 0, y: 0 });
    graph = removeNode(graph, "draw_1");
    graph = addNode(graph, "DrawNode", { x: 0, y: 0 });

    expect(graph.nodes.map((n) => n.id).sort()).toEqual(["draw_1", "draw_2"]);
  });
});

describe("removeNode", () => {
  it("also removes edges touching the removed node", () => {
    let graph = addNode(EMPTY_GRAPH, "FolderImageNode", { x: 0, y: 0 });
    graph = addNode(graph, "SaveImageNode", { x: 0, y: 0 });
    graph = addEdge(graph, "folder_image_1", "save_image_1");

    graph = removeNode(graph, "save_image_1");

    expect(graph.nodes.map((n) => n.id)).toEqual(["folder_image_1"]);
    expect(graph.edges).toEqual([]);
  });
});

describe("addEdge", () => {
  it("rejects a self-loop", () => {
    const graph = addNode(EMPTY_GRAPH, "DrawNode", { x: 0, y: 0 });

    const result = addEdge(graph, "draw_1", "draw_1");

    expect(result.edges).toEqual([]);
  });

  it("is idempotent for a duplicate edge", () => {
    let graph = addNode(EMPTY_GRAPH, "FolderImageNode", { x: 0, y: 0 });
    graph = addNode(graph, "SaveImageNode", { x: 0, y: 0 });
    graph = addEdge(graph, "folder_image_1", "save_image_1");

    graph = addEdge(graph, "folder_image_1", "save_image_1");

    expect(graph.edges).toEqual([{ source: "folder_image_1", target: "save_image_1" }]);
  });
});

describe("removeEdge", () => {
  it("removes only the matching edge", () => {
    let graph = addNode(EMPTY_GRAPH, "FolderImageNode", { x: 0, y: 0 });
    graph = addNode(graph, "SaveImageNode", { x: 0, y: 0 });
    graph = addNode(graph, "DrawNode", { x: 0, y: 0 });
    graph = addEdge(graph, "folder_image_1", "save_image_1");
    graph = addEdge(graph, "folder_image_1", "draw_1");

    graph = removeEdge(graph, "folder_image_1", "save_image_1");

    expect(graph.edges).toEqual([{ source: "folder_image_1", target: "draw_1" }]);
  });
});

describe("moveNode", () => {
  it("updates only the targeted node's position", () => {
    let graph = addNode(EMPTY_GRAPH, "DrawNode", { x: 0, y: 0 });
    graph = addNode(graph, "DrawNode", { x: 0, y: 0 });

    graph = moveNode(graph, "draw_1", { x: 42, y: 99 });

    expect(graph.nodes[0].position).toEqual({ x: 42, y: 99 });
    expect(graph.nodes[1].position).toEqual({ x: 0, y: 0 });
  });
});

describe("updateNodeConfig", () => {
  it("sets a field on the targeted node without touching others", () => {
    let graph = addNode(EMPTY_GRAPH, "FolderImageNode", { x: 0, y: 0 });
    graph = addNode(graph, "SaveImageNode", { x: 0, y: 0 });

    graph = updateNodeConfig(graph, "folder_image_1", "folder_path", "/data/in");

    expect(graph.nodes[0].config).toEqual({ folder_path: "/data/in" });
    expect(graph.nodes[1].config).toEqual({});
  });
});

describe("toNodesConfig", () => {
  it("flattens each node's config alongside id/type/outputs", () => {
    let graph = addNode(EMPTY_GRAPH, "FolderImageNode", { x: 0, y: 0 });
    graph = addNode(graph, "SaveImageNode", { x: 0, y: 0 });
    graph = addEdge(graph, "folder_image_1", "save_image_1");
    graph = updateNodeConfig(graph, "folder_image_1", "folder_path", "/data/in");
    graph = updateNodeConfig(graph, "save_image_1", "save_dir", "/data/out");

    const nodesConfig = toNodesConfig(graph);

    expect(nodesConfig).toEqual([
      { id: "folder_image_1", type: "FolderImageNode", outputs: ["save_image_1"], folder_path: "/data/in" },
      { id: "save_image_1", type: "SaveImageNode", outputs: [], save_dir: "/data/out" },
    ]);
  });
});
