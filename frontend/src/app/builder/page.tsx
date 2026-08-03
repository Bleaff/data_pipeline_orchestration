"use client";

import { useMemo, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { startPipeline, validateConfig, type ConfigValidateResponse } from "@/lib/api";
import {
  EMPTY_GRAPH,
  addEdge,
  addNode,
  moveNode,
  removeNode,
  toNodesConfig,
  updateNodeConfig,
  type BuilderGraph,
} from "@/lib/builderGraph";
import { toPipelineYaml } from "@/lib/yamlExport";
import { useNodeTypes } from "@/hooks/useNodeTypes";
import { NodePalette } from "@/components/NodePalette";
import { BuilderCanvas } from "@/components/BuilderCanvas";
import { NodeConfigForm } from "@/components/NodeConfigForm";
import styles from "./page.module.css";

const NEW_NODE_OFFSET = 40;

export default function BuilderPage() {
  const router = useRouter();
  const { nodeTypes, error: catalogError, loading: catalogLoading } = useNodeTypes();

  const [graph, setGraph] = useState<BuilderGraph>(EMPTY_GRAPH);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [showYaml, setShowYaml] = useState(false);
  const [validation, setValidation] = useState<ConfigValidateResponse | null>(null);
  const [validating, setValidating] = useState(false);
  const [pipelineName, setPipelineName] = useState("");
  const [starting, setStarting] = useState(false);
  const [startError, setStartError] = useState<string | null>(null);

  const nodesConfig = useMemo(() => toNodesConfig(graph), [graph]);
  const yaml = useMemo(() => toPipelineYaml(nodesConfig), [nodesConfig]);
  const selectedNode = graph.nodes.find((n) => n.id === selectedNodeId) ?? null;
  const selectedNodeType = selectedNode ? nodeTypes.find((nt) => nt.type === selectedNode.type) : undefined;

  function handleAddNode(nodeType: string) {
    const position = { x: (graph.nodes.length % 5) * 220, y: Math.floor(graph.nodes.length / 5) * NEW_NODE_OFFSET * 3 };
    const next = addNode(graph, nodeType, position);
    setGraph(next);
    setSelectedNodeId(next.nodes[next.nodes.length - 1].id);
    setValidation(null);
  }

  function handleFieldChange(key: string, value: unknown) {
    if (!selectedNodeId) return;
    setGraph((g) => updateNodeConfig(g, selectedNodeId, key, value));
    setValidation(null);
  }

  async function handleValidate() {
    setValidating(true);
    try {
      const result = await validateConfig(nodesConfig);
      setValidation(result);
    } catch (err) {
      setValidation({ valid: false, error: err instanceof Error ? err.message : String(err) });
    } finally {
      setValidating(false);
    }
  }

  async function handleStart() {
    if (!pipelineName.trim()) {
      setStartError("Pipeline name is required");
      return;
    }
    setStarting(true);
    setStartError(null);
    try {
      await startPipeline(pipelineName.trim(), nodesConfig);
      router.push(`/pipelines/${encodeURIComponent(pipelineName.trim())}`);
    } catch (err) {
      setStartError(err instanceof Error ? err.message : String(err));
    } finally {
      setStarting(false);
    }
  }

  return (
    <main className={styles.main}>
      <div className={styles.topBar}>
        <Link href="/" className={styles.back}>
          ← All pipelines
        </Link>
        <h1>Config builder</h1>
      </div>

      {catalogError && <p className={styles.error}>Couldn't load the node-type catalog: {catalogError}</p>}

      {!catalogError && (
        <div className={styles.layout}>
          <aside className={styles.sidebar}>
            {catalogLoading ? <p>Loading node types…</p> : <NodePalette nodeTypes={nodeTypes} onAddNode={handleAddNode} />}
          </aside>

          <div className={styles.canvas}>
            <BuilderCanvas
              graph={graph}
              selectedNodeId={selectedNodeId}
              onSelectNode={setSelectedNodeId}
              onConnect={(source, target) => {
                setGraph((g) => addEdge(g, source, target));
                setValidation(null);
              }}
              onMoveNode={(nodeId, position) => setGraph((g) => moveNode(g, nodeId, position))}
              onDeleteNode={(nodeId) => {
                setGraph((g) => removeNode(g, nodeId));
                if (nodeId === selectedNodeId) setSelectedNodeId(null);
                setValidation(null);
              }}
            />
          </div>

          <aside className={styles.inspector}>
            <h2 className={styles.heading}>{selectedNode ? selectedNode.id : "No node selected"}</h2>
            {selectedNode && selectedNodeType && (
              <NodeConfigForm nodeType={selectedNodeType} config={selectedNode.config} onFieldChange={handleFieldChange} />
            )}
          </aside>
        </div>
      )}

      <section className={styles.footer}>
        <div className={styles.actions}>
          <button type="button" onClick={handleValidate} disabled={validating || graph.nodes.length === 0}>
            {validating ? "Validating…" : "Validate"}
          </button>
          <button type="button" onClick={() => setShowYaml((v) => !v)}>
            {showYaml ? "Hide YAML" : "Show YAML"}
          </button>
          <input
            type="text"
            placeholder="pipeline name"
            value={pipelineName}
            onChange={(e) => setPipelineName(e.target.value)}
            className={styles.nameInput}
          />
          <button type="button" onClick={handleStart} disabled={starting || graph.nodes.length === 0}>
            {starting ? "Starting…" : "Start pipeline"}
          </button>
        </div>

        {validation && (
          <p className={validation.valid ? styles.valid : styles.invalid}>
            {validation.valid ? "Config is valid." : validation.error}
          </p>
        )}
        {startError && <p className={styles.invalid}>{startError}</p>}

        {showYaml && <pre className={styles.yaml}>{yaml}</pre>}
      </section>
    </main>
  );
}
