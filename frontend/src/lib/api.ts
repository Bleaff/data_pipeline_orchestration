/** Typed client for the neudc control-plane API (#22), consumed by the read-only dashboard (#23). */

export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") ?? "http://localhost:8000";

export type PipelineStatus = "running" | "stopped";

export interface PipelineSummary {
  name: string;
  status: PipelineStatus;
}

export interface NodeConfig {
  id: string;
  type: string;
  outputs?: string[];
  [key: string]: unknown;
}

export interface PipelineDetail {
  name: string;
  status: PipelineStatus;
  nodes: NodeConfig[];
}

/** `{metric_name: {node_label: value}}`, as pushed by `/ws/metrics`. */
export type MetricsSnapshot = Record<string, Record<string, number>>;

// --- Node-type catalog (#24) --------------------------------------------------------

export type FieldKind = "string" | "integer" | "number" | "boolean" | "enum" | "model_config";

export interface NodeFieldSpec {
  name: string;
  kind: FieldKind;
  required: boolean;
  default: unknown;
  description: string;
  options: string[] | null;
}

export interface ModelTypeSpec {
  type: string;
  fields: NodeFieldSpec[];
}

export interface NodeTypeSpec {
  type: string;
  runs_as: "thread" | "process";
  description: string;
  accepts: string[];
  emits: string[];
  fields: NodeFieldSpec[];
  model_types: ModelTypeSpec[] | null;
}

export interface ConfigValidateResponse {
  valid: boolean;
  error: string | null;
}

// --- Pre-label review (#24) ---------------------------------------------------------

export interface BoxSpec {
  class_id: number;
  x_center: number;
  y_center: number;
  width: number;
  height: number;
}

export interface LabelFrameSummary {
  frame_id: string;
  box_count: number;
  reviewed: boolean;
}

export interface LabelFrameDetail {
  frame_id: string;
  boxes: BoxSpec[];
  reviewed: boolean;
}

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE_URL}${path}`);
  if (!res.ok) {
    throw new Error(`GET ${path} failed: ${res.status} ${res.statusText}`);
  }
  return (await res.json()) as T;
}

async function sendJson<T>(method: "POST" | "PUT", path: string, body: unknown): Promise<T> {
  const res = await fetch(`${API_BASE_URL}${path}`, {
    method,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => null);
    throw new Error(detail?.detail ?? `${method} ${path} failed: ${res.status} ${res.statusText}`);
  }
  return (await res.json()) as T;
}

function postJson<T>(path: string, body: unknown): Promise<T> {
  return sendJson<T>("POST", path, body);
}

function putJson<T>(path: string, body: unknown): Promise<T> {
  return sendJson<T>("PUT", path, body);
}

export function listPipelines(): Promise<PipelineSummary[]> {
  return getJson<PipelineSummary[]>("/pipelines");
}

export function getPipelineDetail(name: string): Promise<PipelineDetail> {
  return getJson<PipelineDetail>(`/pipelines/${encodeURIComponent(name)}`);
}

export function listNodeTypes(): Promise<NodeTypeSpec[]> {
  return getJson<NodeTypeSpec[]>("/node-types");
}

export function validateConfig(nodes: NodeConfig[]): Promise<ConfigValidateResponse> {
  return postJson<ConfigValidateResponse>("/config/validate", { nodes });
}

export function startPipeline(name: string, nodes: NodeConfig[]): Promise<PipelineSummary> {
  return postJson<PipelineSummary>(`/pipelines/${encodeURIComponent(name)}/start`, { nodes });
}

export function previewUrl(name: string): string {
  return `${API_BASE_URL}/pipelines/${encodeURIComponent(name)}/preview`;
}

export function listLabels(name: string): Promise<LabelFrameSummary[]> {
  return getJson<LabelFrameSummary[]>(`/pipelines/${encodeURIComponent(name)}/labels`);
}

export function getLabel(name: string, frameId: string): Promise<LabelFrameDetail> {
  return getJson<LabelFrameDetail>(`/pipelines/${encodeURIComponent(name)}/labels/${encodeURIComponent(frameId)}`);
}

export function labelImageUrl(name: string, frameId: string): string {
  return `${API_BASE_URL}/pipelines/${encodeURIComponent(name)}/labels/${encodeURIComponent(frameId)}/image`;
}

export function updateLabel(name: string, frameId: string, boxes: BoxSpec[]): Promise<LabelFrameDetail> {
  return putJson<LabelFrameDetail>(`/pipelines/${encodeURIComponent(name)}/labels/${encodeURIComponent(frameId)}`, {
    boxes,
  });
}

export function metricsSocketUrl(): string {
  return `${API_BASE_URL.replace(/^http/, "ws")}/ws/metrics`;
}
