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

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE_URL}${path}`);
  if (!res.ok) {
    throw new Error(`GET ${path} failed: ${res.status} ${res.statusText}`);
  }
  return (await res.json()) as T;
}

export function listPipelines(): Promise<PipelineSummary[]> {
  return getJson<PipelineSummary[]>("/pipelines");
}

export function getPipelineDetail(name: string): Promise<PipelineDetail> {
  return getJson<PipelineDetail>(`/pipelines/${encodeURIComponent(name)}`);
}

export function previewUrl(name: string): string {
  return `${API_BASE_URL}/pipelines/${encodeURIComponent(name)}/preview`;
}

export function metricsSocketUrl(): string {
  return `${API_BASE_URL.replace(/^http/, "ws")}/ws/metrics`;
}
