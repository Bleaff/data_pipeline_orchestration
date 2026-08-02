/** Pure helpers for reading a node's stats out of a `/ws/metrics` snapshot. */

import type { MetricsSnapshot } from "./api";

export interface NodeStats {
  /** 1 = healthy, 0 = unhealthy, undefined = no sample yet (e.g. node not started). */
  health: number | undefined;
  /** Cumulative count from `neudc_node_messages_processed_total`. */
  messagesProcessed: number | undefined;
  /** Current value of `neudc_node_queue_depth`. */
  queueDepth: number | undefined;
}

const HEALTH_METRIC = "neudc_node_health";
const MESSAGES_PROCESSED_METRIC = "neudc_node_messages_processed_total";
const QUEUE_DEPTH_METRIC = "neudc_node_queue_depth";

export function getNodeStats(snapshot: MetricsSnapshot, nodeId: string): NodeStats {
  return {
    health: snapshot[HEALTH_METRIC]?.[nodeId],
    messagesProcessed: snapshot[MESSAGES_PROCESSED_METRIC]?.[nodeId],
    queueDepth: snapshot[QUEUE_DEPTH_METRIC]?.[nodeId],
  };
}

export type HealthState = "healthy" | "unhealthy" | "unknown";

export function healthState(health: number | undefined): HealthState {
  if (health === undefined) return "unknown";
  return health >= 1 ? "healthy" : "unhealthy";
}
