import { describe, expect, it } from "vitest";
import { getNodeStats, healthState } from "../metrics";
import type { MetricsSnapshot } from "../api";

describe("getNodeStats", () => {
  it("reads all three metrics for a known node", () => {
    const snapshot: MetricsSnapshot = {
      neudc_node_health: { reader: 1 },
      neudc_node_messages_processed_total: { reader: 42 },
      neudc_node_queue_depth: { reader: 3 },
    };

    expect(getNodeStats(snapshot, "reader")).toEqual({
      health: 1,
      messagesProcessed: 42,
      queueDepth: 3,
    });
  });

  it("returns undefined fields for a node with no samples yet", () => {
    const snapshot: MetricsSnapshot = { neudc_node_health: { other: 1 } };

    expect(getNodeStats(snapshot, "reader")).toEqual({
      health: undefined,
      messagesProcessed: undefined,
      queueDepth: undefined,
    });
  });

  it("tolerates a snapshot missing a metric family entirely", () => {
    expect(getNodeStats({}, "reader")).toEqual({
      health: undefined,
      messagesProcessed: undefined,
      queueDepth: undefined,
    });
  });
});

describe("healthState", () => {
  it("reports unknown when no sample has arrived", () => {
    expect(healthState(undefined)).toBe("unknown");
  });

  it("reports healthy at 1", () => {
    expect(healthState(1)).toBe("healthy");
  });

  it("reports unhealthy at 0", () => {
    expect(healthState(0)).toBe("unhealthy");
  });
});
