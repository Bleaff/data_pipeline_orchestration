"use client";

import { useEffect, useState } from "react";
import { listPipelines, type PipelineSummary } from "@/lib/api";

const POLL_INTERVAL_MS = 2000;

export interface PipelinesState {
  pipelines: PipelineSummary[];
  error: string | null;
  loading: boolean;
}

/** Polls `GET /pipelines` on an interval; the control-plane has no list-changed push feed yet. */
export function usePipelines(): PipelinesState {
  const [pipelines, setPipelines] = useState<PipelineSummary[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;

    async function poll() {
      try {
        const result = await listPipelines();
        if (!cancelled) {
          setPipelines(result);
          setError(null);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : String(err));
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    poll();
    const interval = setInterval(poll, POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  return { pipelines, error, loading };
}
