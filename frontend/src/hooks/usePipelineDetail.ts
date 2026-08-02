"use client";

import { useEffect, useState } from "react";
import { getPipelineDetail, type PipelineDetail } from "@/lib/api";

const POLL_INTERVAL_MS = 3000;

export interface PipelineDetailState {
  detail: PipelineDetail | null;
  error: string | null;
  loading: boolean;
}

/** Polls `GET /pipelines/{name}` so status + node config stay current while viewing a pipeline. */
export function usePipelineDetail(name: string): PipelineDetailState {
  const [detail, setDetail] = useState<PipelineDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;

    async function poll() {
      try {
        const result = await getPipelineDetail(name);
        if (!cancelled) {
          setDetail(result);
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
  }, [name]);

  return { detail, error, loading };
}
