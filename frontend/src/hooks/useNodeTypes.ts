"use client";

import { useEffect, useState } from "react";
import { listNodeTypes, type NodeTypeSpec } from "@/lib/api";

export interface NodeTypesState {
  nodeTypes: NodeTypeSpec[];
  error: string | null;
  loading: boolean;
}

/** Fetches the node-type catalog once — it's fixed for the API process's lifetime, no need to poll. */
export function useNodeTypes(): NodeTypesState {
  const [nodeTypes, setNodeTypes] = useState<NodeTypeSpec[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    listNodeTypes()
      .then((result) => {
        if (!cancelled) setNodeTypes(result);
      })
      .catch((err) => {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return { nodeTypes, error, loading };
}
