"use client";

import { useEffect, useRef, useState } from "react";
import { metricsSocketUrl, type MetricsSnapshot } from "@/lib/api";

const RECONNECT_DELAY_MS = 2000;

export interface MetricsSocketState {
  snapshot: MetricsSnapshot;
  connected: boolean;
}

/** Subscribes to `/ws/metrics` and reconnects with a fixed backoff if the socket drops. */
export function useMetricsSocket(): MetricsSocketState {
  const [snapshot, setSnapshot] = useState<MetricsSnapshot>({});
  const [connected, setConnected] = useState(false);
  const closingRef = useRef(false);

  useEffect(() => {
    closingRef.current = false;
    let socket: WebSocket | null = null;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;

    function connect() {
      socket = new WebSocket(metricsSocketUrl());

      socket.onopen = () => setConnected(true);
      socket.onmessage = (event) => {
        try {
          setSnapshot(JSON.parse(event.data) as MetricsSnapshot);
        } catch {
          // Malformed frame: drop it and keep the previous snapshot.
        }
      };
      socket.onclose = () => {
        setConnected(false);
        if (!closingRef.current) {
          reconnectTimer = setTimeout(connect, RECONNECT_DELAY_MS);
        }
      };
      socket.onerror = () => socket?.close();
    }

    connect();
    return () => {
      closingRef.current = true;
      if (reconnectTimer) clearTimeout(reconnectTimer);
      socket?.close();
    };
  }, []);

  return { snapshot, connected };
}
