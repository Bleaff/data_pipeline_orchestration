"use client";

import { useEffect, useRef, useState } from "react";
import { previewUrl } from "@/lib/api";
import { appendFrame, type FrameEntry } from "@/lib/frameFeed";

const POLL_INTERVAL_MS = 1000;
const MAX_FEED_LENGTH = 12;

export interface FrameFeedState {
  frames: FrameEntry[];
  /** True once at least one successful preview response has been seen (vs. "no SaveImageNode yet"). */
  hasEverSeenAFrame: boolean;
}

/**
 * Polls `/pipelines/{name}/preview` and turns repeated "latest frame" snapshots into a
 * feed of distinct recent frames (object URLs, revoked once evicted from the feed).
 */
export function useFrameFeed(name: string): FrameFeedState {
  const [frames, setFrames] = useState<FrameEntry[]>([]);
  const [hasEverSeenAFrame, setHasEverSeenAFrame] = useState(false);
  const framesRef = useRef<FrameEntry[]>([]);

  useEffect(() => {
    let cancelled = false;

    async function poll() {
      try {
        const res = await fetch(previewUrl(name), { cache: "no-store" });
        if (!res.ok || cancelled) return;

        const frameName = res.headers.get("x-frame-name") ?? String(Date.now());
        const current = framesRef.current;
        if (current.length > 0 && current[0].name === frameName) {
          return;
        }

        const blob = await res.blob();
        const entry: FrameEntry = { name: frameName, url: URL.createObjectURL(blob), capturedAt: Date.now() };
        const next = appendFrame(current, entry, MAX_FEED_LENGTH);

        for (const dropped of current) {
          if (!next.includes(dropped)) URL.revokeObjectURL(dropped.url);
        }

        framesRef.current = next;
        if (!cancelled) {
          setFrames(next);
          setHasEverSeenAFrame(true);
        }
      } catch {
        // Pipeline not started / no SaveImageNode yet: leave the existing feed as-is.
      }
    }

    poll();
    const interval = setInterval(poll, POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      clearInterval(interval);
      for (const frame of framesRef.current) URL.revokeObjectURL(frame.url);
      framesRef.current = [];
    };
  }, [name]);

  return { frames, hasEverSeenAFrame };
}
