"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { usePipelineDetail } from "@/hooks/usePipelineDetail";
import { useMetricsSocket } from "@/hooks/useMetricsSocket";
import { useFrameFeed } from "@/hooks/useFrameFeed";
import { PipelineGraph } from "@/components/PipelineGraph";
import { FrameFeed } from "@/components/FrameFeed";
import { StatusBadge } from "@/components/StatusBadge";
import styles from "./page.module.css";

export default function PipelineDetailPage() {
  const { name } = useParams<{ name: string }>();
  const { detail, error, loading } = usePipelineDetail(name);
  const { snapshot, connected } = useMetricsSocket();
  const { frames, hasEverSeenAFrame } = useFrameFeed(name);

  return (
    <main className={styles.main}>
      <Link href="/" className={styles.back}>
        ← All pipelines
      </Link>

      <div className={styles.header}>
        <h1>{name}</h1>
        {detail && <StatusBadge tone={detail.status} />}
        <span className={styles.wsIndicator}>metrics feed: {connected ? "connected" : "reconnecting…"}</span>
      </div>

      {error && <p className={styles.error}>{error}</p>}
      {!error && loading && <p>Loading…</p>}

      {detail && (
        <>
          <section>
            <h2>Graph</h2>
            <PipelineGraph nodes={detail.nodes} metrics={snapshot} />
          </section>

          <section>
            <h2>Recent frames</h2>
            <FrameFeed frames={frames} hasEverSeenAFrame={hasEverSeenAFrame} />
          </section>
        </>
      )}
    </main>
  );
}
