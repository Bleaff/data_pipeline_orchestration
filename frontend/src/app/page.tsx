"use client";

import Link from "next/link";
import { usePipelines } from "@/hooks/usePipelines";
import { StatusBadge } from "@/components/StatusBadge";
import styles from "./page.module.css";

export default function DashboardPage() {
  const { pipelines, error, loading } = usePipelines();

  return (
    <main className={styles.main}>
      <h1>neudc control plane</h1>
      <p className={styles.subtitle}>Read-only dashboard over the pipeline API.</p>

      {error && <p className={styles.error}>Could not reach the control-plane API: {error}</p>}
      {!error && loading && <p>Loading pipelines…</p>}
      {!error && !loading && pipelines.length === 0 && <p>No pipelines are currently managed.</p>}

      <ul className={styles.list}>
        {pipelines.map((pipeline) => (
          <li key={pipeline.name} className={styles.row}>
            <Link href={`/pipelines/${encodeURIComponent(pipeline.name)}`}>{pipeline.name}</Link>
            <StatusBadge tone={pipeline.status} />
          </li>
        ))}
      </ul>
    </main>
  );
}
