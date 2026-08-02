import styles from "./StatusBadge.module.css";

type Tone = "running" | "stopped" | "healthy" | "unhealthy" | "unknown";

const LABELS: Record<Tone, string> = {
  running: "running",
  stopped: "stopped",
  healthy: "healthy",
  unhealthy: "unhealthy",
  unknown: "unknown",
};

export function StatusBadge({ tone }: { tone: Tone }) {
  return (
    <span className={`${styles.badge} ${styles[tone]}`}>
      <span className={styles.dot} />
      {LABELS[tone]}
    </span>
  );
}
