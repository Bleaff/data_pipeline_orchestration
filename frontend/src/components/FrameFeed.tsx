import type { FrameEntry } from "@/lib/frameFeed";
import styles from "./FrameFeed.module.css";

export function FrameFeed({ frames, hasEverSeenAFrame }: { frames: FrameEntry[]; hasEverSeenAFrame: boolean }) {
  if (!hasEverSeenAFrame) {
    return <p className={styles.empty}>No frames yet — this pipeline has no SaveImageNode output.</p>;
  }

  return (
    <div className={styles.strip}>
      {frames.map((frame, i) => (
        // eslint-disable-next-line @next/next/no-img-element -- object URLs aren't next/image-optimizable sources.
        <img
          key={frame.name}
          src={frame.url}
          alt={`Frame ${frame.name}`}
          className={i === 0 ? styles.latest : styles.thumbnail}
        />
      ))}
    </div>
  );
}
