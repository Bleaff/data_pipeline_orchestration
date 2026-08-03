"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useParams } from "next/navigation";
import { getLabel, labelImageUrl, listLabels, updateLabel, type BoxSpec, type LabelFrameSummary } from "@/lib/api";
import { BoxOverlay } from "@/components/BoxOverlay";
import styles from "./page.module.css";

export default function ReviewPage() {
  const { name } = useParams<{ name: string }>();

  const [frames, setFrames] = useState<LabelFrameSummary[]>([]);
  const [listError, setListError] = useState<string | null>(null);
  const [listLoading, setListLoading] = useState(true);
  const [unreviewedOnly, setUnreviewedOnly] = useState(true);

  const [index, setIndex] = useState(0);
  const [boxes, setBoxes] = useState<BoxSpec[]>([]);
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);

  const visibleFrames = useMemo(
    () => (unreviewedOnly ? frames.filter((f) => !f.reviewed) : frames),
    [frames, unreviewedOnly],
  );
  const currentFrame = visibleFrames[index] ?? null;

  const loadFrames = useCallback(() => {
    // No explicit setListLoading(true) here: the initial state is already `true`, and
    // this only ever runs once per mount, so there's nothing to reset (matches
    // usePipelines.ts's pattern). Resetting it *would* need to happen synchronously
    // before the fetch, which React's set-state-in-effect lint rule flags.
    listLabels(name)
      .then((result) => {
        setFrames(result);
        setListError(null);
      })
      .catch((err) => setListError(err instanceof Error ? err.message : String(err)))
      .finally(() => setListLoading(false));
  }, [name]);

  useEffect(() => {
    loadFrames();
  }, [loadFrames]);

  // Resetting box/selection/loading state when the viewed frame changes has to happen
  // synchronously with that change, not "eventually" from inside an effect — so this
  // follows React's documented pattern for it: a plain conditional during render,
  // comparing against the last-seen frame id, rather than a useEffect. Only the actual
  // fetch (and its `.then` continuation) stays in an effect below.
  const [loadedFrameId, setLoadedFrameId] = useState<string | null>(null);
  if (currentFrame && currentFrame.frame_id !== loadedFrameId) {
    setLoadedFrameId(currentFrame.frame_id);
    setBoxes([]);
    setSelectedIndex(null);
    setDetailLoading(true);
  }

  useEffect(() => {
    if (!currentFrame) return;
    getLabel(name, currentFrame.frame_id)
      .then((detail) => setBoxes(detail.boxes))
      .finally(() => setDetailLoading(false));
  }, [name, currentFrame]);

  async function handleSave(advance: boolean) {
    if (!currentFrame) return;
    setSaving(true);
    setSaveError(null);
    try {
      await updateLabel(name, currentFrame.frame_id, boxes);
      setFrames((prev) => prev.map((f) => (f.frame_id === currentFrame.frame_id ? { ...f, reviewed: true } : f)));
      // When filtering to unreviewed-only, marking this frame reviewed drops it from
      // `visibleFrames` on the next render, so the same `index` already lands on
      // what was the next item — no increment needed (and would skip one if we did).
      if (advance && !unreviewedOnly) {
        setIndex((i) => Math.min(i + 1, visibleFrames.length - 1));
      }
    } catch (err) {
      setSaveError(err instanceof Error ? err.message : String(err));
    } finally {
      setSaving(false);
    }
  }

  function updateBox(i: number, box: BoxSpec) {
    setBoxes((prev) => prev.map((b, idx) => (idx === i ? box : b)));
  }

  function addBox(box: BoxSpec) {
    setBoxes((prev) => [...prev, box]);
    setSelectedIndex(boxes.length);
  }

  function removeBox(i: number) {
    setBoxes((prev) => prev.filter((_, idx) => idx !== i));
    setSelectedIndex(null);
  }

  return (
    <main className={styles.main}>
      <Link href={`/pipelines/${encodeURIComponent(name)}`} className={styles.back}>
        ← {name}
      </Link>
      <h1>Pre-label review</h1>

      {listError && <p className={styles.error}>{listError}</p>}
      {!listError && listLoading && <p>Loading frames…</p>}
      {!listError && !listLoading && frames.length === 0 && <p>No labeled frames found for this pipeline.</p>}

      {!listError && frames.length > 0 && (
        <>
          <div className={styles.toolbar}>
            <label>
              <input
                type="checkbox"
                checked={unreviewedOnly}
                onChange={(e) => {
                  setUnreviewedOnly(e.target.checked);
                  setIndex(0);
                }}
              />{" "}
              unreviewed only
            </label>
            <span>
              {visibleFrames.length === 0 ? "0 / 0" : `${index + 1} / ${visibleFrames.length}`}
              {" · "}
              {frames.filter((f) => f.reviewed).length}/{frames.length} reviewed
            </span>
            <button type="button" onClick={() => setIndex((i) => Math.max(0, i - 1))} disabled={index === 0}>
              ← Prev
            </button>
            <button
              type="button"
              onClick={() => setIndex((i) => Math.min(visibleFrames.length - 1, i + 1))}
              disabled={index >= visibleFrames.length - 1}
            >
              Next →
            </button>
          </div>

          {!currentFrame && <p>Nothing left to review.</p>}

          {currentFrame && (
            <div className={styles.layout}>
              <div className={styles.imageColumn}>
                {detailLoading ? (
                  <p>Loading frame…</p>
                ) : (
                  <BoxOverlay
                    imageUrl={labelImageUrl(name, currentFrame.frame_id)}
                    boxes={boxes}
                    selectedIndex={selectedIndex}
                    onSelect={setSelectedIndex}
                    onBoxChange={updateBox}
                    onAddBox={addBox}
                  />
                )}
                <p className={styles.hint}>Drag on empty space to add a box. Drag a box to move it, its corner to resize.</p>
              </div>

              <aside className={styles.sidebar}>
                <h2 className={styles.heading}>{currentFrame.frame_id}</h2>
                <ul className={styles.boxList}>
                  {boxes.map((box, i) => (
                    <li key={i} className={i === selectedIndex ? styles.selectedRow : undefined}>
                      <button type="button" onClick={() => setSelectedIndex(i)} className={styles.boxSelectButton}>
                        box {i}
                      </button>
                      <label>
                        class
                        <input
                          type="number"
                          value={box.class_id}
                          onChange={(e) => updateBox(i, { ...box, class_id: parseInt(e.target.value, 10) || 0 })}
                        />
                      </label>
                      <button type="button" onClick={() => removeBox(i)} className={styles.deleteButton}>
                        delete
                      </button>
                    </li>
                  ))}
                  {boxes.length === 0 && <li className={styles.hint}>No boxes on this frame.</li>}
                </ul>

                <div className={styles.actions}>
                  <button type="button" onClick={() => handleSave(false)} disabled={saving}>
                    {saving ? "Saving…" : "Save"}
                  </button>
                  <button type="button" onClick={() => handleSave(true)} disabled={saving}>
                    Save &amp; next
                  </button>
                </div>
                {saveError && <p className={styles.error}>{saveError}</p>}
              </aside>
            </div>
          )}
        </>
      )}
    </main>
  );
}
