/** Pure logic for turning repeated `/pipelines/{name}/preview` polls into a bounded feed. */

export interface FrameEntry {
  /** The `X-Frame-Name` response header; identifies a frame across polls. */
  name: string;
  url: string;
  capturedAt: number;
}

/**
 * Prepend `entry` to `feed`, deduplicating unchanged polls and capping length.
 *
 * The preview endpoint always returns the *latest* frame, so back-to-back polls
 * commonly see the same file before the pipeline has produced a new one; without
 * dedup the feed would fill with identical repeats instead of distinct frames.
 */
export function appendFrame(feed: FrameEntry[], entry: FrameEntry, maxLength: number): FrameEntry[] {
  if (feed.length > 0 && feed[0].name === entry.name) {
    return feed;
  }
  return [entry, ...feed].slice(0, maxLength);
}
