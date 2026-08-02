import { describe, expect, it } from "vitest";
import { appendFrame, type FrameEntry } from "../frameFeed";

function entry(name: string, capturedAt = 0): FrameEntry {
  return { name, url: `http://x/${name}`, capturedAt };
}

describe("appendFrame", () => {
  it("prepends a new frame to an empty feed", () => {
    const feed = appendFrame([], entry("frame_1.jpg"), 5);

    expect(feed).toEqual([entry("frame_1.jpg")]);
  });

  it("prepends a distinct frame ahead of existing ones", () => {
    const feed = appendFrame([entry("frame_1.jpg")], entry("frame_2.jpg"), 5);

    expect(feed.map((f) => f.name)).toEqual(["frame_2.jpg", "frame_1.jpg"]);
  });

  it("is a no-op when the latest frame hasn't changed", () => {
    const initial = [entry("frame_1.jpg")];

    const feed = appendFrame(initial, entry("frame_1.jpg", 999), 5);

    expect(feed).toBe(initial);
  });

  it("caps the feed at maxLength, dropping the oldest", () => {
    const initial = [entry("frame_3.jpg"), entry("frame_2.jpg"), entry("frame_1.jpg")];

    const feed = appendFrame(initial, entry("frame_4.jpg"), 3);

    expect(feed.map((f) => f.name)).toEqual(["frame_4.jpg", "frame_3.jpg", "frame_2.jpg"]);
  });
});
