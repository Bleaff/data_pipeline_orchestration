from __future__ import annotations

import math
import os

import cv2

from neudc.utils import LOGGER

from .base import BaseLoader

__all__ = ("VideoLoader",)


class VideoLoader(BaseLoader):

    valid_extensions = (".mp4", ".ts", ".avi")

    """
    Iterates through video and allows them to be iterated at specified intervals (a list of pairs in seconds).
    When iterating, it returns (index_frame, video_name, frame).
    """

    def __init__(
        self,
        path: str,
        skip_time_ms: int = 0,
        intervals: list[tuple[float, float]] | None = None,
    ) -> VideoLoader:

        if not os.path.isfile(path):
            LOGGER.error(f"Video not found: {path}")

        if not path.lower().endswith(VideoLoader.valid_extensions):
            LOGGER.error(f"WARNING ⚠️ Extension for the file {path} is not supported.")

        self.video_name = os.path.basename(path)
        self.video_path = path
        self.skip_time_ms = skip_time_ms
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            LOGGER.error(f"Cannot open video: {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not intervals:
            intervals = [(0, self.total_frames / self.fps)]
        self.frame_intervals = []
        for start_sec, end_sec in intervals:
            start_frame = int(start_sec * self.fps)
            end_frame = min(int(end_sec * self.fps), self.total_frames - 1)
            if start_frame <= end_frame:
                self.frame_intervals.append((start_frame, end_frame))
        LOGGER.info(
            f"Added VideoDataLoader for file: path={path}, intervals(frames)={self.frame_intervals}, skip_time_ms={self.skip_time_ms}",
        )

    def register_intervals(self) -> None: ...

    def __iter__(self):
        intervals = self.frame_intervals if self.frame_intervals else [(0, float("inf"))]
        interval_idx = 0
        current_interval = intervals[interval_idx]
        target_timestamp = current_interval[0] / self.fps
        frame_idx = 0
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                LOGGER.warning(f"WARNING ⚠️ Cannot read frame of {self.video_name} at index {frame_idx}")
                break
            frame_idx += 1
            if frame_idx < current_interval[0]:
                continue
            while frame_idx > current_interval[1]:
                interval_idx += 1
                if interval_idx >= len(intervals):
                    return
                current_interval = intervals[interval_idx]
                target_timestamp = current_interval[0] / self.fps
                if frame_idx < current_interval[0]:
                    break
            if frame_idx < current_interval[0]:
                continue
            current_timestamp = frame_idx / self.fps
            if current_timestamp >= target_timestamp:
                yield (frame_idx, self.video_name, current_timestamp), frame
                target_timestamp = current_timestamp + (self.skip_time_ms / 1000.0)

    def __len__(self) -> int:
        total = 0
        effective_step = max(1, math.ceil((self.skip_time_ms / 1000.0) * self.fps))
        for start_frame, end_frame in self.frame_intervals:
            interval_frames = end_frame - start_frame + 1
            total += (interval_frames + effective_step - 1) // effective_step
        return total

    def __del__(self) -> None:
        if self.cap.isOpened():
            self.cap.release()

        del self.cap
