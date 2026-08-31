#!/usr/bin/env python3
"""Benchmark trackers over real cached detections from the Microduck demo.

Replays detections_cache.npz (RF-DETR Nano on the duck's head camera,
captured with DUMP_DETECTIONS=detections_cache.npz python fetch_demo.py)
through each tracker and reports per-update latency and process memory.
"""

import os
import resource
import statistics
import time

import numpy as np
import supervision as sv

from trackers import ByteTrackTracker, SORTTracker
from trackers.utils.iou import BIoU

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.environ.get(
    "DETECTIONS_CACHE", os.path.join(HERE, "detections_cache.npz")
)


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def load_frames():
    if not os.path.exists(CACHE):
        raise FileNotFoundError(
            f"{CACHE} not found. Capture it with "
            "DUMP_DETECTIONS=benchmark/detections_cache.npz DETECTOR=rfdetr python demos/fetch_demo.py"
        )
    data = np.load(CACHE)
    rows, n_frames = data["rows"], int(data["n_frames"])
    frames = []
    for i in range(n_frames):
        sel = rows[rows[:, 0] == i]
        if len(sel):
            frames.append(
                sv.Detections(
                    xyxy=sel[:, 1:5].copy(),
                    confidence=sel[:, 5].copy(),
                    class_id=np.zeros(len(sel), dtype=int),
                )
            )
        else:
            frames.append(sv.Detections.empty())
    return frames


def bench(make_tracker, frames, repeats=5):
    tracker = make_tracker()
    for d in frames[:100]:
        tracker.update(d)
    per_update = []
    for _ in range(repeats):
        tracker = make_tracker()
        for d in frames:
            t0 = time.perf_counter()
            tracker.update(d)
            per_update.append(time.perf_counter() - t0)
    us = [x * 1e6 for x in per_update]
    return statistics.mean(us), np.percentile(us, 95)


frames = load_frames()
counts = [len(d) for d in frames]
print(
    f"cache: {len(frames)} frames, {sum(counts)} detections, "
    f"mean {statistics.mean(counts):.2f} objects/frame, max {max(counts)}"
)
print(f"RSS after imports: {rss_mb():.1f} MB\n")
print(f"{'tracker':<24} {'mean us/update':>14} {'p95 us':>8}")
configs = [
    ("SORT", lambda: SORTTracker(frame_rate=50.0)),
    (
        "SORT+BIoU(2.0)",
        lambda: SORTTracker(frame_rate=50.0, iou=BIoU(buffer_ratio=2.0)),
    ),
    ("ByteTrack", lambda: ByteTrackTracker(frame_rate=50.0)),
]
for name, mk in configs:
    mean_us, p95_us = bench(mk, frames)
    print(f"{name:<24} {mean_us:>14.0f} {p95_us:>8.0f}")
print(f"\nRSS after sustained tracking: {rss_mb():.1f} MB")
