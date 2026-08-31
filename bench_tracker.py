#!/usr/bin/env python3
"""Measure trackers' compute and memory cost for the Microduck hardware review.

Times SORTTracker/ByteTrack update() per frame at several object counts on
synthetic-but-realistic moving boxes (640x360 frame), and reports process RSS
after imports and after sustained tracking.
"""

import resource
import time

import numpy as np


def rss_mb():
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return ru / (1024 * 1024)  # macOS reports bytes


rss_start = rss_mb()
import supervision as sv  # noqa: E402

from trackers import SORTTracker  # noqa: E402
from trackers.utils.iou import BIoU  # noqa: E402

try:
    from trackers import ByteTrackTracker as ByteTrack
except ImportError:
    ByteTrack = None
rss_imports = rss_mb()

W, H = 640, 360
rng = np.random.default_rng(0)


def synth_frames(n_obj, n_frames=2000):
    """Boxes drifting with per-frame jitter, like the duck's head-cam view."""
    pos = rng.uniform([0, 0], [W - 40, H - 40], size=(n_obj, 2))
    vel = rng.uniform(-3, 3, size=(n_obj, 2))
    sizes = rng.uniform(10, 60, size=(n_obj, 1))
    out = []
    for _ in range(n_frames):
        pos += vel + rng.normal(0, 2, size=pos.shape)
        pos = np.clip(pos, 0, [W - 40, H - 40])
        xyxy = np.hstack([pos, pos + sizes])
        out.append(
            sv.Detections(
                xyxy=xyxy.copy(),
                confidence=rng.uniform(0.5, 1.0, n_obj),
                class_id=np.zeros(n_obj, dtype=int),
            )
        )
    return out


def bench(make_tracker, n_obj, n_frames=2000):
    frames = synth_frames(n_obj, n_frames)
    tracker = make_tracker()
    for d in frames[:50]:  # warmup
        tracker.update(d)
    tracker = make_tracker()
    t0 = time.perf_counter()
    for d in frames:
        tracker.update(d)
    dt = time.perf_counter() - t0
    return dt / n_frames * 1e6  # us per update


print(f"RSS at start: {rss_start:.1f} MB")
print(f"RSS after imports (numpy+scipy+supervision+trackers): {rss_imports:.1f} MB")
print()
print(f"{'tracker':<28} {'objs':>4} {'us/update':>10} {'fps-equiv':>10}")
configs = [("SORT", lambda: SORTTracker()),
           ("SORT+BIoU(0.8)", lambda: SORTTracker(iou=BIoU(buffer_ratio=0.8)))]
if ByteTrack is not None:
    configs.append(("ByteTrack", lambda: ByteTrack()))
for name, mk in configs:
    for n in (1, 2, 8, 16):
        us = bench(mk, n)
        print(f"{name:<28} {n:>4} {us:>10.0f} {1e6 / us:>10.0f}")
print()
print(f"RSS after sustained tracking: {rss_mb():.1f} MB")
