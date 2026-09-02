#!/usr/bin/env python3
"""Microduck counts the balls in the park, powered by `trackers`.

Six balls sit on an arc wider than the head camera's lens. The duck stands
still and sweeps its head across them. No single frame ever holds more than
four, so the most an honest per-frame detector can report is four. Tracking
keeps an id on each ball as it enters and leaves the view, so the duck can
say six.

Counting distinct things is the plainest job detection cannot do alone.

    python demos/counting.py
"""

import math
import os

import cv2
import imageio.v2 as imageio
import supervision as sv
from trackers import SORTTracker
from trackers.utils.iou import BIoU

from detection import BallDetector
from scene import ROOT, Microduck, View

RADIUS = 1.05
BEARINGS = [-62, -38, -14, 12, 36, 60]   # degrees, an arc wider than the lens
BALLS = [
    (RADIUS * math.cos(math.radians(b)), RADIUS * math.sin(math.radians(b)))
    for b in BEARINGS
]
STEPS = 600
PAN = 0.9                  # head yaw command at each end of the sweep
CROP = 380                 # video only: trim empty lawn below the balls
SETTLE = 90                # steps to reach the start of the sweep, uncounted
MAGENTA = sv.Color(255, 64, 255)


def sweep(step):
    """Head yaw: one pass, right to left, never doubling back.

    SORT associates on motion alone, so a ball that leaves the view and
    returns comes back as a new id and is counted twice. Counting a scene
    this way is a single pass, or it needs the appearance re-identification
    that does not fit on this robot.
    """
    return -PAN + 2 * PAN * min(step / (STEPS - 1), 1.0)


def draw_counters(frame, in_view, unique):
    """The whole argument, in two numbers."""
    height = frame.shape[0]
    top = height - 92
    panel = frame.copy()
    cv2.rectangle(panel, (0, top), (frame.shape[1], height), (18, 18, 18), -1)
    frame = cv2.addWeighted(panel, 0.82, frame, 0.18, 0)
    cv2.putText(frame, "IN VIEW NOW", (28, top + 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.62, (170, 170, 170), 2, cv2.LINE_AA)
    cv2.putText(frame, str(in_view), (28, top + 74), cv2.FONT_HERSHEY_SIMPLEX,
                1.25, (235, 235, 235), 3, cv2.LINE_AA)
    cv2.putText(frame, "UNIQUE BALLS SEEN", (250, top + 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.62, (255, 140, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, str(unique), (250, top + 74), cv2.FONT_HERSHEY_SIMPLEX,
                1.25, (255, 64, 255), 3, cv2.LINE_AA)
    return frame


def main():
    duck = Microduck(BALLS)
    detector = BallDetector(duck.ball_geom_ids)
    # The duck's-eye view is the whole argument here, so it is the frame
    # rather than an inset: View renders, this file writes.
    view = View(duck, None, headless=True)
    path = os.path.join(ROOT, "counting.mp4")
    writer = imageio.get_writer(path, fps=50, quality=8)
    tracker = SORTTracker(
        lost_track_buffer=150,
        frame_rate=50.0,
        minimum_consecutive_frames=2,
        minimum_iou_threshold=0.05,
        iou=BIoU(buffer_ratio=2.0),
    )
    box = sv.BoxCornerAnnotator(thickness=4, corner_length=14, color=MAGENTA)
    label = sv.LabelAnnotator(text_scale=0.7, text_thickness=2, color=MAGENTA)

    # Bring the head to the start of the sweep before counting anything.
    # Counting from a centred head sees the balls on the far side, loses them
    # as the head swings out, and then counts them again on the way back.
    for _ in range(SETTLE):
        duck.policy.set_vel_cmd(0.0, 0.0, 0.0)
        duck.policy.head_offset[:] = [0.0, 0.0, -PAN, 0.0]
        duck.step()

    seen = {}
    for step in range(STEPS):
        duck.policy.set_vel_cmd(0.0, 0.0, 0.0)
        duck.policy.head_offset[:] = [0.0, 0.0, sweep(step), 0.0]
        duck.step()

        view.start_frame()
        head = view.head_frame()
        detections = detector(head, detector.oracle_boxes(view.head_segmentation()))
        tracked = tracker.update(detections)
        tracked = tracked[tracked.tracker_id != -1]

        for tracker_id in tracked.tracker_id:
            seen.setdefault(int(tracker_id), len(seen) + 1)
        if len(tracked):
            head = box.annotate(head, tracked)
            head = label.annotate(
                head, tracked,
                labels=[f"#{seen[int(i)]}" for i in tracked.tracker_id],
            )
        writer.append_data(draw_counters(head[:CROP], len(detections), len(seen)))

    writer.close()
    print(f"wrote {path}: {len(BALLS)} balls in the park, counted {len(seen)}")


if __name__ == "__main__":
    main()
