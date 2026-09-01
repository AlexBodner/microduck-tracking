#!/usr/bin/env python3
"""Microduck plays fetch, powered by `trackers`.

The duck stands among identical resting balls. An owner throws one more
identical ball into the scene. SORTTracker keeps an id on every ball, and the
thrown one is singled out purely by track motion: the duck locks that id and
goes to play with exactly that ball, walking past the lookalikes. "The ball
that was just thrown" only exists as a track, so detection alone cannot
express any of this.

The controller is closed on vision. Bearing and range come from the tracked
ball's box; the simulator's ball positions reach the owner's hand, never the
duck.

    python demos/fetch_demo.py                  # segmentation stand-in
    DETECTOR=rfdetr python demos/fetch_demo.py  # RF-DETR on the camera frames
    THROW_SEED=2 TRIAL=1 HEADLESS=1 ...         # one scored trial, no video
"""

import math
import os

import numpy as np
from trackers import SORTTracker
from trackers.utils.iou import BIoU

from detection import BallDetector
from policy import FetchPolicy, focal_px
from scene import CWD, PANEL_W, ROOT, Microduck, Owner, View

SIM_SECONDS = float(os.environ.get("SIM_SECONDS", 28))
THROW_SEED = int(os.environ.get("THROW_SEED", 3))    # where the throws land
TRACK_EVERY = int(os.environ.get("TRACK_EVERY", 1))  # 3 is the robot's ~16.7 Hz
HEADLESS = bool(os.environ.get("HEADLESS"))          # skip video, for batches
TRIAL = bool(os.environ.get("TRIAL"))                # score each fetch
FIRST_THROW = 1.2
MAX_THROWS = 2
RETHROW_DELAY = 2.0    # the owner waits a beat after the duck's touch
THROW_TIMEOUT = 18.0   # throw again anyway if a fetch stalls
# Close enough that the duck is visibly walking past lookalikes, far enough
# off the throw's line that it never knocks one on the way in: the ball rests
# within about 0.15 m of the centre line and two balls collide inside 0.07 m,
# which puts the nearest safe lane at 0.22 m. These sit at 0.30 m for margin.
DISTRACTORS = [(0.58, -0.30), (0.88, 0.31), (1.28, -0.29), (1.62, 0.32)]


def build_tracker():
    """SORT, tuned for small balls seen from a walking robot.

    Buffered IoU is the one non-default choice: the head bobs with every step,
    which moves a distant ball's box further than its own width between
    frames, and plain IoU association breaks there.
    """
    return SORTTracker(
        lost_track_buffer=150 // TRACK_EVERY,
        frame_rate=50.0 / TRACK_EVERY,
        minimum_consecutive_frames=2 if TRACK_EVERY == 1 else 1,
        minimum_iou_threshold=0.05,
        iou=BIoU(buffer_ratio=2.0),
    )


def trial_scoring(duck, policy):
    """Report how far the thrown ball was, and whether the beak moved it."""
    state = {"ball": None}

    def on_grab(t):
        trunk_xy, yaw = duck.trunk_frame()
        offset = duck.ball_position()[:2] - trunk_xy
        c, s = math.cos(-yaw), math.sin(-yaw)
        fwd, left = c * offset[0] - s * offset[1], s * offset[0] + c * offset[1]
        state["ball"] = duck.ball_position()
        print(
            f"TRIAL_GRAB t={t:5.1f} play_ball_distance={math.hypot(fwd, left):.3f} "
            f"fwd={fwd:+.3f} left={left:+.3f}"
        )

    def on_touch(t):
        if state["ball"] is None:
            return
        moved = float(np.linalg.norm(duck.ball_position()[:2] - state["ball"][:2]))
        print(
            f"TRIAL_PICK t={t:5.1f} ball_moved={moved:.3f} -> "
            f"{'TOUCHED' if moved > 0.005 else 'MISSED'}"
        )

    policy.on_grab, policy.on_touch = on_grab, on_touch


def main():
    duck = Microduck(DISTRACTORS)
    owner = Owner(duck, seed=THROW_SEED)
    detector = BallDetector(duck.ball_geom_ids)
    tracker = build_tracker()
    view = View(duck, os.path.join(ROOT, "fetch_demo.mp4"), headless=HEADLESS)
    policy = FetchPolicy(duck, PANEL_W, focal_px(PANEL_W))
    if TRIAL:
        trial_scoring(duck, policy)

    throws, next_throw = 0, FIRST_THROW
    tracked, detections, dumped = None, None, []
    steps = int(SIM_SECONDS / duck.control_dt)

    for step in range(steps):
        t = step * duck.control_dt
        duck.policy.update_behavior(duck.control_dt)

        # The owner throws when the duck is free, then again a beat after each
        # touch, retrieving the ball in between.
        free = duck.policy.behavior_mode is None and not duck.policy.ground_pick_mode
        if throws < MAX_THROWS and t >= next_throw and not owner.busy and free:
            policy.on_throw(owner.start_throw(t))
            throws += 1
            next_throw = t + THROW_TIMEOUT
        owner.update(t)

        duck.policy.update_ground_pick_phase(duck.control_dt)
        if policy.note_pick(t):
            next_throw = t + RETHROW_DELAY
        policy.act(t)
        duck.step()

        # Look: render the duck's-eye view, detect the balls in it, and give
        # the boxes to the tracker so each ball keeps its own id.
        view.start_frame(beside=policy.pecking or t < policy.play_until)
        head = view.head_frame()
        oracle_boxes = detector.oracle_boxes(view.head_segmentation())
        if step % TRACK_EVERY == 0:
            detections = detector(head, oracle_boxes)
            if os.environ.get("DUMP_DETECTIONS"):
                for box, confidence in zip(detections.xyxy, detections.confidence):
                    dumped.append((step, *box, confidence))
            tracked = tracker.update(detections)
            tracked = tracked[tracked.tracker_id != -1]  # confirmed tracks only

        if len(tracked):
            policy.observe(tracked, detections, t)
            head = view.annotate(head, tracked, policy.target_id, policy.births, t)
        view.write(head)

    view.close()
    if os.environ.get("DUMP_DETECTIONS"):
        cache = os.environ["DUMP_DETECTIONS"]
        if not os.path.isabs(cache):
            cache = os.path.join(CWD, cache)
        np.savez_compressed(
            cache, rows=np.array(dumped, dtype=np.float64), n_frames=steps
        )
        print(f"wrote {cache}: {len(dumped)} detections over {steps} frames")


if __name__ == "__main__":
    main()
