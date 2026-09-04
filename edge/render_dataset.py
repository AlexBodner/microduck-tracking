#!/usr/bin/env python3
"""Render a labelled dataset from the duck's own camera.

The simulator knows exactly where every ball is, so the labels come out of
MuJoCo's segmentation buffer rather than out of a human. Frames are the duck's
real head camera: same lens, same 25 degree downward mount, same park.

Writes YOLO format, ready to upload to Roboflow.

    python edge/render_dataset.py --out ball_dataset --frames 300
"""

import argparse
import math
import os
import random

import cv2

from detection import BallDetector
from scene import Microduck, View

CLASS_NAME = "ball"


def scatter(rng, count):
    """Balls in front of the duck, at the ranges it actually sees them.

    Weighted toward close: a uniform spread over distance puts almost every
    ball near the horizon, and the frames that matter most are the ones from
    the end of an approach, where the ball is large and low in the view.
    """
    placements = []
    for _ in range(count):
        bearing = rng.uniform(-1.0, 1.0)
        distance = 0.18 + (2.6 - 0.18) * rng.random() ** 2.2
        placements.append(
            (distance * math.cos(bearing), distance * math.sin(bearing))
        )
    return placements


def gaze(rng, placements):
    """Look the way the fetch policy looks: the closer the ball it is working
    on, the further down the head is pitched. Copying that here keeps the
    training frames in the same distribution as the deployed ones."""
    distance = min(math.hypot(x, y) for x, y in placements)
    pitch = min(max(1.2 * (0.55 - distance), 0.0), 0.55)
    return [
        0.0,
        pitch + rng.uniform(-0.15, 0.25),
        rng.uniform(-0.9, 0.9),
        0.0,
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="ball_dataset")
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    images_dir = os.path.join(args.out, "images")
    labels_dir = os.path.join(args.out, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    written, empty = 0, 0
    scene_index = 0
    placements = scatter(rng, 6)
    duck = Microduck(placements)
    detector = BallDetector(duck.ball_geom_ids)
    view = View(duck, None, headless=True)

    while written < args.frames:
        # A fresh scatter every so often, so the set is not one park with one
        # arrangement seen from slightly different angles.
        if written % 25 == 0 and written:
            scene_index += 1
            placements = scatter(rng, rng.randint(3, 6))
            duck = Microduck(placements)
            detector = BallDetector(duck.ball_geom_ids)
            view = View(duck, None, headless=True)

        duck.policy.set_vel_cmd(0.0, 0.0, 0.0)
        duck.policy.head_offset[:] = gaze(rng, placements)
        for _ in range(rng.randint(2, 6)):
            duck.step()

        view.start_frame()
        frame = view.head_frame()
        boxes = detector.oracle_boxes(view.head_segmentation())
        if not boxes:
            empty += 1
            if empty % 5:
                continue

        height, width = frame.shape[:2]
        name = f"duck_{written:05d}"
        cv2.imwrite(os.path.join(images_dir, f"{name}.jpg"),
                    cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        with open(os.path.join(labels_dir, f"{name}.txt"), "w") as handle:
            for x1, y1, x2, y2 in boxes:
                cx = (x1 + x2) / 2 / width
                cy = (y1 + y2) / 2 / height
                bw = (x2 - x1) / width
                bh = (y2 - y1) / height
                handle.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
        written += 1

    with open(os.path.join(args.out, "data.yaml"), "w") as handle:
        handle.write(f"names:\n  0: {CLASS_NAME}\nnc: 1\n")
    print(f"wrote {written} images to {args.out} ({empty} empty frames skipped)")


if __name__ == "__main__":
    main()
