#!/usr/bin/env python3
"""Score a trained model on freshly rendered scenes, against oracle boxes.

The split inside a Roboflow version is not a fair test for this dataset:
consecutive rendered frames share a scene, so held-out frames are near
duplicates of training ones and the metrics come back at 100%. Rendering new
scenes with an unused seed gives the model something it has genuinely not seen.

    python edge/eval_in_sim.py <model package dir> --seed 4242 --frames 120
"""

import argparse
import random

import numpy as np
from render_dataset import gaze, scatter
from yololite_detector import YOLOLiteDetector

from detection import BallDetector
from scene import Microduck, View

MATCH_IOU = 0.5


def iou(a, b):
    wide = min(a[2], b[2]) - max(a[0], b[0])
    high = min(a[3], b[3]) - max(a[1], b[1])
    if wide <= 0 or high <= 0:
        return 0.0
    overlap = wide * high
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - overlap
    return overlap / union


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("package", help="a downloaded Roboflow model package")
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--confidence", type=float, default=0.5)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    model = YOLOLiteDetector(args.package, confidence=args.confidence)

    true_positives = false_positives = false_negatives = 0
    matched_ious, missed_widths, found_widths = [], [], []

    placements = scatter(rng, 5)
    duck = Microduck(placements)
    oracle = BallDetector(duck.ball_geom_ids)
    view = View(duck, None, headless=True)

    for step in range(args.frames):
        if step % 20 == 0 and step:
            placements = scatter(rng, rng.randint(3, 6))
            duck = Microduck(placements)
            oracle = BallDetector(duck.ball_geom_ids)
            view = View(duck, None, headless=True)

        duck.policy.set_vel_cmd(0.0, 0.0, 0.0)
        duck.policy.head_offset[:] = gaze(rng, placements)
        for _ in range(rng.randint(2, 6)):
            duck.step()

        view.start_frame()
        frame = view.head_frame()
        truth = oracle.oracle_boxes(view.head_segmentation())
        predicted = list(model(frame).xyxy)

        used = set()
        for box in truth:
            best, best_iou = None, MATCH_IOU
            for index, candidate in enumerate(predicted):
                if index in used:
                    continue
                overlap = iou(box, candidate)
                if overlap >= best_iou:
                    best, best_iou = index, overlap
            if best is None:
                false_negatives += 1
                missed_widths.append(box[2] - box[0])
            else:
                used.add(best)
                true_positives += 1
                matched_ious.append(best_iou)
                found_widths.append(box[2] - box[0])
        false_positives += len(predicted) - len(used)

    denominator = true_positives + false_positives
    precision = true_positives / denominator if denominator else 0.0
    denominator = true_positives + false_negatives
    recall = true_positives / denominator if denominator else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    print(f"frames={args.frames} seed={args.seed} (scenes never seen in training)")
    print(f"TP={true_positives} FP={false_positives} FN={false_negatives}")
    print(f"precision={precision:.3f} recall={recall:.3f} F1={f1:.3f} "
          f"mean IoU of matches={np.mean(matched_ious):.3f}")
    if missed_widths:
        print(f"missed box widths px: median {np.median(missed_widths):.0f}, "
              f"max {max(missed_widths):.0f}")
        print(f"found box widths px: median {np.median(found_widths):.0f}, "
              f"min {min(found_widths):.0f}")


if __name__ == "__main__":
    main()
