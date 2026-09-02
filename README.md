# microduck-tracking

Multi-object tracking powered by [`trackers`](https://github.com/roboflow/trackers)
for the [Microduck](https://github.com/pollen-robotics/microduck).

![Microduck fetch demo](assets/fetch_demo.gif)

## Why tracking

Microduck ships per-frame detection with no identity: two identical balls are just
two boxes, every frame. In this demo an owner's hand throws one more identical ball
into the scene, and the thrown one is singled out purely by its track's velocity.
The duck locks that ID, fetches exactly that ball past the identical distractors,
and re-locks after every throw. "The ball that was just thrown" only exists as a
track.

## Quickstart

```bash
pip install trackers

pip install -r requirements.txt
git clone https://github.com/pollen-robotics/microduck_rl  # verified at d424a0c

mkdir -p policies && cd policies
for f in alpha_walking alpha_stand ball_kick_left ball_kick_right alpha_ground_pick; do
  curl -sLO "https://huggingface.co/pollen-robotics/microduck-policies/resolve/main/$f.onnx"
done
cd ..

python minimal_tracking.py
python demos/fetch_demo.py

pip install rfdetr
DETECTOR=rfdetr python demos/fetch_demo.py
```

Tracking is four lines:

```python
from trackers import SORTTracker
from trackers.utils.iou import BIoU

tracker = SORTTracker(frame_rate=50.0, iou=BIoU(buffer_ratio=2.0))
tracked = tracker.update(detections)
```

`detections` is an `sv.Detections` from any detector, and `tracked.tracker_id`
carries a stable id per object. Other trackers and options are in the
[trackers docs](https://trackers.roboflow.com).

[`minimal_tracking.py`](minimal_tracking.py) wires that seam into the MuJoCo
simulator: head-camera frames in, ids out.

![Minimal tracking demo](assets/minimal_tracking.gif)

[`demos/fetch_demo.py`](demos/fetch_demo.py) is the full fetch choreography.
Its `main()` holds the loop: detect, track, decide, step. Around it,
[`detection.py`](demos/detection.py) finds the balls,
[`policy.py`](demos/policy.py) turns tracks into motor commands, and
[`scene/`](demos/scene) is the world: the robot and its park, the owner's
hand, and the rendering.
[`demos/duck_tracking.py`](demos/duck_tracking.py) swaps in `duck_detect.onnx`,
the single-class YOLO the robot ships on its NPU.

![Duck detector tracking](assets/duck_tracking.gif)

[`demos/counting.py`](demos/counting.py) counts: six balls on an arc wider than
the lens, never more than four in frame at once, one head sweep, six ids. The
sweep only goes one way, because SORT associates on motion alone, so a ball
that leaves the view and comes back returns as a new id.

![Counting demo](assets/counting.gif)

## How it works

**Policies.** Pollen's own `PolicyInference` runner and pretrained ONNX policies,
unmodified, at 50 Hz.

**Detection.** RF-DETR Nano on the head-camera frames (`DETECTOR=rfdetr`), or the
segmentation renderer as a perfect-detector stand-in.

**Tracking.** `SORTTracker` at the full 50 Hz camera rate with
`BIoU(buffer_ratio=2.0)`: the walking head-bob moves a small ball box more than its
own width between frames, and buffered IoU is what holds identity through it.

**Target selection.** Each track keeps an EMA of its box-center speed in image
space. After a throw releases, the duck locks the fastest track. It stands still
through the throw, so image speed alone identifies the thrown ball.

**Navigation.** From the camera image and the tracked ball's position in it, the
duck works out which way to turn and how far away the ball is, and walks there:
bearing from the box centre, range from the box's apparent diameter against the
ball's known 70 mm. The track says which detection is ours and the detection
says where it is, so nothing is measured from a track coasting on its Kalman
prediction. Simulator state reaches the owner's hand, never the duck.

Both detectors stop resolving the ball at about 0.19 m, so after its last
sighting the duck walks one fixed 0.55 s step before ducking. That length comes
from counting frames of beak-to-ball contact against a ball placed at known
offsets: at 0.14 m the beak only grazes it, at 0.08 to 0.12 m it lands on it
properly, and at 0.06 m the ankles reach the ball first and kick it away. The
step aims for the middle of that band.

Across six randomised seeds, eleven throws, the duck pecked and moved the ball
on seven, contact confirmed from the simulator's own contact list rather than by
eye. Those seven land 0.110 to 0.137 m from it. The four failures share a cause:
the last range fix came off a lookalike instead of the target, so the duck
commits its blind step on a bad number and pecks 0.13 to 0.75 m short. Spacing
the lookalikes further apart hides that, at the cost of the thing the demo is
for. One thing the POV
inset does not show is the moment of contact: Pollen mount the lens low and
behind the beak, so the peck swings the camera through the ball and the view
goes briefly empty. Moving the camera forward fixes it and stops it being this
robot's camera, so it stays where it is. Reproduce with
`THROW_SEED=<n> TRIAL=1 HEADLESS=1 python demos/fetch_demo.py`. That is with the
segmentation stand-in; RF-DETR is weaker here, padding boxes and often missing
the ball in flight, so the duck sometimes never re-locks after a throw. The lock
has no fallback for that on purpose: every fallback we tried locked onto
detector flicker, and fetching the wrong ball is worse than fetching none.

## Does it fit the real robot?

Yes. SORT costs 73 to 76 µs per update on an Apple M4 over the demo's real cached
detections, an estimated 2 to 7 ms on Microduck's Cortex-A55, next to the ≈60 ms
its NPU detector already spends per frame: ≈14 Hz tracked detection end to end.
Measurements in [hardware-feasibility.md](benchmark/hardware-feasibility.md).

## Acknowledgements

Simulator, policies, and the Microduck itself are by
[Pollen Robotics](https://github.com/pollen-robotics) and Hugging Face.
