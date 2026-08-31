# microduck-tracking

[Microduck](https://github.com/pollen-robotics/microduck) ships object *detection*
(a single-class YOLO11n on its NPU) but no notion of identity: if two identical
things are in view, or the target moves, there is no "that one, the same one as
before". This project adds multi-object tracking with
[`trackers`](https://github.com/roboflow/trackers) and shows what identity buys —
in the robot's own MuJoCo simulator, with its real pretrained policies.

**The demo: fetch.** The duck stands among identical balls. An off-screen owner
throws one more identical ball through the scene. Every ball carries a track ID;
the thrown one is singled out purely by its track's image-space velocity — no
ground truth — and the duck locks that ID and plays with exactly that ball,
ignoring the identical distractors. Each throw re-locks onto the new moving
track. "The ball that was just thrown" only exists as a track; no detector can
express it.

## Run it

```bash
pip install -r requirements.txt

# the simulator (used as a library — we import its PolicyInference runner)
git clone https://github.com/pollen-robotics/microduck_rl

# the pretrained behavior policies
mkdir -p policies && cd policies
for f in alpha_walking alpha_stand ball_kick_left ball_kick_right; do
  curl -sLO "https://huggingface.co/pollen-robotics/microduck-policies/resolve/main/$f.onnx"
done
cd ..

python fetch_demo.py                    # oracle detections (segmentation renderer)
DETECTOR=rfdetr python fetch_demo.py    # real RF-DETR Nano on the camera frames
```

Writes `fetch_demo.mp4`. `SIM_SECONDS` sets the length, `DEBUG_LOG=1` prints
per-frame tracking state. `MICRODUCK_RL` / `MICRODUCK_POLICIES` override the
default sibling-directory locations.

## How it works

All robot control is Pollen's own `PolicyInference` (imported from
`microduck_rl`), running their ONNX policies at 50 Hz. On top of it:

1. **Scene surgery** (`MjSpec`): the MJCF head camera points backward with a 90°
   roll from inside the jaw mesh — re-aimed forward, pitched down, widened.
   Extra identical balls added, `condim=6` so rolling friction actually applies.
2. **Behavior**: a small state machine — wait for a throw, lock the fast track,
   pursue it (gaze is a head-pose command through the policy), kick on arrival.
3. **Detection** (`detect()`): RF-DETR Nano on the rendered head-camera frames,
   or the segmentation renderer as a perfect-detector stand-in.
4. **Tracking**: `SORTTracker` at the full 50 Hz camera rate with
   `BIoU(buffer_ratio=2.0)` — the walking head-bob moves a 10 px ball box more
   than its own size between frames, so plain IoU association shatters; buffered
   IoU holds identity together.

## Would it fit the real robot?

Yes — see [hardware-feasibility.md](hardware-feasibility.md) for the measured
review. Short version: SORT/ByteTrack cost 43–291 µs per update on a laptop,
an estimated 1.5–7 ms on Microduck's Cortex-A55 — noise next to the ~60 ms the
NPU detector already spends per frame. RAM (~118 MB for a Python stack on the
1 GB board) is the number to watch; a Rust port would erase it. RF-DETR Nano
measured at 198 ms even on an RK3588's NPU+CPU split, so on-device the recipe
is: keep the YOLO-class NPU detector, add `trackers` for identity. Appearance-
based trackers (re-ID, McByte) stay off-board.
