# Can `trackers` run on Microduck's hardware?

**Verdict: yes for motion-based trackers (SORT / ByteTrack / OC-SORT), with compute to spare: the binding constraint is RAM, not CPU. Appearance-based trackers (DeepSORT, BoT-SORT re-ID, McByte) do not fit on-board.**

## The hardware

| Component | Spec | Implication |
|---|---|---|
| SoC | Rockchip RK3566, 4× Cortex-A55 @ 1.8 GHz | In-order cores; one core runs the 50 Hz control loop |
| NPU | 0.8 TOPS (INT8) | Fully consumed by the detector; nothing else runs there |
| RAM | **1 GB** | Shared with robotd, mediad, tofd, the RL policy runtime |
| Storage | 32 GB eMMC | Not a constraint (~200 MB for a minimal Python env) |
| Camera | Front RGB via `mediad` (WebRTC) + 8×8 ToF | Frames accessible on-board and off-board |
| Existing vision | `duck-detect`: YOLO11n INT8 RKNN, 320×320, single class, ~60 ms/look after their preprocessing fix | ~15 Hz detection ceiling; **per-frame only, no tracking anywhere in the stack** |

## Which detector feeds the tracker?

Measured and researched for the "can we use a small RF-DETR?" question:

| Detector | Where it runs | Latency | Verdict |
|---|---|---|---|
| YOLO11n 320×320 INT8 (what Microduck ships) | RK3566 NPU | ~60 ms | **The on-robot path.** Retrain with the classes you need (Roboflow → RKNN export); trackers consumes its boxes for ~1–2 ms more on CPU |
| RF-DETR Nano (30.5 M params, 384×384) | Dev machine (M-series) | 60 ms measured | Works great off-board; it is what the `DETECTOR=rfdetr` demo mode runs |
| RF-DETR Nano, split NPU-backbone + CPU-head ([rfdetr-on-rockchip-npu](https://github.com/AlexanderDhoore/rfdetr-on-rockchip-npu)) | **RK3588** (6 TOPS, A76 cores) | **198 ms measured** by that project | ~5 fps on a chip several times stronger than Microduck's |
| RF-DETR Nano, same split | RK3566 (0.8 TOPS, A55) | est. 0.6–1 s | **Does not fit for live tracking.** DETR attention ops don't convert to RKNN end-to-end; even the split deployment is CPU-bound on the head, and the A55s are far slower than the RK3588's A76s |

So: on-device the recipe is *keep the YOLO-class NPU detector, add `trackers` for identity*; RF-DETR is the off-board / development detector. This split is exactly what the demo implements.

## End-to-end pipeline rate on the robot

Adding up the pipeline that actually fits (detector on NPU, tracker on one A55 core):

| Stage | Cost | Source |
|---|---|---|
| YOLO11n 320×320 INT8 detection | ~60 ms | Pollen's own on-robot figure (duck-detect) |
| SORT/ByteTrack update, ≤5 objects | 1.5–4 ms | our benchmark × conservative 25× A55 scaling |
| Frame grab + preprocessing glue | ~5 ms | duck-detect's post-fix pipeline |
| **Total** | **~66–70 ms** | **≈ 14 Hz tracked detection** |

The tracker adds ~5 % to the frame budget the detector already spends: on this
hardware, tracking is effectively free once you have detection.

**Verified at robot cadence**: the fetch demo run with tracker updates every
3rd frame (16.7 Hz, `TRACK_EVERY=3`, `minimum_consecutive_frames=1`, buffers
scaled) completes the full loop (lock, pursuit, beak-pick, kick) with the
lock ID held ~8 s through the approach. The one fragile case at 15 Hz is
locking a ball in free flight: a ~0.5 s toss yields only ~8 looks, and in our
test the second in-flight lock failed (it succeeds at 50 Hz). Practical
mitigations: lock on the roll-out after landing (works), throw gentler, or run
the detector's full rate only during throw windows.

## Measured cost of `trackers` (v2.6, this machine: Apple M-series)

Primary numbers replay **real cached detections** from the demo (RF-DETR Nano on
the duck's head camera, 1800 frames, 3228 detections, mean 1.79 objects/frame,
max 6; `bench_tracker.py`, 5 repeats):

| Tracker | mean per update | p95 |
|---|---|---|
| SORT | 68 µs | 158 µs |
| SORT + BIoU(2.0) | 65 µs | 117 µs |
| ByteTrack | 53 µs | 126 µs |

Scaling with object count, on synthetic 640×360 scenes (2000 frames per cell):

| Tracker | 1 obj | 2 obj | 8 obj | 16 obj |
|---|---|---|---|---|
| SORT | 43 µs | 58 µs | 157 µs | 257 µs |
| SORT + BIoU(0.8) | 50 µs | 66 µs | 150 µs | 269 µs |
| ByteTrack | 47 µs | 68 µs | 165 µs | 291 µs |

Memory, same process (macOS `ru_maxrss`):

| Stage | RSS |
|---|---|
| Bare Python 3.11 | 24 MB |
| After `import trackers` (pulls numpy, scipy, supervision→cv2) | **112 MB** |
| After sustained 16-object tracking | 118 MB |

Disk: full demo env 532 MB; a robot-minimal env (numpy, scipy, opencv-headless, supervision, trackers) estimates at ~200 MB, irrelevant against 32 GB.

## Scaling to the Cortex-A55

No RK3566 was available to measure on, so the CPU numbers must be scaled. A conservative single-core factor for numpy-light workloads (in-order A55 @ 1.8 GHz vs an M-series performance core) is **15–25×**. Taking the worst case, 25×:

- 16 objects, ByteTrack: 291 µs × 25 ≈ **7.3 ms/update**
- 2 objects (the demo scenario), SORT: 58 µs × 25 ≈ **1.5 ms/update**

Against the budget: the detector delivers a frame every ~60 ms (15 Hz). Even the worst scaled case consumes ~12 % of one core at that rate, and the robot has four cores, with the control loop needing one. **Compute is not the problem, even before any optimization.** This scaling factor is the one unmeasured link in the chain; a 30-minute benchmark on any RK3566 dev board (~$40) would close it.

## RAM is the real constraint

~118 MB of Python runtime on a 1 GB board that already runs the Rust daemon stack (robotd, mediad, configd, tofd, updaterd), the ONNX policy runtime, and buffers WebRTC video. That likely fits today, since the Rust daemons are lean, but it is the number to watch, and it is pure interpreter+library overhead: the tracker state itself is kilobytes.

## What does NOT fit on-board

- **DeepSORT / BoT-SORT with re-ID**: appearance embedding needs a CNN forward pass per detection. The 0.8 TOPS NPU is already saturated by detection; running embeddings on the A55s costs tens of ms each. Off-board only.
- **McByte**: needs per-frame masks (SAM-class model). Not on this silicon.
- **Torch anything**: no torch on the robot; the deployment path is ONNX/RKNN only.

## Three integration paths

| Path | Effort | Latency | Verdict |
|---|---|---|---|
| **Off-board**: subscribe to mediad's WebRTC stream, run detector + trackers on a laptop, command via the JSON-RPC socket API | Works today (the sim demo is exactly this pattern) | +20–50 ms network RTT | Right for demos, tutorials, promo content |
| **On-board Python sidecar**: a `trackd` daemon consuming duck-detect's boxes, speaking the same Unix-socket JSON-RPC as every other client | Small; aarch64 manylinux wheels exist for every dependency | none | Right for a community integration / blog post |
| **Rust port**: SORT/ByteTrack is a Kalman filter + Hungarian assignment; a `duck-track` crate beside duck-detect, using our clean-room implementations as the reference | Days, not hours | none, ~zero RAM | Right for upstreaming to pollen-robotics |

## Bottom line

Microduck ships detection with no identity. Our motion trackers close that gap within ~1.5–7 ms and ~kilobytes of state on its CPU. The measured demo (target lock, kick-and-chase, ID persistence through occlusion) is behavior that detection alone cannot express. The honest caveats: the A55 scaling factor is estimated, not measured; Python's ~118 MB RSS is the deployment risk on 1 GB; and anything needing appearance embeddings stays off-board.
