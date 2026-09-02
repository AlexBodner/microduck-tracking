# Can `trackers` run on Microduck's hardware?

**Verdict: yes for motion-based trackers (SORT / ByteTrack / OC-SORT), with compute to spare. Appearance-based trackers (DeepSORT, BoT-SORT re-ID, McByte) do not fit on-board.**

## The hardware

| Component | Spec | Implication |
|---|---|---|
| SoC | Rockchip RK3566, 4× Cortex-A55 @ 1.8 GHz | In-order cores; one core runs the 50 Hz control loop |
| NPU | 0.8 TOPS (INT8) | Fully consumed by the detector; nothing else runs there |
| RAM | **1 GB** | Shared with robotd, mediad, tofd, the RL policy runtime |
| Storage | 32 GB eMMC | Not a constraint (≈200 MB for a minimal Python env) |
| Camera | Front RGB via `mediad` (WebRTC) + 8×8 ToF | Frames accessible on-board and off-board |
| Existing vision | `duck-detect`: YOLO11n INT8 RKNN, 320×320, single class, ≈60 ms/look after their preprocessing fix | ≈15 Hz detection ceiling; **per-frame only, no tracking anywhere in the stack** |

## Which detector feeds the tracker?

Measured and researched for the "can we use a small RF-DETR?" question:

| Detector | Where it runs | Latency | Verdict |
|---|---|---|---|
| YOLO11n 320×320 INT8 (what Microduck ships) | RK3566 NPU | ≈60 ms per look, ≈12–20 ms of it inference | **The on-robot path.** Retrain with the classes you need (Roboflow → RKNN export); trackers consumes its boxes for ≈1–2 ms more on CPU |
| RF-DETR Nano (30.5 M params, 384×384) | Apple M4 MacBook Pro, CPU | 33 ms measured (median 33, p95 35, idle machine) | ≈30 fps off-board; it is what the `DETECTOR=rfdetr` demo mode runs |
| RF-DETR Nano, split NPU-backbone + CPU-head ([rfdetr-on-rockchip-npu](https://github.com/AlexanderDhoore/rfdetr-on-rockchip-npu)) | **RK3588** (6 TOPS, A76 cores) | **198 ms measured** by that project | ≈5 fps on a chip several times stronger than Microduck's |
| RF-DETR Nano, same split | RK3566 (0.8 TOPS, A55) | est. 0.6–1 s | **Does not fit for live tracking.** DETR attention ops don't convert to RKNN end-to-end; even the split deployment is CPU-bound on the head, and the A55s are far slower than the RK3588's A76s |

So: on-device the recipe is *keep the YOLO-class NPU detector, add `trackers` for identity*; RF-DETR is the off-board / development detector. This split is exactly what the demo implements.

## Does a trainable detector reach the NPU?

RF-DETR does not: its attention does not convert to RKNN, which strands the
head on the A55s. YOLO-Lite does. Compiling `edge_n` at 320x320 with
rknn-toolkit2 for `rk3566`, the compiler places every operator it is given:

| Build | Operators on NPU | On CPU | Model size |
|---|---|---|---|
| FP16 | 87 | 9 | 1.37 MB |
| INT8 | 87 | 9 | 0.87 MB |

The nine CPU operators are the input node, four per-level `Transpose`s and
four output nodes: the head's output layout, which is CPU work in any
deployment. No convolution, activation or add falls back. YOLO-Lite uses ReLU
rather than SiLU, which is part of why it quantizes and maps this cleanly.

`.github/workflows/rknn.yml` reproduces this on every change to `edge/`, since
rknn-toolkit2 needs Linux x86_64, `onnx==1.15.0` (it calls `onnx.mapping`,
removed in 1.16) and `numpy<2`.

Two limits on what this shows. The graph comes from the architecture in
`roboflow/yololite` with untrained weights, so a model trained on the Roboflow
platform could differ if its package fuses NMS into the graph, in which case
the graph has to be cut at the head outputs. And placement is not speed: no
RK3566 has run this file.

## End-to-end pipeline rate on the robot

Adding up the pipeline that actually fits (detector on NPU, tracker on one A55 core):

| Stage | Cost | Source |
|---|---|---|
| YOLO11n 320×320 INT8, capture to boxes | ≈60 ms | Pollen's own on-robot figure (duck-detect) |
| SORT update, real cached detections | 2.0–4.5 ms | our benchmark × conservative 25× A55 scaling |
| **Total** | **≈62–65 ms** | **≈ 15 Hz tracked detection** |

Pollen's ≈60 ms is a whole look, capture through boxes, so preprocessing is
already inside it and is not added again here.

The tracker adds ≈4 % to the frame budget the detector already spends: on this
hardware, tracking is effectively free once you have detection.

### Where that 60 ms goes

Rockchip publish INT8 benchmarks for the RK3566 NPU itself
([rknn_model_zoo](https://github.com/airockchip/rknn_model_zoo)), taken at
maximum NPU frequency and excluding pre- and post-processing:

| Model (640×640, INT8) | RK3566/RK3568 | Per frame |
|---|---|---|
| MobileNetV2 (224×224) | 180.7 fps | 5.5 ms |
| YOLOv6n | 48.8 fps | 20.5 ms |
| YOLOv5n | 39.7 fps | 25.2 ms |
| YOLOv8n | 34.0 fps | 29.4 ms |
| YOLO11n | 20.6 fps | 48.5 ms |

YOLO11n costs 48.5 ms at 640×640. Microduck runs it at 320×320, a quarter of
the pixels, which puts its NPU inference on the order of 12–20 ms once fixed
per-inference overhead is allowed for. Pollen measure ≈60 ms for the whole
look. **Most of a Microduck look is camera pipeline, not inference.**

Two consequences. A faster detector buys less than its own benchmark suggests,
because the model is the smaller half of the budget: the capture path is where
the frame rate actually lives. And the tracker's 2–4.5 ms is even less
significant than the total above implies.

This split is derived, not measured. It scales Rockchip's 640×640 figure by
pixel count, which is roughly how convolutional FLOPs scale but ignores fixed
overhead, and Pollen's number may include work a leaner pipeline would not.
Settling it needs an RK3566 board.

**Verified at robot cadence**: the fetch demo run with tracker updates every
3rd frame (16.7 Hz, `TRACK_EVERY=3`, `minimum_consecutive_frames=1`, buffers
scaled) completes the full loop (lock, pursuit, beak-pick, kick) with the
lock ID held ≈8 s through the approach. The one fragile case at 15 Hz is
locking a ball in free flight: a ≈0.5 s toss yields only ≈8 looks, and in our
test the second in-flight lock failed (it succeeds at 50 Hz). Practical
mitigations: lock on the roll-out after landing (works), throw gentler, or run
the detector's full rate only during throw windows.

## Measured cost of `trackers` (v2.6, Apple M4 MacBook Pro)

Primary numbers replay **real cached detections** from the demo (RF-DETR Nano on
the duck's head camera, 1800 frames, 3228 detections, mean 1.79 objects/frame,
max 6; `bench_tracker.py`, 5 repeats):

All three run with matched settings for the parameters they share (activation
0.25, 3 consecutive frames, minimum IoU 0.3). ByteTrack keeps its own
`high_conf_det_threshold` default of 0.6, since that split is what its two-stage
association exists to exploit; 37% of these detections fall below it.

| Tracker | mean per update | p95 |
|---|---|---|
| SORT | 73-76 µs | 165-175 µs |
| SORT + BIoU(2.0) | 67-72 µs | 118-126 µs |
| ByteTrack | 64-68 µs | 167-184 µs |

Ranges are the min and max over five runs on an idle machine. At this scale
(mean 1.79 objects per frame) the three sit within about 10 µs of each other,
so the ordering is not a meaningful result. Read the table as "all three cost
well under 100 µs per update", not as a ranking.

Two ways to get a misleading ranking out of the same cache, both of which we hit
before settling on the table above. Forcing ByteTrack's high-confidence
threshold down to 0.25 so that every parameter "matches" makes it slower (81 µs):
it then runs one full-size association plus the bookkeeping for a second pass
that never has anything to do. Leaving both trackers on library defaults makes
it faster (58-59 µs): it activates tracks at 0.7 against SORT's 0.25, so it
maintains fewer tracklets on the same input. `bench_tracker.py` prints the
matched configuration and the defaults side by side.

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

Disk: full demo env 532 MB; a robot-minimal env (numpy, scipy, opencv-headless, supervision, trackers) estimates at ≈200 MB, irrelevant against 32 GB.

## Scaling to the Cortex-A55

No RK3566 was available to measure on, so the CPU numbers must be scaled. The scaling factor comes from Geekbench 6 single-core scores: the RK3566's Cortex-A55 at 1.8 GHz scores ≈210 ([Orange Pi 3B run](https://browser.geekbench.com/v6/cpu/2677440), [Notebookcheck](https://www.notebookcheck.net/Rockchip-RK3566-Processor-Benchmarks-and-Specs.741611.0.html) reports 203) against ≈3,800 for the Apple M4, a ratio of ≈18×. We apply **25×** as the conservative bound, since Geekbench weights vectorizable work more heavily than this mostly scalar association code. Taking that worst case:

- 16 objects, ByteTrack: 291 µs × 25 ≈ **7.3 ms/update**
- the demo's real detections, SORT: 76 µs mean × 25 ≈ **1.9 ms/update** (p95 175 µs ≈ 4.4 ms)

Against the budget: the detector delivers a frame every ≈60 ms (15 Hz). Even the worst scaled case consumes ≈12 % of one core at that rate, and the robot has four cores, with the control loop needing one. **Compute is not the problem, even before any optimization.** This scaling factor is the one unmeasured link in the chain; a 30-minute benchmark on any RK3566 dev board (≈$40) would close it.

## What does NOT fit on-board

- **DeepSORT / BoT-SORT with re-ID**: appearance embedding needs a CNN forward pass per detection. The 0.8 TOPS NPU is already saturated by detection; running embeddings on the A55s costs tens of ms each. Off-board only.
- **McByte**: needs per-frame masks (SAM-class model). Not on this silicon.
- **Torch anything**: no torch on the robot; the deployment path is ONNX/RKNN only.

## Three integration paths

| Path | Effort | Latency | Verdict |
|---|---|---|---|
| **Off-board**: subscribe to mediad's WebRTC stream, run detector + trackers on a laptop, command via the JSON-RPC socket API | Works today (the sim demo is exactly this pattern) | +20–50 ms network RTT | Right for demos, tutorials, promo content |
| **On-board Python sidecar**: a `trackd` daemon consuming duck-detect's boxes, speaking the same Unix-socket JSON-RPC as every other client | Small; aarch64 manylinux wheels exist for every dependency | none | Right for a community integration / blog post |
| **Rust port**: SORT/ByteTrack is a Kalman filter + Hungarian assignment; a `duck-track` crate beside duck-detect, using our clean-room implementations as the reference | Days, not hours | none, near-zero RAM | Right for upstreaming to pollen-robotics |

## Bottom line

Microduck ships detection with no identity. Our motion trackers close that gap within ≈2–7 ms and a few kilobytes of state on its CPU. The measured demo (target lock, kick-and-chase, ID persistence through occlusion) is behavior that detection alone cannot express. The honest caveats: the A55 scaling factor is estimated, not measured; a Python deployment carries ≈118 MB of interpreter and library RSS against the board's 1 GB, where a Rust port would carry almost none; and anything needing appearance embeddings stays off-board.
