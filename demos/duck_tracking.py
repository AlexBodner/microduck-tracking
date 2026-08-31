#!/usr/bin/env python3
"""Track other Microducks with Microduck's own shipped duck detector.

Two microducks glide past each other in front of the observer duck. Frames
from the head camera go through duck_detect.onnx, the exact single-class
YOLO model the robot runs on its NPU, and SORTTracker keeps an ID on each
duck through the crossing.
"""

import math
import os
import sys
import urllib.request

import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np
import supervision as sv

from trackers import SORTTracker
from trackers.utils.iou import BIoU

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RL = os.environ.get("MICRODUCK_RL", os.path.join(ROOT, "microduck_rl"))
MODEL = os.path.join(ROOT, "duck_detect.onnx")
MODEL_URL = (
    "https://raw.githubusercontent.com/pollen-robotics/microduck/main/"
    "duck-detect/models/duck_detect.onnx"
)
sys.path.insert(0, os.path.join(RL, "scripts"))
os.chdir(RL)

from infer_policy import DEFAULT_POSE  # noqa: E402

if not os.path.exists(MODEL):
    print(f"downloading duck_detect.onnx from pollen-robotics/microduck ...")
    urllib.request.urlretrieve(MODEL_URL, MODEL)

import onnxruntime as ort  # noqa: E402

W, H = 960, 544

spec = mujoco.MjSpec.from_file("src/mjlab_microduck/robot/microduck/scene.xml")
spec.visual.global_.offwidth = 1280
spec.visual.global_.offheight = 1280
for c in spec.cameras:
    if c.name == "head_camera":
        th = math.radians(12)
        c.quat = [math.cos(th / 2), 0, math.sin(th / 2), 0]
        c.pos = [0.05, 0.0, -0.06]
        c.fovy = 90
GLIDERS = [(1.05, 1.1, math.pi + 0.4), (1.35, -1.2, math.pi - 0.4)]
for k, (px, py, yaw) in enumerate(GLIDERS):
    child = mujoco.MjSpec.from_file(
        "src/mjlab_microduck/robot/microduck/robot_walk.xml"
    )
    frame = spec.worldbody.add_frame(pos=[px, py, 0.125])
    frame.attach_body(child.worldbody.first_body(), f"d{k}_", "")
model = spec.compile()
data = mujoco.MjData(model)

trunk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
tadr = int(model.jnt_qposadr[trunk])
for i in range(model.nu):
    qi = int(model.jnt_qposadr[model.actuator_trnid[i, 0]])
    name = model.actuator(i).name
    base = name.split("_", 1)[1] if name.startswith("d") and "_" in name else name
    idx = [
        "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee",
        "left_ankle", "neck_pitch", "head_pitch", "head_yaw", "head_roll",
        "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee",
        "right_ankle",
    ]
    data.qpos[qi] = DEFAULT_POSE[idx.index(base)] if base in idx else 0.0
    data.ctrl[i] = data.qpos[qi]

GLIDER_ADRS = []
for k, (px, py, yaw) in enumerate(GLIDERS):
    jid = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, f"d{k}_trunk_base_freejoint"
    )
    GLIDER_ADRS.append((int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])))


def pin(adr, vadr, x, y, yaw):
    data.qpos[adr : adr + 3] = [x, y, 0.125]
    data.qpos[adr + 3 : adr + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    data.qvel[vadr : vadr + 6] = 0.0


pin(tadr, int(model.jnt_dofadr[trunk]), 0.0, 0.0, 0.0)
mujoco.mj_forward(model, data)

renderer = mujoco.Renderer(model, height=W, width=H)

session = ort.InferenceSession(MODEL)
tracker = SORTTracker(frame_rate=50.0, iou=BIoU(buffer_ratio=2.0))
box_annotator = sv.BoxAnnotator(thickness=3)
label_annotator = sv.LabelAnnotator(text_scale=0.8, text_thickness=2)


def detect(frame):
    """Letterbox to 320x320 padded with 114 gray, per the duck-detect crate docs.

    The crate documents RGB input, but on these renders the model scores 0.96
    in BGR against 0.19 in RGB with identical box placement, so BGR it is.
    """
    h, w = frame.shape[:2]
    scale = 320.0 / max(h, w)
    resized = cv2.resize(frame[..., ::-1], (int(w * scale), int(h * scale)))
    pad_y = (320 - resized.shape[0]) // 2
    pad_x = (320 - resized.shape[1]) // 2
    canvas = np.full((320, 320, 3), 114, np.uint8)
    canvas[pad_y : pad_y + resized.shape[0], pad_x : pad_x + resized.shape[1]] = resized
    inp = (canvas.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
    cx, cy, bw, bh, conf = session.run(None, {"images": inp})[0][0]
    keep = conf > 0.5
    if not keep.any():
        return sv.Detections.empty()
    x1 = (cx[keep] - bw[keep] / 2 - pad_x) / scale
    y1 = (cy[keep] - bh[keep] / 2 - pad_y) / scale
    x2 = (cx[keep] + bw[keep] / 2 - pad_x) / scale
    y2 = (cy[keep] + bh[keep] / 2 - pad_y) / scale
    xyxy = np.stack([x1, y1, x2, y2], axis=1)
    scores = conf[keep]
    order = cv2.dnn.NMSBoxes(
        [
            (float(a), float(b), float(c - a), float(d - b))
            for a, b, c, d in xyxy
        ],
        scores.tolist(), 0.5, 0.5,
    )
    order = np.array(order).flatten()
    return sv.Detections(
        xyxy=xyxy[order],
        confidence=scores[order],
        class_id=np.zeros(len(order), dtype=int),
    )


frames = []
for step in range(300):
    t = step * 0.02
    u = min(t / 5.0, 1.0)
    pin(*GLIDER_ADRS[0], 1.05, 1.1 - 2.2 * u, GLIDERS[0][2])
    pin(*GLIDER_ADRS[1], 1.35, -1.2 + 2.4 * u, GLIDERS[1][2])
    pin(tadr, int(model.jnt_dofadr[trunk]), 0.0, 0.0, 0.0)
    for _ in range(4):
        mujoco.mj_step(model, data)
    renderer.update_scene(data, camera="head_camera")
    frame = np.ascontiguousarray(np.rot90(renderer.render()))
    tracked = tracker.update(detect(frame))
    tracked = tracked[tracked.tracker_id != -1]
    if len(tracked):
        labels = [f"duck #{i}" for i in tracked.tracker_id]
        frame = box_annotator.annotate(frame, tracked)
        frame = label_annotator.annotate(frame, tracked, labels=labels)
    frames.append(frame)

out_path = os.path.join(ROOT, "duck_tracking.mp4")
imageio.mimwrite(out_path, frames, fps=50, quality=8)
print(f"wrote {out_path}")
