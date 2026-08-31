#!/usr/bin/env python3
"""Track other Microducks with Microduck's own shipped duck detector.

Two microducks glide past each other in front of the observer. Head-camera
frames go through duck_detect.onnx, the exact single-class YOLO the robot
runs on its NPU, and SORTTracker keeps an ID on each duck through the
crossing. Detector docs: duck-detect/src/lib.rs in pollen-robotics/microduck.
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
sys.path.insert(0, os.path.join(RL, "scripts"))
os.chdir(RL)

from infer_policy import DEFAULT_POSE  # noqa: E402

if not os.path.exists(MODEL):
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/pollen-robotics/microduck/main/"
        "duck-detect/models/duck_detect.onnx",
        MODEL,
    )
import onnxruntime as ort  # noqa: E402

ROBOT_DIR = "src/mjlab_microduck/robot/microduck"
spec = mujoco.MjSpec.from_file(f"{ROBOT_DIR}/scene.xml")
spec.visual.global_.offwidth = spec.visual.global_.offheight = 1280
for c in spec.cameras:
    if c.name == "head_camera":
        th = math.radians(12)
        c.quat = [math.cos(th / 2), 0, math.sin(th / 2), 0]
        c.pos = [0.09, 0.0, -0.045]
        c.fovy = 90
for k in range(2):
    child = mujoco.MjSpec.from_file(f"{ROBOT_DIR}/robot_walk.xml")
    frame = spec.worldbody.add_frame(pos=[1.0, 0.0, 0.125])
    frame.attach_body(child.worldbody.first_body(), f"d{k}_", "")
model = spec.compile()
data = mujoco.MjData(model)

for i in range(model.nu):
    qi = int(model.jnt_qposadr[model.actuator_trnid[i, 0]])
    data.qpos[qi] = data.ctrl[i] = DEFAULT_POSE[i % 14]

BASES = []
for name in ("trunk_base_freejoint", "d0_trunk_base_freejoint", "d1_trunk_base_freejoint"):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    BASES.append((int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])))


def pin(base, x, y, yaw):
    qadr, vadr = base
    data.qpos[qadr : qadr + 3] = [x, y, 0.125]
    data.qpos[qadr + 3 : qadr + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    data.qvel[vadr : vadr + 6] = 0.0


renderer = mujoco.Renderer(model, height=960, width=544)
session = ort.InferenceSession(MODEL)
tracker = SORTTracker(frame_rate=50.0, iou=BIoU(buffer_ratio=2.0))
box_annotator = sv.BoxAnnotator(thickness=3)
label_annotator = sv.LabelAnnotator(text_scale=0.8, text_thickness=2)


def detect(frame):
    """Letterbox to 320x320 padded with 114 gray, per the duck-detect docs.

    The crate documents RGB input, but on these renders the model scores 0.96
    in BGR against 0.19 in RGB with identical box placement, so BGR it is.
    """
    scale = 320.0 / max(frame.shape[:2])
    resized = cv2.resize(frame[..., ::-1], None, fx=scale, fy=scale)
    pad_y, pad_x = (320 - resized.shape[0]) // 2, (320 - resized.shape[1]) // 2
    canvas = np.full((320, 320, 3), 114, np.uint8)
    canvas[pad_y : pad_y + resized.shape[0], pad_x : pad_x + resized.shape[1]] = resized
    inp = (canvas.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
    cx, cy, w, h, conf = session.run(None, {"images": inp})[0][0]
    keep = conf > 0.5
    if not keep.any():
        return sv.Detections.empty()
    xyxy = np.stack(
        [cx[keep] - w[keep] / 2, cy[keep] - h[keep] / 2,
         cx[keep] + w[keep] / 2, cy[keep] + h[keep] / 2], axis=1,
    )
    xyxy = (xyxy - [pad_x, pad_y, pad_x, pad_y]) / scale
    wh = xyxy[:, 2:] - xyxy[:, :2]
    order = cv2.dnn.NMSBoxes(
        np.hstack([xyxy[:, :2], wh]).tolist(), conf[keep].tolist(), 0.5, 0.5
    )
    order = np.array(order).flatten()
    return sv.Detections(
        xyxy=xyxy[order],
        confidence=conf[keep][order],
        class_id=np.zeros(len(order), dtype=int),
    )


frames = []
for step in range(300):
    u = min(step * 0.02 / 5.0, 1.0)
    pin(BASES[0], 0.0, 0.0, 0.0)
    pin(BASES[1], 1.05, 1.1 - 2.2 * u, math.pi + 0.4)
    pin(BASES[2], 1.35, -1.2 + 2.4 * u, math.pi - 0.4)
    for _ in range(4):
        mujoco.mj_step(model, data)
    renderer.update_scene(data, camera="head_camera")
    frame = np.ascontiguousarray(np.rot90(renderer.render()))
    tracked = tracker.update(detect(frame))
    tracked = tracked[tracked.tracker_id != -1]
    if len(tracked):
        frame = box_annotator.annotate(frame, tracked)
        frame = label_annotator.annotate(
            frame, tracked, labels=[f"duck #{i}" for i in tracked.tracker_id]
        )
    frames.append(frame)

imageio.mimwrite(os.path.join(ROOT, "duck_tracking.mp4"), frames, fps=50, quality=8)
print("wrote duck_tracking.mp4")
