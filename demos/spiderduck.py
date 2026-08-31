#!/usr/bin/env python3
"""Three microducks recognising each other: the pointing meme, in tracking.

Each duck stares at the other two through duck_detect.onnx, the single-class
YOLO the robot ships on its NPU, and each runs its own SORTTracker. The frame
is the three duck's-eye views plus the third-person shot.
"""

import math
import os
import sys

import imageio.v2 as imageio
import mujoco
import numpy as np
import supervision as sv

from trackers import SORTTracker
from trackers.utils.iou import BIoU

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RL = os.environ.get("MICRODUCK_RL", os.path.join(ROOT, "microduck_rl"))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(RL, "scripts"))
os.chdir(RL)

from duck_detector import DuckDetector  # noqa: E402
from infer_policy import DEFAULT_POSE  # noqa: E402

ROBOT_DIR = "src/mjlab_microduck/robot/microduck"
RADIUS = 0.62
START = [math.radians(a) for a in (90, 210, 330)]
DRIFT = [0.15, 0.17, 0.13]


def aim_camera(spec):
    for c in spec.cameras:
        if c.name == "head_camera":
            th = math.radians(12)
            c.quat = [math.cos(th / 2), 0, math.sin(th / 2), 0]
            c.pos = [0.09, 0.0, -0.045]
            c.fovy = 90


spec = mujoco.MjSpec.from_file(f"{ROBOT_DIR}/scene.xml")
spec.visual.global_.offwidth = spec.visual.global_.offheight = 1280
aim_camera(spec)
for k in range(2):
    child = mujoco.MjSpec.from_file(f"{ROBOT_DIR}/robot_walk.xml")
    aim_camera(child)
    frame = spec.worldbody.add_frame(pos=[1.0, 0.0, 0.125])
    frame.attach_body(child.worldbody.first_body(), f"d{k}_", "")
model = spec.compile()
data = mujoco.MjData(model)

for i in range(model.nu):
    data.qpos[int(model.jnt_qposadr[model.actuator_trnid[i, 0]])] = DEFAULT_POSE[i % 14]

PREFIXES = ("", "d0_", "d1_")
BASES, HEAD_YAW = [], []
for p in PREFIXES:
    jid = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, f"{p}trunk_base_freejoint"
    )
    BASES.append(int(model.jnt_qposadr[jid]))
    hid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{p}head_yaw")
    HEAD_YAW.append(int(model.jnt_qposadr[hid]))

pov = mujoco.Renderer(model, height=640, width=360)
wide = mujoco.Renderer(model, height=360, width=640)
chase = mujoco.MjvCamera()
chase.lookat = [0, 0, 0.15]
chase.distance = 1.9
chase.elevation = -20

detector = DuckDetector()
trackers = [
    SORTTracker(frame_rate=50.0, iou=BIoU(buffer_ratio=2.0)) for _ in PREFIXES
]
box_annotator = sv.BoxAnnotator(thickness=3)
label_annotator = sv.LabelAnnotator(text_scale=0.7, text_thickness=2)

frames = []
for step in range(400):
    t = step * 0.02
    for i, qadr in enumerate(BASES):
        angle = START[i] + DRIFT[i] * t
        yaw = angle + math.pi  # every duck faces the middle
        data.qpos[qadr : qadr + 3] = [
            RADIUS * math.cos(angle), RADIUS * math.sin(angle), 0.125,
        ]
        data.qpos[qadr + 3 : qadr + 7] = [
            math.cos(yaw / 2), 0, 0, math.sin(yaw / 2),
        ]
        data.qpos[HEAD_YAW[i]] = 0.3 * math.sin(1.1 * t + 2.1 * i)
    mujoco.mj_forward(model, data)

    panels = []
    for i, prefix in enumerate(PREFIXES):
        pov.update_scene(data, camera=f"{prefix}head_camera")
        view = np.ascontiguousarray(np.rot90(pov.render()))
        tracked = trackers[i].update(detector(view))
        tracked = tracked[tracked.tracker_id != -1]
        if len(tracked):
            view = box_annotator.annotate(view, tracked)
            view = label_annotator.annotate(
                view, tracked, labels=[f"duck #{j}" for j in tracked.tracker_id]
            )
        panels.append(view)
    chase.azimuth = 60 + 6 * math.sin(0.3 * t)
    wide.update_scene(data, camera=chase)
    panels.append(wide.render())
    frames.append(np.vstack([np.hstack(panels[:2]), np.hstack(panels[2:])]))

imageio.mimwrite(os.path.join(ROOT, "spiderduck.mp4"), frames, fps=50, quality=8)
print("wrote spiderduck.mp4")
