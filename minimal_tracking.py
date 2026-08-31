#!/usr/bin/env python3
"""Minimal Microduck sim + trackers integration.

Rolls a ball through the duck's head-camera view, tracks it with SORTTracker,
and writes a short annotated clip. This is the seam to copy into your own
project: frames in, sv.Detections through tracker.update(), IDs out.
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
RL = os.environ.get("MICRODUCK_RL", os.path.join(HERE, "microduck_rl"))
SCENE = os.path.join(RL, "src/mjlab_microduck/robot/microduck/scene_ball.xml")
sys.path.insert(0, os.path.join(RL, "scripts"))

from infer_policy import DEFAULT_POSE  # noqa: E402

spec = mujoco.MjSpec.from_file(SCENE)
spec.visual.global_.offwidth = 1280
spec.visual.global_.offheight = 1280
for c in spec.cameras:
    if c.name == "head_camera":
        th = math.radians(25)
        c.quat = [math.cos(th / 2), 0, math.sin(th / 2), 0]
        c.fovy = 90
model = spec.compile()
data = mujoco.MjData(model)

trunk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
tadr = int(model.jnt_qposadr[trunk])
data.qpos[tadr : tadr + 3] = [0.0, 0.0, 0.125]
data.qpos[tadr + 3 : tadr + 7] = [1, 0, 0, 0]
for i in range(model.nu):
    data.qpos[int(model.jnt_qposadr[model.actuator_trnid[i, 0]])] = DEFAULT_POSE[i]
data.ctrl[:] = DEFAULT_POSE[: model.nu]

jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
qadr, vadr = int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])
data.qpos[qadr : qadr + 3] = [0.8, -0.6, 0.035]
data.qvel[vadr + 1] = 1.2
mujoco.mj_forward(model, data)

renderer = mujoco.Renderer(model, height=960, width=544)
seg = mujoco.Renderer(model, height=960, width=544)
seg.enable_segmentation_rendering()
head_opt = mujoco.MjvOption()
head_opt.geomgroup[2] = 0
ball_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")

tracker = SORTTracker(frame_rate=50.0, iou=BIoU(buffer_ratio=2.0))
box_annotator = sv.BoxAnnotator(thickness=3)
label_annotator = sv.LabelAnnotator(text_scale=0.8, text_thickness=2)


def detect(seg_frame):
    mask = (seg_frame[..., 0] == ball_gid) & (
        seg_frame[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
    )
    ys, xs = np.nonzero(mask)
    if len(xs) < 3:
        return sv.Detections.empty()
    box = [[xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]]
    return sv.Detections(
        xyxy=np.array(box, dtype=float),
        confidence=np.ones(1),
        class_id=np.zeros(1, dtype=int),
    )


frames = []
for step in range(150):
    for _ in range(4):
        mujoco.mj_step(model, data)
    data.qpos[tadr : tadr + 3] = [0.0, 0.0, 0.125]
    data.qpos[tadr + 3 : tadr + 7] = [1, 0, 0, 0]
    data.qvel[int(model.jnt_dofadr[trunk]) : int(model.jnt_dofadr[trunk]) + 6] = 0.0
    renderer.update_scene(data, camera="head_camera", scene_option=head_opt)
    frame = np.ascontiguousarray(np.rot90(renderer.render()))
    seg.update_scene(data, camera="head_camera", scene_option=head_opt)
    tracked = tracker.update(detect(np.rot90(seg.render())))
    tracked = tracked[tracked.tracker_id != -1]
    if len(tracked):
        labels = [f"ball #{i}" for i in tracked.tracker_id]
        frame = box_annotator.annotate(frame, tracked)
        frame = label_annotator.annotate(frame, tracked, labels=labels)
    frames.append(frame)

out = os.path.join(HERE, "minimal_tracking.mp4")
imageio.mimwrite(out, frames, fps=50, quality=8)
print(f"wrote {out}")
