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
ball2 = spec.worldbody.add_body(name="ball2", pos=[0.9, 0.6, 0.035])
ball2.add_freejoint(name="ball2_free")
ball2.add_geom(
    name="ball2_geom",
    type=mujoco.mjtGeom.mjGEOM_SPHERE,
    size=[0.035, 0, 0],
    rgba=[1, 0.55, 0, 1],
    condim=6,
    friction=[0.5, 0.005, 0.001],
    mass=0.015,
)
model = spec.compile()
data = mujoco.MjData(model)

trunk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
tadr = int(model.jnt_qposadr[trunk])
data.qpos[tadr : tadr + 3] = [0.0, 0.0, 0.125]
data.qpos[tadr + 3 : tadr + 7] = [1, 0, 0, 0]
for i in range(model.nu):
    data.qpos[int(model.jnt_qposadr[model.actuator_trnid[i, 0]])] = DEFAULT_POSE[i]
data.ctrl[:] = DEFAULT_POSE[: model.nu]

for jname, pos, vy in (
    ("ball_free", [0.8, -0.6, 0.035], 1.2),
    ("ball2_free", [1.1, 0.7, 0.035], -1.0),
):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
    qadr, vadr = int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])
    data.qpos[qadr : qadr + 3] = pos
    data.qvel[vadr + 1] = vy
mujoco.mj_forward(model, data)

renderer = mujoco.Renderer(model, height=960, width=544)
seg = mujoco.Renderer(model, height=960, width=544)
seg.enable_segmentation_rendering()
head_opt = mujoco.MjvOption()
head_opt.geomgroup[2] = 0
BALL_GIDS = [
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)
    for n in ("ball_geom", "ball2_geom")
]

tracker = SORTTracker(frame_rate=50.0, iou=BIoU(buffer_ratio=2.0))
box_annotator = sv.BoxAnnotator(thickness=3)
label_annotator = sv.LabelAnnotator(text_scale=0.8, text_thickness=2)


def detect(seg_frame):
    is_geom = seg_frame[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
    masks = np.stack([(seg_frame[..., 0] == gid) & is_geom for gid in BALL_GIDS])
    visible = masks.reshape(len(BALL_GIDS), -1).sum(axis=1) >= 3
    if not visible.any():
        return sv.Detections.empty()
    boxes = sv.mask_to_xyxy(masks[visible], coordinate_convention="exclusive")
    return sv.Detections(
        xyxy=boxes.astype(float),
        confidence=np.ones(len(boxes)),
        class_id=np.zeros(len(boxes), dtype=int),
    )


out = os.path.join(HERE, "minimal_tracking.mp4")
writer = imageio.get_writer(out, fps=50, quality=8)
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
    writer.append_data(frame)

writer.close()
print(f"wrote {out}")
