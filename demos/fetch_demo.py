#!/usr/bin/env python3
"""Microduck plays fetch, powered by `trackers`.

The duck stands among IDENTICAL resting balls. An off-screen owner throws
another identical ball through the scene. trackers.SORTTracker keeps an ID on
every ball; the thrown one is singled out purely by TRACK MOTION (its image-
space velocity), and the duck locks that ID and goes to play with exactly that
ball, ignoring the identical distractors. Each new throw re-locks onto the new
moving track. Detection alone cannot express any of this — "the ball that was
just thrown" only exists as a track.

The controller is closed on vision: bearing comes from the tracked box's
centre, range from its apparent diameter against the ball's known 70 mm, and
the head yaw joint puts that bearing back in the body frame. The simulator's
ball positions are used only by the owner's hand to throw, never by the duck.

Pretrained Microduck policies (walking/stand/pick) run in CPU MuJoCo.
Detections: set DETECTOR=rfdetr to run RF-DETR on the rendered head-camera
frames; the default is the simulator's segmentation renderer as a perfect-
detector stand-in. Either way the boxes flow through the tracker into the
controller.
"""

import math
import os
import sys

import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np
import supervision as sv

from trackers import SORTTracker
from trackers.utils.iou import BIoU

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
# See README.md: clone pollen-robotics/microduck_rl in the repo root and
# download the ONNX policies from pollen-robotics/microduck-policies (HF Hub).
RL = os.environ.get("MICRODUCK_RL", os.path.join(ROOT, "microduck_rl"))
POLICIES = os.environ.get("MICRODUCK_POLICIES", os.path.join(ROOT, "policies"))
sys.path.insert(0, os.path.join(RL, "scripts"))
CWD = os.getcwd()  # DUMP_DETECTIONS paths resolve against the caller, not RL
os.chdir(RL)  # infer_policy uses repo-relative XML paths

from infer_policy import (  # noqa: E402
    MICRODUCK_BALL_XML,
    PolicyInference,
)

SIM_SECONDS = float(os.environ.get("SIM_SECONDS", 28))
THROW_SEED = int(os.environ.get("THROW_SEED", 1))   # vary where throws land
HEADLESS = bool(os.environ.get("HEADLESS"))         # skip video for batch runs
TRIAL = bool(os.environ.get("TRIAL"))               # report each grab
FIRST_THROW = 1.2                # the owner's first throw
MAX_THROWS = 2                   # throws in the video
RETHROW_DELAY = 2.0              # owner waits for the duck to finish its kick
LOCK_WINDOW = 8.0                # seconds after a throw to lock the fast track
LOCK_SPEED = 4.0                 # px per tracked frame that means "thrown"
BALL_DIAMETER = 0.07             # metres, for range from apparent size
# The beak pick reaches a ball up to 0.15 m ahead of the feet and misses at
# 0.17 m, measured by triggering it against a ball at known offsets. Ducking
# the head far enough down keeps the ball in frame to 0.06 m, so the duck can
# watch it the whole way in and stop while it is still in reach, instead of
# walking into it and booting it away.
GRAB_RANGE = 0.20                # range at which both detectors still see it
CLOSING_STEP = 0.30              # seconds of straight walk before the pick
STALE_FIX = 0.4                  # seconds without a detection before standing
TRACK_EVERY = int(os.environ.get("TRACK_EVERY", 1))  # 3 = ~16.7 Hz robot cadence
LOCK_SPEED *= TRACK_EVERY        # a longer interval moves the ball further
MAIN_W, MAIN_H = 1280, 720       # chase view canvas
PANEL_W, PANEL_H = 960, 540      # duck POV render size (annotated, then shrunk)
INSET_W, INSET_H = 480, 270      # POV picture-in-picture, top-left corner

spec = mujoco.MjSpec.from_file(MICRODUCK_BALL_XML)
spec.visual.global_.offwidth = 1280
spec.visual.global_.offheight = 1280  # head cam renders portrait, needs height
# Park styling: mowed-lawn greens instead of the blue checker, a hazier sky,
# matte ground, and a handful of low-poly trees on the horizon.
for t in spec.textures:
    if t.name == "groundplane":
        t.rgb1 = [0.40, 0.56, 0.24]
        t.rgb2 = [0.33, 0.49, 0.20]
        t.mark = mujoco.mjtMark.mjMARK_NONE
    elif t.type == mujoco.mjtTexture.mjTEXTURE_SKYBOX:
        t.rgb1 = [0.46, 0.70, 0.93]
        t.rgb2 = [0.88, 0.94, 1.0]
for m in spec.materials:
    if m.name == "groundplane":
        m.reflectance = 0.0
TREES = [(-2.5, 1.8, 1.0), (3.2, 2.4, 1.3), (2.0, -3.0, 0.9), (-3.0, -2.2, 1.15), (4.0, -0.5, 1.05)]
for k, (tx, ty, sc) in enumerate(TREES):
    tb = spec.worldbody.add_body(name=f"tree{k}", pos=[tx, ty, 0])
    tb.add_geom(
        name=f"trunk{k}", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[0.06 * sc, 0.35 * sc, 0], pos=[0, 0, 0.35 * sc],
        rgba=[0.42, 0.30, 0.20, 1], contype=0, conaffinity=0,
    )
    tb.add_geom(
        name=f"canopy{k}", type=mujoco.mjtGeom.mjGEOM_ELLIPSOID,
        size=[0.42 * sc, 0.42 * sc, 0.5 * sc], pos=[0, 0, 0.95 * sc],
        rgba=[0.24 + 0.05 * (k % 3), 0.5 + 0.04 * (k % 2), 0.2, 1],
        contype=0, conaffinity=0,
    )
# The MJCF head_camera points backward with a 90-degree roll; face it forward,
# pitch it 25 degrees down toward the ground the robot acts on, widen to ~90.
for c in spec.cameras:
    if c.name == "head_camera":
        th = math.radians(25)
        c.quat = [math.cos(th / 2), 0, math.sin(th / 2), 0]
        c.fovy = 90
        # The lens stays where Pollen mounts it. That puts it on the beak's
        # arc, so during the peck it passes through the ball and the view goes
        # briefly empty; moving it forward would avoid that but would stop
        # being this robot's camera.
# Rolling friction so thrown/kicked balls settle within ~1 m (condim=6
# activates it; the default condim=3 silently ignores the coefficient).
for g in spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM):
    if g.name == "ball_geom":
        g.condim = 6
        g.friction = [0.5, 0.005, 0.004]
# Identical DISTRACTOR balls resting in the field. "ball_free" (from the base
# scene) is the play ball the owner throws.
DISTRACTOR_POS = [
    (0.55, 0.35),
    (0.75, -0.30),
    (1.05, 0.15),
    (1.20, -0.45),
]
for k, (bx, by) in enumerate(DISTRACTOR_POS, start=2):
    bb = spec.worldbody.add_body(name=f"ball{k}", pos=[bx, by, 0.035])
    bb.add_freejoint(name=f"ball{k}_free")
    bb.add_geom(
        name=f"ball{k}_geom",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.035, 0, 0],
        rgba=[1, 0.55, 0, 1],
        condim=6,
        friction=[0.5, 0.005, 0.001],
        mass=0.015,
    )
# The owner's hand: a kinematic (mocap) body that sweeps in from behind the
# chase camera carrying the ball, releases it, and retracts.
SKIN = [0.93, 0.77, 0.64, 1.0]
hand = spec.worldbody.add_body(name="owner_hand", pos=[6.0, 6.0, 0.5], mocap=True)
hand.add_geom(
    name="palm", type=mujoco.mjtGeom.mjGEOM_BOX,
    size=[0.050, 0.042, 0.011], rgba=SKIN, contype=0, conaffinity=0,
)
for fi, fy in enumerate((-0.030, -0.010, 0.010, 0.030)):
    hand.add_geom(
        name=f"finger{fi}", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        fromto=[0.05, fy, 0.004, 0.105, fy * 1.5, 0.018],
        size=[0.0085, 0, 0], rgba=SKIN, contype=0, conaffinity=0,
    )
hand.add_geom(
    name="thumb", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
    fromto=[0.01, -0.042, 0.004, 0.045, -0.075, 0.022],
    size=[0.009, 0, 0], rgba=SKIN, contype=0, conaffinity=0,
)
model = spec.compile()
model.opt.timestep = 0.005
data = mujoco.MjData(model)

policy = PolicyInference(
    model,
    data,
    walking_onnx_path=os.path.join(POLICIES, "alpha_walking.onnx"),
    standing_onnx_path=os.path.join(POLICIES, "alpha_stand.onnx"),
    kick_left_onnx_path=os.path.join(POLICIES, "ball_kick_left.onnx"),
    kick_right_onnx_path=os.path.join(POLICIES, "ball_kick_right.onnx"),
    ground_pick_onnx_path=os.path.join(POLICIES, "alpha_ground_pick.onnx"),
    ground_pick_period=2.8,  # snappier peck than the 4.0 s default
    new_cmd_obs=True,
    use_projected_gravity=True,
    kick_duration=2.0,
)

# The kick trigger snaps the ball to the trained kick spot ("_place_ball") —
# a visible teleport. We gate kicks tightly around that spot instead and let
# the policy play the ball where it actually lies. This reaches into a
# private method of microduck_rl, verified against upstream d424a0c.
policy._place_ball = lambda behavior: None

# Initial pose (mirrors infer_policy.main)
adr = policy._trunk_qpos_adr
data.qpos[adr + 0 : adr + 3] = [0.0, 0.0, 0.125]
data.qpos[adr + 3 : adr + 7] = [1, 0, 0, 0]
for i, qi in enumerate(policy.joint_qpos_indices):
    data.qpos[qi] = policy.default_pose[i]
data.ctrl[:] = policy.default_pose

PLAY_JOINT = "ball_free"
BALL_JOINTS = [PLAY_JOINT] + [
    f"ball{k}_free" for k in range(2, 2 + len(DISTRACTOR_POS))
]
BALLS = {}
for jname in BALL_JOINTS:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
    BALLS[jname] = (int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid]))
# Play ball waits far out of sight until the first throw.
data.qpos[BALLS[PLAY_JOINT][0] : BALLS[PLAY_JOINT][0] + 3] = [6.0, 6.0, 0.035]
mujoco.mj_forward(model, data)

BALL_GEOM_IDS = []
for gname in ["ball_geom"] + [f"ball{k}_geom" for k in range(2, 2 + len(DISTRACTOR_POS))]:
    BALL_GEOM_IDS.append(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, gname))

# Renderers: chase view, head RGB, head segmentation
main_r = mujoco.Renderer(model, height=MAIN_H, width=MAIN_W)
# Head cam renders portrait and is rotated 90 degrees to landscape (the
# mounted camera frame is rolled relative to the image we want).
head_r = mujoco.Renderer(model, height=PANEL_W, width=PANEL_H)
seg_r = mujoco.Renderer(model, height=PANEL_W, width=PANEL_H)
seg_r.enable_segmentation_rendering()
# Hide the soft-jaw meshes (group 2) that sit in front of the head camera.
FOCAL_PX = (PANEL_W / 2) / math.tan(math.radians(90) / 2)
_head_yaw_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "head_yaw")
head_yaw_adr = int(model.jnt_qposadr[_head_yaw_jid])
head_opt = mujoco.MjvOption()
head_opt.geomgroup[2] = 0

chase = mujoco.MjvCamera()
chase.type = mujoco.mjtCamera.mjCAMERA_TRACKING
chase.trackbodyid = policy.trunk_base_id
chase.distance = 1.25
chase.elevation = -22
chase.azimuth = 18.0  # steered behind the duck every frame (over-the-shoulder)
chase_az = 18.0

tracker = SORTTracker(
    lost_track_buffer=150 // TRACK_EVERY,
    frame_rate=50.0 / TRACK_EVERY,
    minimum_consecutive_frames=2 if TRACK_EVERY == 1 else 1,
    minimum_iou_threshold=0.05,
    # Buffered IoU: expand boxes for association, so the tiny ball box still
    # matches across the walking head-bob and at long range.
    iou=BIoU(buffer_ratio=2.0),
)
MAGENTA = sv.Color(255, 64, 255)
GRAY = sv.Color(235, 235, 235)
target_box = sv.BoxCornerAnnotator(thickness=6, corner_length=18, color=MAGENTA)
target_label = sv.LabelAnnotator(
    text_scale=0.9, text_thickness=2, color=MAGENTA,
    text_position=sv.Position.TOP_CENTER,
)
other_box = sv.BoxAnnotator(thickness=2, color=GRAY)
other_label = sv.LabelAnnotator(text_scale=0.7, text_thickness=2, color=GRAY)
trace_ann = sv.TraceAnnotator(thickness=4, trace_length=15, color=MAGENTA)
display_num: dict[int, int] = {}


def display_id(tid):
    tid = int(tid)
    if tid not in display_num:
        display_num[tid] = len(display_num) + 1
    return display_num[tid]


# --- optional real detector -------------------------------------------------
DETECTOR = os.environ.get("DETECTOR", "segmentation")
rfdetr_model = None
if DETECTOR == "rfdetr":
    from rfdetr import RFDETRNano

    rfdetr_model = RFDETRNano()
    rfdetr_model.optimize_for_inference()


def trunk_yaw_frame():
    trunk_xy = data.qpos[adr : adr + 2].copy()
    qw, qx, qy, qz = data.qpos[adr + 3 : adr + 7]
    yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    return trunk_xy, yaw


PH_REACH, PH_CARRY, PH_THROW, PH_RETRACT = 0.35, 0.40, 0.30, 0.30
hand_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "owner_hand")
hand_mid = int(model.body_mocapid[hand_bid])
throw_anim = None  # dict while the hand is animating
throw_rng = np.random.default_rng(THROW_SEED)


def _smooth(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def _bezier(p0, p1, p2, u):
    return (1 - u) ** 2 * p0 + 2 * u * (1 - u) * p1 + u**2 * p2


def start_throw(t):
    """Owner's throw, staged: the hand reaches down to wherever the ball lies,
    carries it back behind the duck, and lobs it underhand into the field."""
    trunk_xy, yaw = trunk_yaw_frame()
    f = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    r = np.array([f[1], -f[0], 0.0])  # the duck's right
    base = np.array([trunk_xy[0], trunk_xy[1], 0.0])
    qadr = BALLS[PLAY_JOINT][0]
    ball = data.qpos[qadr : qadr + 3].copy()
    first = ball[2] < 0 or np.linalg.norm(ball[:2] - base[:2]) > 3.0
    hold = base - 0.72 * f - 0.30 * r + [0, 0, 0.40]
    return {
        "t0": t,
        "yaw": yaw,
        "first": first,           # first throw: hand enters already holding it
        "enter": hold + [0, 0, 0.45],
        "ball0": ball,
        "hold": hold,
        "release": base - 0.30 * f - 0.14 * r + [0, 0, 0.50],
        "vel": (
            (1.5 + throw_rng.uniform(-0.30, 0.45)) * f
            + (0.28 + throw_rng.uniform(-0.45, 0.45)) * r
            + [0, 0, 1.3 + throw_rng.uniform(-0.15, 0.25)]
        ),
        "released": False,
    }


def _hand_quat(yaw, pitch):
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    # yaw about z composed with pitch about the hand's y (palm tilt)
    return [cy * cp, -sy * sp, cp * 0 + sp * cy, sy * cp]


def animate_hand(anim, t):
    """Advance the hand animation; returns False when finished."""
    qadr, vadr = BALLS[PLAY_JOINT]
    dt_ = t - anim["t0"]
    yaw = anim["yaw"]
    hold, rel = anim["hold"], anim["release"]

    if anim["first"]:
        reach_end = 0.0
    else:
        reach_end = PH_REACH + PH_CARRY

    if not anim["first"] and dt_ <= PH_REACH:
        # Reach: from the entry point down to the resting ball.
        u = _smooth(dt_ / PH_REACH)
        pos = anim["enter"] + u * (anim["ball0"] + [0, 0, 0.055] - anim["enter"])
        data.mocap_pos[hand_mid] = pos
        data.mocap_quat[hand_mid] = _hand_quat(yaw, 0.35 * u)
        return True
    if not anim["first"] and dt_ <= reach_end:
        # Carry: ball rides on the palm back to the hold point.
        u = _smooth((dt_ - PH_REACH) / PH_CARRY)
        p0 = anim["ball0"] + [0, 0, 0.055]
        mid = 0.5 * (p0 + hold) + [0, 0, 0.22]
        pos = _bezier(p0, mid, hold, u)
        data.mocap_pos[hand_mid] = pos
        data.mocap_quat[hand_mid] = _hand_quat(yaw, 0.35 * (1 - u))
        data.qpos[qadr : qadr + 3] = pos + [0, 0, 0.045]
        data.qpos[qadr + 3 : qadr + 7] = [1, 0, 0, 0]
        data.qvel[vadr : vadr + 6] = 0.0
        return True
    if dt_ <= reach_end + PH_THROW:
        # Underhand scoop: dip below the line then sweep up to the release,
        # wrist rolling from pulled-back to followed-through.
        u = _smooth((dt_ - reach_end) / PH_THROW)
        dip = 0.5 * (hold + rel) - [0, 0, 0.16]
        pos = _bezier(hold, dip, rel, u)
        data.mocap_pos[hand_mid] = pos
        data.mocap_quat[hand_mid] = _hand_quat(yaw, -0.45 + 0.8 * u)
        data.qpos[qadr : qadr + 3] = pos + [0, 0, 0.045]
        data.qpos[qadr + 3 : qadr + 7] = [1, 0, 0, 0]
        data.qvel[vadr : vadr + 6] = 0.0
        return True
    if dt_ <= reach_end + PH_THROW + PH_RETRACT:
        if not anim["released"]:
            anim["released"] = True
            data.qvel[vadr : vadr + 6] = 0.0
            data.qvel[vadr : vadr + 3] = anim["vel"]
        u = _smooth((dt_ - reach_end - PH_THROW) / PH_RETRACT)
        data.mocap_pos[hand_mid] = rel + u * (anim["enter"] - rel + [0, 0, 0.1])
        data.mocap_quat[hand_mid] = _hand_quat(yaw, 0.35 * (1 - u))
        return True
    data.mocap_pos[hand_mid] = [6.0, 6.0, 0.5]  # park out of sight
    return False


def seg_boxes(seg):
    """Segmentation frame -> list of xyxy ball boxes."""
    is_geom = seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
    masks = np.stack([(seg[..., 0] == gid) & is_geom for gid in BALL_GEOM_IDS])
    visible = masks.reshape(len(BALL_GEOM_IDS), -1).sum(axis=1) >= 3
    if not visible.any():
        return []
    boxes = sv.mask_to_xyxy(masks[visible], coordinate_convention="exclusive")
    return boxes.tolist()


def detect(head_rgb, boxes):
    """Detections the tracker consumes.

    segmentation mode: the oracle boxes themselves.
    rfdetr mode: RF-DETR on the rendered frame ("sports ball" class); the
    oracle boxes are used for nothing but control-side identification.
    """
    if rfdetr_model is not None:
        dets = rfdetr_model.predict(head_rgb, threshold=0.3)
        keep = np.isin(dets.class_id, (37, 53, 55))
        return dets[keep]
    if not boxes:
        return sv.Detections.empty()
    n = len(boxes)
    return sv.Detections(
        xyxy=np.array(boxes, dtype=float),
        confidence=np.ones(n),
        class_id=np.zeros(n, dtype=int),
    )


def detection_for(track_box, detections, minimum_iou=0.3):
    """The detection this track is standing on, or None if it is coasting."""
    best, best_iou = None, minimum_iou
    for box in detections.xyxy:
        wide = min(track_box[2], box[2]) - max(track_box[0], box[0])
        high = min(track_box[3], box[3]) - max(track_box[1], box[1])
        if wide <= 0 or high <= 0:
            continue
        overlap = wide * high
        union = (
            (track_box[2] - track_box[0]) * (track_box[3] - track_box[1])
            + (box[2] - box[0]) * (box[3] - box[1])
            - overlap
        )
        if union > 0 and overlap / union > best_iou:
            best, best_iou = box, overlap / union
    return best


def measure(box, head_yaw):
    """Bearing and range to a ball from its box alone.

    The rotated head view spans the camera's 90 degree fovy across its 960 px
    width, so the focal length is 480 px. A sphere of known diameter gives
    range from its apparent size; the box centre gives bearing in the head
    frame, and adding the head yaw joint (an encoder reading, not simulator
    state) puts that bearing back in the body frame.
    """
    centre_x = (box[0] + box[2]) / 2
    diameter_px = max(box[2] - box[0], box[3] - box[1])
    bearing = -math.atan2(centre_x - PANEL_W / 2, FOCAL_PX) + head_yaw
    return bearing, FOCAL_PX * BALL_DIAMETER / max(diameter_px, 1.0)


def report_grab(t):
    """Distance from the beak's target to the ball the owner actually threw."""
    trunk_xy, yaw = trunk_yaw_frame()
    qadr = BALLS[PLAY_JOINT][0]
    d = data.qpos[qadr : qadr + 2] - trunk_xy
    c, sn = math.cos(-yaw), math.sin(-yaw)
    fwd, left = c * d[0] - sn * d[1], sn * d[0] + c * d[1]
    print(f"TRIAL_GRAB t={t:5.1f} play_ball_distance={math.hypot(fwd, left):.3f}")


control_dt = 4 * model.opt.timestep
n_steps = int(SIM_SECONDS / control_dt)
out = os.path.join(ROOT, "fetch_demo.mp4")
writer = None if HEADLESS else imageio.get_writer(out, fps=50, quality=8)
frame_count = 0
dump_rows = []  # (frame, x1, y1, x2, y2, confidence) when DUMP_DETECTIONS is set
kick_cooldown = 0.0
frames_with_tracks = 0
target_id = None            # locked tracker ID (the thrown ball's track)
target_fix = None           # (bearing, range) from the last real detection
target_last_seen = -1.0
track_speed: dict[int, float] = {}   # tracker_id -> EMA image-space speed
track_center: dict[int, tuple] = {}  # tracker_id -> last box center
track_birth: dict[int, float] = {}   # tracker_id -> first time seen
last_throw_t = -1.0
throws_done = 0
next_throw_at = FIRST_THROW
prev_behavior = None
grabbed_this_throw = False
lock_open_until = -1.0      # lock window after each throw
grab_settle_until = None    # stand still before triggering the (blind) pick
closing_until = None        # one fixed step between seeing it and pecking
recover_until = -1.0        # play window after the pick
play_start = 0.0
prev_pick_mode = False

for step in range(n_steps):
    t = step * control_dt
    policy.update_behavior(control_dt)
    kick_cooldown = max(0.0, kick_cooldown - control_dt)

    # The owner throws when the duck is free: first at FIRST_THROW, then a
    # beat after each kick lands (or a timeout if the fetch drags on).
    if (
        throws_done < MAX_THROWS
        and t >= next_throw_at
        and throw_anim is None
        and policy.behavior_mode is None
        and not policy.ground_pick_mode
    ):
        throw_anim = start_throw(t)
        throws_done += 1
        release_t = t + (
            (0.0 if throw_anim["first"] else PH_REACH + PH_CARRY) + PH_THROW
        )
        last_throw_t = release_t
        lock_open_until = release_t + LOCK_WINDOW
        next_throw_at = t + 18.0  # timeout fallback if the fetch stalls
        grabbed_this_throw = False
        closing_until = None
        target_id = None    # wait for the new moving track
        target_fix = None
    if throw_anim is not None and not animate_hand(throw_anim, t):
        throw_anim = None

    policy.update_ground_pick_phase(control_dt)
    if prev_pick_mode and not policy.ground_pick_mode:
        recover_until = t + 1.9  # playful shuffle happens in this window
        play_start = t
        next_throw_at = t + RETHROW_DELAY  # touch done: the owner retrieves
    prev_pick_mode = policy.ground_pick_mode
    if policy.behavior_mode is None and not policy.ground_pick_mode:
        if grab_settle_until is not None:
            # The pick policy was trained from a standing start: stop first.
            policy.set_vel_cmd(0.0, 0.0, 0.0)
            policy.head_offset[:] = 0.0
            if t >= grab_settle_until:
                grab_settle_until = None
                policy.trigger_ground_pick()
        elif t < recover_until:
            if grabbed_this_throw:
                # Post-touch play: admire the ball, then an excited wiggle
                # while the owner reaches in — a dog asking for the next throw.
                ph = t - play_start
                if ph < 0.5:
                    policy.set_vel_cmd(0.0, 0.0, 0.0)
                    policy.head_offset[:] = [0.0, 0.45, 0.0, 0.0]
                elif ph < 1.2:
                    policy.set_vel_cmd(0.0, 0.0, 0.9)
                    policy.head_offset[:] = [0.0, 0.2, 0.35, 0.0]
                else:
                    policy.set_vel_cmd(0.0, 0.0, -0.9)
                    policy.head_offset[:] = [0.0, 0.1, -0.2, 0.0]
            else:
                policy.set_vel_cmd(0.0, 0.0, 0.0)
                policy.head_offset[:] = 0.0
        elif target_fix is None:
            # Waiting for the owner: stand, head level, watch the field.
            policy.set_vel_cmd(0.0, 0.0, 0.0)
            policy.head_offset[:] = [0.0, 0.1, 0.0, 0.0]
        else:
            # Bearing and range both come from the last real detection.
            err, dist = target_fix
            # Duck-like gaze toward the LOCKED target only.
            policy.head_offset[:] = [
                0.0,
                float(np.clip(1.2 * (0.55 - dist), 0.0, 0.55)),
                float(np.clip(err, -0.5, 0.5)),
                0.0,
            ]
            fresh = t - target_last_seen < STALE_FIX
            centred = abs(err) < 0.35
            if grabbed_this_throw:
                # Touch done: stand over the ball and wait for the owner.
                policy.set_vel_cmd(0.0, 0.0, 0.0)
            elif closing_until is not None:
                # One fixed step to cover the last few centimetres, started
                # from a detection rather than a guess.
                policy.set_vel_cmd(lin_vel_x=0.3, lin_vel_y=0.0, ang_vel_z=0.0)
                if t >= closing_until:
                    closing_until = None
                    grab_settle_until = t + 0.3
                    grabbed_this_throw = True
                    if TRIAL:
                        report_grab(t)
            elif fresh and dist < GRAB_RANGE and centred:
                # Seen, close and lined up. The ball sits a few centimetres
                # beyond the beak at the range both detectors still resolve,
                # so walk one fixed step before ducking rather than stopping.
                closing_until = t + CLOSING_STEP
            elif not fresh:
                # No detection under the target: stand and look rather than
                # walk on a guess.
                policy.set_vel_cmd(0.0, 0.0, 0.0)
            else:
                # The policy only breaks into a gait near its max command, and
                # turns far better while walking, so always push full speed.
                policy.set_vel_cmd(
                    lin_vel_x=0.3,
                    lin_vel_y=0.0,
                    ang_vel_z=float(np.clip(2.0 * err, -1.2, 1.2)),
                )

    action = policy.infer()
    policy.apply_action(action)
    for _ in range(4):
        mujoco.mj_step(model, data)

    # --- render + track (every control step -> 50 fps) ---
    # Over-the-shoulder chase cam: follow the trunk yaw with smoothing so the
    # third-person view always shows what the duck is walking toward.
    _, yaw_now = trunk_yaw_frame()
    # Swing to a side view while the duck works the ball with its beak, so
    # the contact isn't hidden behind its body from the rear camera.
    picking = policy.ground_pick_mode or grab_settle_until is not None
    az_offset = 75.0 if picking or t < recover_until else 18.0
    az_t = math.degrees(yaw_now) + az_offset
    chase_az += 0.04 * ((az_t - chase_az + 180.0) % 360.0 - 180.0)
    chase.azimuth = chase_az
    if HEADLESS:
        frame = None
    else:
        main_r.update_scene(data, camera=chase)
        frame = main_r.render()
    head_r.update_scene(data, camera="head_camera", scene_option=head_opt)
    head = np.ascontiguousarray(np.rot90(head_r.render()))
    seg_r.update_scene(data, camera="head_camera", scene_option=head_opt)
    ctrl_boxes = seg_boxes(np.rot90(seg_r.render()))
    if step % TRACK_EVERY == 0:
        dets = detect(head, ctrl_boxes)
        if os.environ.get("DUMP_DETECTIONS"):
            for box, conf in zip(dets.xyxy, dets.confidence):
                dump_rows.append((step, *box, conf))
        tracked = tracker.update(dets)
        tracked = tracked[tracked.tracker_id != -1]  # confirmed tracks only

    if len(tracked):
        frames_with_tracks += 1
        # Image-space speed per track (EMA) — this is what identifies the
        # thrown ball: the one track that is actually MOVING.
        for i, tid_ in enumerate(tracked.tracker_id):
            tid = int(tid_)
            b = tracked.xyxy[i]
            c = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
            if tid not in track_birth:
                track_birth[tid] = t
            if tid in track_center:
                sp = math.hypot(
                    c[0] - track_center[tid][0], c[1] - track_center[tid][1]
                )
                track_speed[tid] = 0.5 * track_speed.get(tid, 0.0) + 0.5 * sp
            track_center[tid] = c

        # Lock: after the release, adopt the fastest track that clears the
        # motion threshold. The duck stands still through the whole throw, so
        # egomotion is negligible and image speed singles out the thrown ball.
        live = {int(i): b for i, b in zip(tracked.tracker_id, tracked.xyxy)}
        head_yaw = float(data.qpos[head_yaw_adr])
        if target_id is None and last_throw_t <= t <= lock_open_until:
            # The thrown ball is either the fastest thing in frame or, when
            # the detector misses it in flight, the newest track to appear
            # once it lands. The distractors have been sitting there under
            # old track ids the whole time.
            # Motion is the only signal precise enough to trust: a "newest
            # track" fallback locks onto detector flicker on the distractors,
            # and fetching the wrong ball is worse than fetching none.
            fast = [
                (sp, tid)
                for tid, sp in track_speed.items()
                if sp >= LOCK_SPEED and tid in live
            ]
            if fast:
                _, target_id = max(fast)

        if target_id in live:
            # The track says which detection is ours; the detection says where
            # it is. A coasting track still reports a box, but it is a Kalman
            # guess, so nothing is ever measured from it.
            box = detection_for(live[target_id], dets)
            if box is not None:
                target_fix = measure(box, head_yaw)
                target_last_seen = t

        is_target = np.array(
            [int(tid) == target_id for tid in tracked.tracker_id], dtype=bool
        )
        tgt, rest = tracked[is_target], tracked[~is_target]
        if len(tgt):
            # Trace only the locked target: full-scene traces just paint the
            # walking head-bob over everything.
            head = trace_ann.annotate(head, tgt)
        if len(rest):
            head = other_box.annotate(head, rest)
            mature = np.array(
                [t - track_birth.get(int(i), t) > 0.6 for i in rest.tracker_id],
                dtype=bool,
            )
            if mature.any():
                head = other_label.annotate(
                    head,
                    rest[mature],
                    labels=[f"#{display_id(i)}" for i in rest[mature].tracker_id],
                )
        if len(tgt):
            head = target_box.annotate(head, tgt)
            head = target_label.annotate(
                head, tgt, labels=["FETCH" for _ in tgt.tracker_id]
            )

    if os.environ.get("DEBUG_LOG"):
        ids = tracked.tracker_id.tolist() if len(tracked) else []
        bq = BALLS[PLAY_JOINT][0]
        spd = {k: round(v, 1) for k, v in track_speed.items() if v > 2}
        print(
            f"BALL t={t:5.2f} pos=({data.qpos[bq]:+.2f},{data.qpos[bq+1]:+.2f},"
            f"{data.qpos[bq+2]:+.2f}) fast={spd}"
        )
        print(
            f"LOG t={t:5.2f} "
            f"mode={policy.behavior_mode or policy.current_policy:9s} "
            f"det={len(dets)} ids={ids} tgt={target_id} "
            f"maxsp={max(track_speed.values(), default=0):.1f}"
        )

    # picture-in-picture: shrunken POV in the top-RIGHT, clear of the owner's
    # hand which enters at the left edge
    if HEADLESS:
        continue
    inset = cv2.resize(head, (INSET_W, INSET_H), interpolation=cv2.INTER_AREA)
    pad = 12
    frame[pad : pad + INSET_H + 4, MAIN_W - INSET_W - pad - 4 : MAIN_W - pad] = 30
    frame[
        pad + 2 : pad + 2 + INSET_H, MAIN_W - INSET_W - pad - 2 : MAIN_W - pad - 2
    ] = inset
    writer.append_data(frame)
    frame_count += 1

if writer is not None:
    writer.close()
    print(f"wrote {out}: {frame_count} frames, tracks visible in {frames_with_tracks}")
if os.environ.get("DUMP_DETECTIONS"):
    cache = os.environ["DUMP_DETECTIONS"]
    if not os.path.isabs(cache):
        cache = os.path.join(CWD, cache)
    np.savez_compressed(
        cache, rows=np.array(dump_rows, dtype=np.float64), n_frames=n_steps
    )
    print(f"wrote {cache}: {len(dump_rows)} detections over {n_steps} frames")
