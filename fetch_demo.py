#!/usr/bin/env python3
"""Microduck plays fetch, powered by `trackers`.

The duck stands among IDENTICAL resting balls. An off-screen owner throws
another identical ball through the scene. trackers.SORTTracker keeps an ID on
every ball; the thrown one is singled out purely by TRACK MOTION (its image-
space velocity), and the duck locks that ID and goes to play with exactly that
ball, ignoring the identical distractors. Each new throw re-locks onto the new
moving track. Detection alone cannot express any of this — "the ball that was
just thrown" only exists as a track.

Pretrained Microduck policies (walking/stand/kicks) run in CPU MuJoCo.
Detections: set DETECTOR=rfdetr to run RF-DETR on the rendered head-camera
frames; the default is the simulator's segmentation renderer (a perfect-
detector stand-in). Control-side ball identification always uses the
segmentation boxes, so the detector choice only changes what the tracker sees.
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
# See README.md: clone pollen-robotics/microduck_rl next to this file and
# download the ONNX policies from pollen-robotics/microduck-policies (HF Hub).
RL = os.environ.get("MICRODUCK_RL", os.path.join(HERE, "microduck_rl"))
POLICIES = os.environ.get("MICRODUCK_POLICIES", os.path.join(HERE, "policies"))
sys.path.insert(0, os.path.join(RL, "scripts"))
os.chdir(RL)  # infer_policy uses repo-relative XML paths

from infer_policy import (  # noqa: E402
    MICRODUCK_BALL_XML,
    PolicyInference,
)

SIM_SECONDS = float(os.environ.get("SIM_SECONDS", 28))
FIRST_THROW = 2.5                # the owner's first throw
MAX_THROWS = 2                   # throws in the video
RETHROW_DELAY = 2.0              # owner waits for the duck to finish its kick
LOCK_WINDOW = 4.0                # seconds after a throw to lock the fast track
LOCK_SPEED = 6.0                 # px/frame EMA speed that means "thrown"
TRACK_EVERY = int(os.environ.get("TRACK_EVERY", 1))  # 3 = ~16.7 Hz robot cadence
MAIN_W, MAIN_H = 1280, 720       # chase view canvas
PANEL_W, PANEL_H = 960, 540      # duck POV render size (annotated, then shrunk)
INSET_W, INSET_H = 480, 270      # POV picture-in-picture, top-left corner

spec = mujoco.MjSpec.from_file(MICRODUCK_BALL_XML)
spec.visual.global_.offwidth = 1280
spec.visual.global_.offheight = 1280  # head cam renders portrait, needs height
# The MJCF head_camera points backward with a 90-degree roll; face it forward,
# pitch it 25 degrees down toward the ground the robot acts on, widen to ~90.
for c in spec.cameras:
    if c.name == "head_camera":
        th = math.radians(25)
        c.quat = [math.cos(th / 2), 0, math.sin(th / 2), 0]
        c.fovy = 90
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
    new_cmd_obs=True,
    use_projected_gravity=True,
    kick_duration=2.0,
)

# The kick trigger snaps the ball to the trained kick spot ("_place_ball") —
# a visible teleport. We gate kicks tightly around that spot instead and let
# the policy play the ball where it actually lies.
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

GEOM_TO_JOINT = {}
BALL_GEOM_IDS = []
BALL_GEOMS = [("ball_geom", PLAY_JOINT)] + [
    (f"ball{k}_geom", f"ball{k}_free") for k in range(2, 2 + len(DISTRACTOR_POS))
]
for gname, jname in BALL_GEOMS:
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, gname)
    BALL_GEOM_IDS.append(gid)
    GEOM_TO_JOINT[gid] = jname

# Renderers: chase view, head RGB, head segmentation
main_r = mujoco.Renderer(model, height=MAIN_H, width=MAIN_W)
# Head cam renders portrait and is rotated 90 degrees to landscape (the
# mounted camera frame is rolled relative to the image we want).
head_r = mujoco.Renderer(model, height=PANEL_W, width=PANEL_H)
seg_r = mujoco.Renderer(model, height=PANEL_W, width=PANEL_H)
seg_r.enable_segmentation_rendering()
# Hide the soft-jaw meshes (group 2) that sit in front of the head camera.
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
GRAY = sv.Color(200, 200, 200)
target_box = sv.BoxAnnotator(thickness=5, color=MAGENTA)
target_label = sv.LabelAnnotator(text_scale=0.9, text_thickness=2, color=MAGENTA)
other_box = sv.BoxAnnotator(thickness=3, color=GRAY)
other_label = sv.LabelAnnotator(text_scale=0.8, text_thickness=2, color=GRAY)
trace_ann = sv.TraceAnnotator(thickness=4, trace_length=15, color=MAGENTA)


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


def ball_relative(joint):
    """Ball center in the trunk yaw frame: (forward, left)."""
    trunk_xy, yaw = trunk_yaw_frame()
    qadr = BALLS[joint][0]
    d = data.qpos[qadr : qadr + 2] - trunk_xy
    c, s = math.cos(-yaw), math.sin(-yaw)
    return c * d[0] - s * d[1], s * d[0] + c * d[1]


HAND_WINDUP = 0.55   # hand sweeps forward carrying the ball
HAND_RETRACT = 0.6   # hand pulls back after release
hand_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "owner_hand")
hand_mid = int(model.body_mocapid[hand_bid])
throw_anim = None  # dict while the hand is animating


def start_throw(t):
    """Owner's throw: the visible hand sweeps in from behind the duck's left
    shoulder, carries the ball forward, and releases it gently into the field."""
    trunk_xy, yaw = trunk_yaw_frame()
    f = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    r = np.array([f[1], -f[0], 0.0])  # the duck's right
    base = np.array([trunk_xy[0], trunk_xy[1], 0.0])
    return {
        "t0": t,
        "start": base - 0.72 * f - 0.30 * r + [0, 0, 0.34],
        "release": base - 0.38 * f - 0.16 * r + [0, 0, 0.46],
        "vel": 1.5 * f + 0.28 * r + [0, 0, 1.3],
        "quat": [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)],
        "released": False,
    }


def animate_hand(anim, t):
    """Advance the hand animation; returns False when finished."""
    qadr, vadr = BALLS[PLAY_JOINT]
    dt_ = t - anim["t0"]
    if dt_ <= HAND_WINDUP:
        s_ = dt_ / HAND_WINDUP
        s_ = s_ * s_ * (3 - 2 * s_)  # smoothstep
        pos = anim["start"] + s_ * (anim["release"] - anim["start"])
        data.mocap_pos[hand_mid] = pos
        data.mocap_quat[hand_mid] = anim["quat"]
        # Ball rides on the palm until release.
        data.qpos[qadr : qadr + 3] = pos + [0, 0, 0.045]
        data.qpos[qadr + 3 : qadr + 7] = [1, 0, 0, 0]
        data.qvel[vadr : vadr + 6] = 0.0
        return True
    if dt_ <= HAND_WINDUP + HAND_RETRACT:
        if not anim["released"]:
            anim["released"] = True
            data.qvel[vadr : vadr + 6] = 0.0
            data.qvel[vadr : vadr + 3] = anim["vel"]
        s_ = (dt_ - HAND_WINDUP) / HAND_RETRACT
        data.mocap_pos[hand_mid] = anim["release"] + s_ * (
            anim["start"] - anim["release"] + [0, 0, 0.15]
        )
        return True
    data.mocap_pos[hand_mid] = [6.0, 6.0, 0.5]  # park out of sight
    return False


def seg_boxes(seg):
    """Segmentation frame -> (list of xyxy boxes, joint name per box)."""
    boxes, joints = [], []
    for gid in BALL_GEOM_IDS:
        mask = (seg[..., 0] == gid) & (
            seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
        )
        ys, xs = np.nonzero(mask)
        if len(xs) < 3:
            continue
        boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
        joints.append(GEOM_TO_JOINT[gid])
    return boxes, joints


def detect(head_rgb, boxes, joints):
    """Detections the tracker consumes.

    segmentation mode: the oracle boxes themselves.
    rfdetr mode: RF-DETR on the rendered frame ("sports ball" class); the
    oracle boxes are used for nothing but control-side identification.
    """
    if rfdetr_model is not None:
        dets = rfdetr_model.predict(head_rgb, threshold=0.3)
        keep = dets.class_id == 37  # COCO "sports ball"
        return dets[keep]
    if not boxes:
        return sv.Detections.empty()
    n = len(boxes)
    return sv.Detections(
        xyxy=np.array(boxes, dtype=float),
        confidence=np.ones(n),
        class_id=np.zeros(n, dtype=int),
    )


def match_tracks_to_joints(tracked, det_boxes, det_joints):
    """Per-frame map tracker_id -> ball joint, by box center proximity."""
    out = {}
    for i, tid in enumerate(tracked.tracker_id):
        tb = tracked.xyxy[i]
        tc = ((tb[0] + tb[2]) / 2, (tb[1] + tb[3]) / 2)
        best, best_d = None, 1e9
        for b, j in zip(det_boxes, det_joints):
            c = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
            d = (tc[0] - c[0]) ** 2 + (tc[1] - c[1]) ** 2
            if d < best_d:
                best, best_d = j, d
        out[int(tid)] = best
    return out


control_dt = 4 * model.opt.timestep
n_steps = int(SIM_SECONDS / control_dt)
frames = []
kick_cooldown = 0.0
frames_with_tracks = 0
target_id = None            # locked tracker ID (the thrown ball's track)
target_joint = None         # ball joint the locked ID maps to
id_to_joint: dict[int, str] = {}
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
recover_until = -1.0        # stand still after a pick to absorb the landing
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
        last_throw_t = t
        next_throw_at = t + 18.0  # timeout fallback if the fetch never kicks
        grabbed_this_throw = False
        lock_open_until = t + LOCK_WINDOW + HAND_WINDUP
        target_id = None    # wait for the new moving track
        target_joint = None
    if throw_anim is not None and not animate_hand(throw_anim, t):
        throw_anim = None

    if prev_behavior in ("kick_left", "kick_right") and policy.behavior_mode is None:
        next_throw_at = t + RETHROW_DELAY  # kick finished: owner throws again
    prev_behavior = policy.behavior_mode
    if throws_done and next_throw_at < t - 0.1 and throws_done < MAX_THROWS:
        pass  # (fallback scheduling handled by the kick-end trigger above)
    policy.update_ground_pick_phase(control_dt)
    if prev_pick_mode and not policy.ground_pick_mode:
        recover_until = t + 1.5  # pick just ended; let the stand policy settle
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
            policy.set_vel_cmd(0.0, 0.0, 0.0)
            policy.head_offset[:] = 0.0
        elif target_joint is None:
            # Waiting for the owner: stand, head level, watch the field.
            policy.set_vel_cmd(0.0, 0.0, 0.0)
            policy.head_offset[:] = [0.0, 0.1, 0.0, 0.0]
        else:
            fwd, left = ball_relative(target_joint)
            err = math.atan2(left, fwd)
            dist = math.hypot(fwd, left)
            # Duck-like gaze toward the LOCKED target only.
            policy.head_offset[:] = [
                0.0,
                float(np.clip(1.2 * (0.55 - dist), 0.0, 0.55)),
                float(np.clip(err, -0.5, 0.5)),
                0.0,
            ]
            at_ball = 0.04 < fwd < 0.15 and abs(left) < 0.10
            if os.environ.get("DEBUG_LOG") and math.hypot(fwd, left) < 0.35:
                print(f"NEAR t={t:5.2f} fwd={fwd:+.3f} left={left:+.3f} "
                      f"cool={kick_cooldown:.1f} grabbed={grabbed_this_throw}")
            if at_ball and kick_cooldown == 0.0:
                if not grabbed_this_throw:
                    # First reach: duck down and worry the ball with the beak,
                    # the way a dog gets its mouth on a fresh ball.
                    grab_settle_until = t + 0.7
                    grabbed_this_throw = True
                else:
                    policy.trigger_behavior(
                        "kick_left" if left > 0 else "kick_right"
                    )
                kick_cooldown = 4.0
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
    az_t = math.degrees(yaw_now) + 18.0
    chase_az += 0.03 * ((az_t - chase_az + 180.0) % 360.0 - 180.0)
    chase.azimuth = chase_az
    main_r.update_scene(data, camera=chase)
    frame = main_r.render()
    head_r.update_scene(data, camera="head_camera", scene_option=head_opt)
    head = np.ascontiguousarray(np.rot90(head_r.render()))
    seg_r.update_scene(data, camera="head_camera", scene_option=head_opt)
    ctrl_boxes, ctrl_joints = seg_boxes(np.rot90(seg_r.render()))
    if step % TRACK_EVERY == 0:
        dets = detect(head, ctrl_boxes, ctrl_joints)
        tracked = tracker.update(dets)
        tracked = tracked[tracked.tracker_id != -1]  # confirmed tracks only

    if len(tracked):
        frames_with_tracks += 1
        id_to_joint = match_tracks_to_joints(tracked, ctrl_boxes, ctrl_joints)

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
                track_speed[tid] = 0.7 * track_speed.get(tid, 0.0) + 0.3 * sp
            track_center[tid] = c

        # Lock: within the window after a throw, adopt the fastest track that
        # clears the motion threshold. No ground truth involved.
        if target_id is None and t <= lock_open_until:
            # The thrown ball is the track that is BOTH fast and newborn:
            # egomotion can make old (distractor) tracks look fast, but only
            # the throw creates a brand-new fast track.
            fast = [
                (sp, tid)
                for tid, sp in track_speed.items()
                if sp >= LOCK_SPEED
                and tid in id_to_joint
                and track_birth.get(tid, -1.0) >= last_throw_t
            ]
            if fast:
                _, target_id = max(fast)
                target_joint = id_to_joint[target_id]
        elif target_joint is not None:
            # Keep the lock attached across ID churn on the same ball.
            joint_ids = [i for i, j in id_to_joint.items() if j == target_joint]
            if target_id not in joint_ids and joint_ids:
                target_id = joint_ids[0]

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
            head = other_label.annotate(
                head, rest, labels=[f"ball #{i}" for i in rest.tracker_id]
            )
        if len(tgt):
            head = target_box.annotate(head, tgt)
            head = target_label.annotate(
                head, tgt, labels=[f"FETCH ball #{i}" for i in tgt.tracker_id]
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
            f"det={len(dets)} ids={ids} tgt={target_id}/{target_joint}"
        )

    # picture-in-picture: shrunken POV in the top-left (empty sky), thin border
    inset = cv2.resize(head, (INSET_W, INSET_H), interpolation=cv2.INTER_AREA)
    pad = 12
    frame[pad : pad + INSET_H + 4, pad : pad + INSET_W + 4] = 30
    frame[pad + 2 : pad + 2 + INSET_H, pad + 2 : pad + 2 + INSET_W] = inset
    frames.append(frame)

out = os.path.join(HERE, "fetch_demo.mp4")
imageio.mimwrite(out, frames, fps=50, quality=8)
print(f"wrote {out}: {len(frames)} frames, tracks visible in {frames_with_tracks}")
