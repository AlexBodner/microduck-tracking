"""Load the Microduck: its scene, its balls, and its pretrained policies.

Everything here is setup. The simulator, the robot model and the walking,
standing and ground-pick policies all come from pollen-robotics/microduck_rl
unchanged; this module only dresses the scene as a park, adds the balls and
the owner's hand, and aims the head camera.
"""

import math
import os
import sys

import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
# See README.md: clone pollen-robotics/microduck_rl in the repo root and
# download the ONNX policies from pollen-robotics/microduck-policies (HF Hub).
RL = os.environ.get("MICRODUCK_RL", os.path.join(ROOT, "microduck_rl"))
POLICIES = os.environ.get("MICRODUCK_POLICIES", os.path.join(ROOT, "policies"))
CWD = os.getcwd()  # output paths resolve against the caller, not RL
sys.path.insert(0, os.path.join(RL, "scripts"))
os.chdir(RL)  # infer_policy uses repo-relative XML paths

from infer_policy import MICRODUCK_BALL_XML, PolicyInference  # noqa: E402

BALL_DIAMETER = 0.07     # the play ball, and every lookalike
CAMERA_PITCH = 25        # degrees down from the head frame
CAMERA_FOV = 90          # degrees, vertical before the view is rotated
PLAY_JOINT = "ball_free"
TREES = [
    (-2.5, 1.8, 1.0),
    (3.2, 2.4, 1.3),
    (2.0, -3.0, 0.9),
    (-3.0, -2.2, 1.15),
    (4.0, -0.5, 1.05),
]
SKIN = [0.93, 0.77, 0.64, 1.0]


def _dress_as_park(spec):
    """Mowed-lawn greens instead of the blue checker, hazier sky, some trees."""
    for texture in spec.textures:
        if texture.name == "groundplane":
            texture.rgb1 = [0.40, 0.56, 0.24]
            texture.rgb2 = [0.33, 0.49, 0.20]
            texture.mark = mujoco.mjtMark.mjMARK_NONE
        elif texture.type == mujoco.mjtTexture.mjTEXTURE_SKYBOX:
            texture.rgb1 = [0.46, 0.70, 0.93]
            texture.rgb2 = [0.88, 0.94, 1.0]
    for material in spec.materials:
        if material.name == "groundplane":
            material.reflectance = 0.0
    for k, (x, y, scale) in enumerate(TREES):
        tree = spec.worldbody.add_body(name=f"tree{k}", pos=[x, y, 0])
        tree.add_geom(
            name=f"trunk{k}", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[0.06 * scale, 0.35 * scale, 0], pos=[0, 0, 0.35 * scale],
            rgba=[0.42, 0.30, 0.20, 1], contype=0, conaffinity=0,
        )
        tree.add_geom(
            name=f"canopy{k}", type=mujoco.mjtGeom.mjGEOM_ELLIPSOID,
            size=[0.42 * scale, 0.42 * scale, 0.5 * scale],
            pos=[0, 0, 0.95 * scale],
            rgba=[0.24 + 0.05 * (k % 3), 0.5 + 0.04 * (k % 2), 0.2, 1],
            contype=0, conaffinity=0,
        )


def _aim_head_camera(spec):
    """The MJCF camera points backward with a 90 degree roll. Face it forward
    and pitch it at the ground the robot acts on.

    The lens stays where Pollen mounts it, low and behind the beak. That puts
    it on the beak's arc, so during a peck it passes through the ball and the
    view goes briefly empty; moving it forward would avoid that but would stop
    being this robot's camera.
    """
    for camera in spec.cameras:
        if camera.name == "head_camera":
            half = math.radians(CAMERA_PITCH) / 2
            camera.quat = [math.cos(half), 0, math.sin(half), 0]
            camera.fovy = CAMERA_FOV


def _add_balls(spec, distractors):
    """Rolling friction so thrown balls settle within about a metre (condim=6
    activates it; the default condim=3 ignores the coefficient), plus the
    identical lookalikes the duck has to tell its own ball apart from."""
    for geom in spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM):
        if geom.name == "ball_geom":
            geom.condim = 6
            geom.friction = [0.5, 0.005, 0.004]
    for k, (x, y) in enumerate(distractors, start=2):
        body = spec.worldbody.add_body(name=f"ball{k}", pos=[x, y, 0.035])
        body.add_freejoint(name=f"ball{k}_free")
        body.add_geom(
            name=f"ball{k}_geom", type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[BALL_DIAMETER / 2, 0, 0], rgba=[1, 0.55, 0, 1],
            condim=6, friction=[0.5, 0.005, 0.004], mass=0.015,
        )


def _add_owner_hand(spec):
    """A kinematic (mocap) hand that carries and throws the ball."""
    hand = spec.worldbody.add_body(
        name="owner_hand", pos=[6.0, 6.0, 0.5], mocap=True
    )
    hand.add_geom(
        name="palm", type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.050, 0.042, 0.011], rgba=SKIN, contype=0, conaffinity=0,
    )
    for k, y in enumerate((-0.030, -0.010, 0.010, 0.030)):
        hand.add_geom(
            name=f"finger{k}", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            fromto=[0.05, y, 0.004, 0.105, y * 1.5, 0.018],
            size=[0.0085, 0, 0], rgba=SKIN, contype=0, conaffinity=0,
        )
    hand.add_geom(
        name="thumb", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        fromto=[0.01, -0.042, 0.004, 0.045, -0.075, 0.022],
        size=[0.009, 0, 0], rgba=SKIN, contype=0, conaffinity=0,
    )


class Microduck:
    """The robot, its scene and its policies, ready to step."""

    def __init__(self, distractors, render_size=1280):
        spec = mujoco.MjSpec.from_file(MICRODUCK_BALL_XML)
        spec.visual.global_.offwidth = render_size
        spec.visual.global_.offheight = render_size  # head cam renders portrait
        _dress_as_park(spec)
        _aim_head_camera(spec)
        _add_balls(spec, distractors)
        _add_owner_hand(spec)
        self.model = spec.compile()
        self.model.opt.timestep = 0.005
        self.data = mujoco.MjData(self.model)

        self.policy = PolicyInference(
            self.model, self.data,
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
        # Triggering a kick snaps the ball to the trained kick spot, a visible
        # teleport. We gate the pick tightly instead and let the policy play
        # the ball where it lies. Private method of microduck_rl, verified
        # against upstream d424a0c.
        self.policy._place_ball = lambda behavior: None

        self._trunk = self.policy._trunk_qpos_adr
        self.data.qpos[self._trunk + 0 : self._trunk + 3] = [0.0, 0.0, 0.125]
        self.data.qpos[self._trunk + 3 : self._trunk + 7] = [1, 0, 0, 0]
        for i, qi in enumerate(self.policy.joint_qpos_indices):
            self.data.qpos[qi] = self.policy.default_pose[i]
        self.data.ctrl[:] = self.policy.default_pose

        names = [PLAY_JOINT] + [f"ball{k}_free" for k in range(2, 2 + len(distractors))]
        self.balls = {}
        for name in names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self.balls[name] = (
                int(self.model.jnt_qposadr[jid]),
                int(self.model.jnt_dofadr[jid]),
            )
        # The play ball waits out of sight until the owner's first throw.
        play = self.balls[PLAY_JOINT][0]
        self.data.qpos[play : play + 3] = [6.0, 6.0, 0.035]
        mujoco.mj_forward(self.model, self.data)

        self.ball_geom_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in ["ball_geom"]
            + [f"ball{k}_geom" for k in range(2, 2 + len(distractors))]
        ]
        yaw_jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "head_yaw")
        self._head_yaw = int(self.model.jnt_qposadr[yaw_jid])
        hand_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "owner_hand")
        self.hand_mocap = int(self.model.body_mocapid[hand_bid])
        self.control_dt = 4 * self.model.opt.timestep

    def trunk_frame(self):
        """Trunk position and heading, for placing the owner's throw."""
        adr = self._trunk
        xy = self.data.qpos[adr : adr + 2].copy()
        qw, qx, qy, qz = self.data.qpos[adr + 3 : adr + 7]
        yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        return xy, yaw

    @property
    def head_yaw(self):
        """The head yaw joint, an encoder reading rather than simulator state."""
        return float(self.data.qpos[self._head_yaw])

    def ball_position(self, joint=PLAY_JOINT):
        adr = self.balls[joint][0]
        return self.data.qpos[adr : adr + 3].copy()

    def step(self):
        """One control step: run the policy and advance the physics."""
        self.policy.apply_action(self.policy.infer())
        for _ in range(4):
            mujoco.mj_step(self.model, self.data)
