"""The owner: a hand that picks the ball up and throws it again.

This is the environment, not the robot. It reads the ball's position from the
simulator because a person can see where the ball is; nothing here is
available to the duck.
"""

import math

import numpy as np

from .microduck import PLAY_JOINT

ENTER = 0.30    # first throw: the hand glides in already holding the ball
REACH = 0.35    # down to the resting ball
CARRY = 0.40    # back to the hold point
THROW = 0.30    # the underhand scoop
RETRACT = 0.30  # back out of frame


def _smooth(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def _bezier(p0, p1, p2, u):
    return (1 - u) ** 2 * p0 + 2 * u * (1 - u) * p1 + u**2 * p2


def _hand_quat(yaw, pitch):
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    # yaw about z composed with pitch about the hand's y (palm tilt)
    return [cy * cp, -sy * sp, cp * 0 + sp * cy, sy * cp]


class Owner:
    """Throws the play ball, and retrieves it between throws."""

    def __init__(self, duck, seed=0):
        self.duck = duck
        self.rng = np.random.default_rng(seed)
        self.throw = None  # live throw state, or None between throws

    @property
    def busy(self):
        return self.throw is not None

    def start_throw(self, t):
        """Stage a throw from wherever the ball currently lies."""
        trunk_xy, yaw = self.duck.trunk_frame()
        forward = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        right = np.array([forward[1], -forward[0], 0.0])
        base = np.array([trunk_xy[0], trunk_xy[1], 0.0])
        ball = self.duck.ball_position()
        # First throw of the run: the ball is parked out of sight, so there is
        # nothing to pick up and the hand arrives already holding it.
        first = ball[2] < 0 or np.linalg.norm(ball[:2] - base[:2]) > 3.0
        hold = base - 0.72 * forward - 0.30 * right + [0, 0, 0.40]
        self.throw = {
            "t0": t,
            "yaw": yaw,
            "first": first,
            "enter": hold - 0.40 * forward + [0, 0, 0.06],
            "ball0": ball,
            "hold": hold,
            "release": base - 0.30 * forward - 0.14 * right + [0, 0, 0.50],
            "vel": (
                (1.5 + self.rng.uniform(-0.30, 0.45)) * forward
                + (0.28 + self.rng.uniform(-0.45, 0.45)) * right
                + [0, 0, 1.3 + self.rng.uniform(-0.15, 0.25)]
            ),
            "released": False,
        }
        return t + self.release_delay(first)

    @staticmethod
    def release_delay(first):
        """How long after the throw starts the ball actually leaves the hand."""
        return (ENTER if first else REACH + CARRY) + THROW

    def _carry_ball(self, position):
        qadr, vadr = self.duck.balls[PLAY_JOINT]
        # An offset, not a concatenation: position is a vector, and this
        # lifts the ball to sit in the hand rather than inside it.
        self.duck.data.qpos[qadr : qadr + 3] = position + np.array([0, 0, 0.032])
        self.duck.data.qpos[qadr + 3 : qadr + 7] = [1, 0, 0, 0]
        self.duck.data.qvel[vadr : vadr + 6] = 0.0

    def update(self, t):
        """Advance the hand. Clears the throw once the hand is out of frame."""
        if self.throw is None:
            return
        anim, data, mocap = self.throw, self.duck.data, self.duck.hand_mocap
        elapsed = t - anim["t0"]
        yaw, hold, release = anim["yaw"], anim["hold"], anim["release"]
        reach_end = ENTER if anim["first"] else REACH + CARRY

        if anim["first"] and elapsed <= ENTER:
            u = _smooth(elapsed / ENTER)
            position = anim["enter"] + u * (hold - anim["enter"])
            data.mocap_pos[mocap] = position
            data.mocap_quat[mocap] = _hand_quat(yaw, 0.0)
            self._carry_ball(position)
            return
        if not anim["first"] and elapsed <= REACH:
            u = _smooth(elapsed / REACH)
            target = anim["ball0"] + [0, 0, 0.038]
            data.mocap_pos[mocap] = anim["enter"] + u * (target - anim["enter"])
            data.mocap_quat[mocap] = _hand_quat(yaw, 0.35 * u)
            return
        if not anim["first"] and elapsed <= reach_end:
            u = _smooth((elapsed - REACH) / CARRY)
            start = anim["ball0"] + [0, 0, 0.055]
            arc = 0.5 * (start + hold) + [0, 0, 0.22]
            position = _bezier(start, arc, hold, u)
            data.mocap_pos[mocap] = position
            data.mocap_quat[mocap] = _hand_quat(yaw, 0.35 * (1 - u))
            self._carry_ball(position)
            return
        if elapsed <= reach_end + THROW:
            # Underhand scoop: dip below the line then sweep up to the release,
            # wrist rolling from pulled back to followed through.
            u = _smooth((elapsed - reach_end) / THROW)
            dip = 0.5 * (hold + release) - [0, 0, 0.16]
            position = _bezier(hold, dip, release, u)
            data.mocap_pos[mocap] = position
            data.mocap_quat[mocap] = _hand_quat(yaw, -0.45 + 0.8 * u)
            self._carry_ball(position)
            return
        if elapsed <= reach_end + THROW + RETRACT:
            if not anim["released"]:
                anim["released"] = True
                _, vadr = self.duck.balls[PLAY_JOINT]
                data.qvel[vadr : vadr + 6] = 0.0
                data.qvel[vadr : vadr + 3] = anim["vel"]
            u = _smooth((elapsed - reach_end - THROW) / RETRACT)
            data.mocap_pos[mocap] = release + u * (anim["enter"] - release)
            data.mocap_quat[mocap] = _hand_quat(yaw, 0.35 * (1 - u))
            return
        data.mocap_pos[mocap] = [6.0, 6.0, 0.5]  # park out of sight
        self.throw = None
