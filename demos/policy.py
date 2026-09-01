"""The fetch policy: pick a ball out of the tracks, then walk to it.

Two decisions, both made from the tracker's output alone.

Which ball: after a throw, the one track that is actually moving. Every
lookalike has been sitting still under an old track id.

Where to walk: bearing and range measured from that track's detection, into a
proportional heading controller at a fixed forward speed.
"""

import math

import numpy as np

from scene.microduck import BALL_DIAMETER, CAMERA_FOV


def focal_px(width):
    """Pixels per radian at the image centre, from the camera's own field of
    view. The head view renders portrait and is rotated, so the camera's
    vertical fovy spans the width of the image the tracker sees."""
    return (width / 2) / math.tan(math.radians(CAMERA_FOV) / 2)


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


def measure(box, head_yaw, width, focal):
    """Bearing and range to a ball, from its box alone.

    A sphere of known diameter gives range from its apparent size. The box
    centre gives bearing in the head frame, and adding the head yaw joint puts
    that bearing back in the body frame.
    """
    centre_x = (box[0] + box[2]) / 2
    diameter_px = max(box[2] - box[0], box[3] - box[1])
    bearing = -math.atan2(centre_x - width / 2, focal) + head_yaw
    return bearing, focal * BALL_DIAMETER / max(diameter_px, 1.0)


LOCK_WINDOW = 8.0     # seconds after a release in which to lock a moving track
LOCK_SPEED = 4.0      # px per tracked frame that counts as "just thrown"
GRAB_RANGE = 0.20     # range at which both detectors still resolve the ball
CLOSING_STEP = 0.55   # seconds of straight walk between seeing it and pecking
STALE_FIX = 0.4       # seconds without a detection before standing still
WALK_SPEED = 0.3      # the gait only starts near the policy's max command
TURN_GAIN = 2.0
SETTLE = 0.3          # the pick policy was trained from a standing start
PLAY_WINDOW = 1.9     # the little dance after a touch, while the owner comes in


class FetchPolicy:
    """Chooses the ball and drives the duck toward it."""

    def __init__(self, duck, width, focal):
        self.duck = duck
        self.width = width
        self.focal = focal
        self.target_id = None
        self.fix = None              # (bearing, range) from the last detection
        self.fix_time = -1.0
        self.speeds = {}             # tracker id -> EMA image-space speed
        self.centres = {}
        self.births = {}
        self.release_time = -1.0
        self.lock_until = -1.0
        self.grabbed = False
        self.settle_until = None
        self.closing_until = None
        self.play_until = -1.0
        self.play_start = 0.0
        self._was_picking = False
        self.on_grab = None          # optional callbacks, for trial scoring
        self.on_touch = None

    # -- events ------------------------------------------------------------

    def on_throw(self, release_time):
        """A throw is under way: forget the old ball and wait for the new one."""
        self.release_time = release_time
        self.lock_until = release_time + LOCK_WINDOW
        self.target_id = None
        self.fix = None
        self.grabbed = False
        self.closing_until = None

    def note_pick(self, t):
        """Watch for the ground pick finishing, and start the little dance."""
        picking = self.duck.policy.ground_pick_mode
        finished = self._was_picking and not picking
        self._was_picking = picking
        if finished:
            if self.on_touch:
                self.on_touch(t)
            self.play_until = t + PLAY_WINDOW
            self.play_start = t
        return finished

    @property
    def pecking(self):
        return self.duck.policy.ground_pick_mode or self.settle_until is not None

    # -- perception --------------------------------------------------------

    def observe(self, tracked, detections, t):
        """Update track motion, lock onto the thrown ball, and take a fix."""
        live = {}
        for i, tracker_id in enumerate(tracked.tracker_id):
            tracker_id = int(tracker_id)
            box = tracked.xyxy[i]
            live[tracker_id] = box
            centre = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
            self.births.setdefault(tracker_id, t)
            if tracker_id in self.centres:
                moved = math.dist(centre, self.centres[tracker_id])
                self.speeds[tracker_id] = (
                    0.5 * self.speeds.get(tracker_id, 0.0) + 0.5 * moved
                )
            self.centres[tracker_id] = centre

        # Motion is the only signal precise enough to trust here. A "newest
        # track" fallback for throws the detector misses in flight locks onto
        # detector flicker on the lookalikes instead, and fetching the wrong
        # ball is a worse failure than fetching none.
        if self.target_id is None and self.release_time <= t <= self.lock_until:
            moving = [
                (speed, tracker_id)
                for tracker_id, speed in self.speeds.items()
                if speed >= LOCK_SPEED and tracker_id in live
            ]
            if moving:
                _, self.target_id = max(moving)

        if self.target_id in live:
            box = detection_for(live[self.target_id], detections)
            if box is not None:
                self.fix = measure(box, self.duck.head_yaw, self.width, self.focal)
                self.fix_time = t

    # -- control -----------------------------------------------------------

    def act(self, t):
        """One control decision. Sets the duck's velocity and gaze commands."""
        policy = self.duck.policy
        if policy.behavior_mode is not None or policy.ground_pick_mode:
            return

        if self.settle_until is not None:
            policy.set_vel_cmd(0.0, 0.0, 0.0)
            policy.head_offset[:] = 0.0
            if t >= self.settle_until:
                self.settle_until = None
                policy.trigger_ground_pick()
            return

        if t < self.play_until:
            self._play(t, policy)
            return

        if self.fix is None:
            # Waiting for the owner: stand, head level, watch the field.
            policy.set_vel_cmd(0.0, 0.0, 0.0)
            policy.head_offset[:] = [0.0, 0.1, 0.0, 0.0]
            return

        bearing, distance = self.fix
        # Duck-like gaze, toward the locked target only.
        policy.head_offset[:] = [
            0.0,
            float(np.clip(1.2 * (0.55 - distance), 0.0, 0.55)),
            float(np.clip(bearing, -0.5, 0.5)),
            0.0,
        ]
        fresh = t - self.fix_time < STALE_FIX
        lined_up = abs(bearing) < 0.35

        if self.grabbed:
            # Touch done: stand over the ball and wait for the owner.
            policy.set_vel_cmd(0.0, 0.0, 0.0)
        elif self.closing_until is not None:
            # The last few centimetres, walked straight. Started from a
            # detection rather than a guess.
            policy.set_vel_cmd(lin_vel_x=WALK_SPEED, lin_vel_y=0.0, ang_vel_z=0.0)
            if t >= self.closing_until:
                self.closing_until = None
                self.settle_until = t + SETTLE
                self.grabbed = True
                if self.on_grab:
                    self.on_grab(t)
        elif fresh and distance < GRAB_RANGE and lined_up:
            # Seen, close and lined up. The beak reaches about 0.15 m and both
            # detectors lose the ball around 0.19 m, so close that gap with one
            # fixed step rather than stopping here and pecking short.
            self.closing_until = t + CLOSING_STEP
        elif not fresh:
            # No detection under the target: stand and look, rather than walk
            # on a guess.
            policy.set_vel_cmd(0.0, 0.0, 0.0)
        else:
            policy.set_vel_cmd(
                lin_vel_x=WALK_SPEED,
                lin_vel_y=0.0,
                ang_vel_z=float(np.clip(TURN_GAIN * bearing, -1.2, 1.2)),
            )

    def _play(self, t, policy):
        """After a touch: admire the ball, then an excited wiggle. A dog asking
        for the next throw."""
        if not self.grabbed:
            policy.set_vel_cmd(0.0, 0.0, 0.0)
            policy.head_offset[:] = 0.0
            return
        phase = t - self.play_start
        if phase < 0.5:
            policy.set_vel_cmd(0.0, 0.0, 0.0)
            policy.head_offset[:] = [0.0, 0.45, 0.0, 0.0]
        elif phase < 1.2:
            policy.set_vel_cmd(0.0, 0.0, 0.9)
            policy.head_offset[:] = [0.0, 0.2, 0.35, 0.0]
        else:
            policy.set_vel_cmd(0.0, 0.0, -0.9)
            policy.head_offset[:] = [0.0, 0.1, -0.2, 0.0]
