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
    """Chooses the ball and drives the duck toward it.

    Every frame is one `observe` then one `act`. Observation reduces the
    tracks to a single measurement, `self.fix`, which is the bearing and range
    of the ball the duck has locked. Control is a walk down six phases, in
    priority order, and `act` runs exactly one of them:

        pretrained  a shipped policy owns the body: the peck, or a scripted move
        settling    standing still so the peck starts from a clean stance
        playing     the wiggle after a touch, while the owner comes to collect
        searching   no fix yet: stand and watch for the throw
        closing     the last step, walked on a timer, ending in a peck
        chasing     walk at the ball, steering on the bearing of its box
    """

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
        """Read this frame's tracks, in three steps.

        How is each track moving, which one is our ball, and where is it?
        """
        live = self._track_motion(tracked, t)
        self._lock_thrown_ball(live, t)
        self._take_fix(live, detections, t)

    def _track_motion(self, tracked, t):
        """Image-space speed per track id. Returns the ids visible right now."""
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
        return live

    def _lock_thrown_ball(self, live, t):
        """Adopt the fastest-moving track in the window after a throw.

        Motion is the only signal precise enough to trust here. A "newest
        track" fallback for throws the detector misses in flight locks onto
        detector flicker on the lookalikes instead, and fetching the wrong
        ball is a worse failure than fetching none.
        """
        if self.target_id is not None or not self.release_time <= t <= self.lock_until:
            return
        moving = [
            (speed, tracker_id)
            for tracker_id, speed in self.speeds.items()
            if speed >= LOCK_SPEED and tracker_id in live
        ]
        if moving:
            _, self.target_id = max(moving)

    def _take_fix(self, live, detections, t):
        """Measure bearing and range, but only from a real detection.

        A track with no detection under it is coasting on its Kalman
        prediction. That is fine for keeping the id alive and useless for
        measuring distance, so the last good fix is left to go stale instead.
        """
        if self.target_id not in live:
            return
        box = detection_for(live[self.target_id], detections)
        if box is not None:
            self.fix = measure(box, self.duck.head_yaw, self.width, self.focal)
            self.fix_time = t

    # -- control -----------------------------------------------------------

    def act(self, t):
        """One control decision: work out which phase the duck is in, run it."""
        policy = self.duck.policy
        if policy.behavior_mode is not None or policy.ground_pick_mode:
            return  # a pretrained policy owns the body right now
        if self.settle_until is not None:
            self._settle(t, policy)
        elif t < self.play_until:
            self._play(t, policy)
        elif self.fix is None:
            self._search(policy)
        else:
            self._chase(t, policy)

    def _settle(self, t, policy):
        """Stand still for a beat: the pick policy was trained from a stance."""
        policy.set_vel_cmd(0.0, 0.0, 0.0)
        policy.head_offset[:] = 0.0
        if t >= self.settle_until:
            self.settle_until = None
            policy.trigger_ground_pick()

    def _search(self, policy):
        """Nothing measured yet. Stand, head level, watch for the throw."""
        policy.set_vel_cmd(0.0, 0.0, 0.0)
        policy.head_offset[:] = [0.0, 0.1, 0.0, 0.0]

    def _chase(self, t, policy):
        """There is a fix. Look at the ball, then walk, close, or hold."""
        bearing, distance = self.fix
        policy.head_offset[:] = [  # duck-like gaze, at the locked target only
            0.0,
            float(np.clip(1.2 * (0.55 - distance), 0.0, 0.55)),
            float(np.clip(bearing, -0.5, 0.5)),
            0.0,
        ]
        fresh = t - self.fix_time < STALE_FIX

        if self.grabbed:
            policy.set_vel_cmd(0.0, 0.0, 0.0)  # touched it: wait for the owner
        elif self.closing_until is not None:
            self._close(t, policy)
        elif fresh and distance < GRAB_RANGE and abs(bearing) < 0.35:
            # Close and lined up, so commit to the peck. The beak reaches about
            # 0.15 m and both detectors lose the ball around 0.19 m, so that
            # last gap has to be crossed on a timer rather than on sight. The
            # gait command from the previous frame carries this one.
            self.closing_until = t + CLOSING_STEP
        elif not fresh:
            policy.set_vel_cmd(0.0, 0.0, 0.0)  # coasting track: don't walk on a guess
        else:
            policy.set_vel_cmd(
                lin_vel_x=WALK_SPEED,
                lin_vel_y=0.0,
                ang_vel_z=float(np.clip(TURN_GAIN * bearing, -1.2, 1.2)),
            )

    def _close(self, t, policy):
        """The last few centimetres, walked straight, ending in a peck."""
        policy.set_vel_cmd(lin_vel_x=WALK_SPEED, lin_vel_y=0.0, ang_vel_z=0.0)
        if t >= self.closing_until:
            self.closing_until = None
            self.settle_until = t + SETTLE
            self.grabbed = True
            if self.on_grab:
                self.on_grab(t)

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
