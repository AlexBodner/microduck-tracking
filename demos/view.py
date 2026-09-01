"""Rendering: the chase camera, the duck's-eye view, and the annotations.

None of this feeds the controller. It exists so a viewer can see what the duck
is tracking and decide whether to believe it.
"""

import math

import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np
import supervision as sv

MAIN_W, MAIN_H = 1280, 720    # chase view canvas
PANEL_W, PANEL_H = 960, 540   # duck POV, annotated at this size then shrunk
INSET_W, INSET_H = 480, 270   # POV picture-in-picture
BEHIND = 18.0                 # degrees off the duck's heading, over the shoulder
BESIDE = 75.0                 # swung round while it works the ball with its beak
MAGENTA = sv.Color(255, 64, 255)
GRAY = sv.Color(235, 235, 235)
LABEL_AFTER = 0.6             # seconds before a track is worth labelling


class View:
    """Renders the chase view, the head camera, and the composed frame."""

    def __init__(self, duck, path, headless=False, fps=50):
        self.duck = duck
        self.chase_renderer = mujoco.Renderer(duck.model, height=MAIN_H, width=MAIN_W)
        # The head cam renders portrait: its mount is rolled 90 degrees
        # relative to the image we want, so every frame is rotated.
        self.head_renderer = mujoco.Renderer(duck.model, height=PANEL_W, width=PANEL_H)
        self.seg_renderer = mujoco.Renderer(duck.model, height=PANEL_W, width=PANEL_H)
        self.seg_renderer.enable_segmentation_rendering()
        # Hide the soft-jaw meshes that sit in front of the lens.
        self.head_options = mujoco.MjvOption()
        self.head_options.geomgroup[2] = 0

        self.chase = mujoco.MjvCamera()
        self.chase.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.chase.trackbodyid = duck.policy.trunk_base_id
        self.chase.distance = 1.25
        self.chase.elevation = -22
        self.chase.azimuth = BEHIND
        self._azimuth = BEHIND

        self.target_box = sv.BoxCornerAnnotator(
            thickness=6, corner_length=18, color=MAGENTA
        )
        self.target_label = sv.LabelAnnotator(
            text_scale=0.9, text_thickness=2, color=MAGENTA,
            text_position=sv.Position.TOP_CENTER,
        )
        self.other_box = sv.BoxAnnotator(thickness=2, color=GRAY)
        self.other_label = sv.LabelAnnotator(
            text_scale=0.7, text_thickness=2, color=GRAY
        )
        self.trace = sv.TraceAnnotator(thickness=4, trace_length=15, color=MAGENTA)
        self._display_ids = {}

        self.headless = headless
        self.path = path
        self.writer = None if headless else imageio.get_writer(path, fps=fps, quality=8)
        self.frames_written = 0

    def _display_id(self, tracker_id):
        """Small stable numbers, so the labels stay readable when ids churn."""
        tracker_id = int(tracker_id)
        if tracker_id not in self._display_ids:
            self._display_ids[tracker_id] = len(self._display_ids) + 1
        return self._display_ids[tracker_id]

    def start_frame(self, beside=False):
        """Aim the chase camera and render it, before the head views."""
        self._aim_chase(beside)
        if self.headless:
            self._chase_frame = None
        else:
            self.chase_renderer.update_scene(self.duck.data, camera=self.chase)
            self._chase_frame = self.chase_renderer.render()

    def _aim_chase(self, beside=False):
        """Follow the duck's heading, smoothed, so the third-person view always
        shows what it is walking toward. Swing to the side while it pecks, or
        the contact is hidden behind its own body."""
        _, yaw = self.duck.trunk_frame()
        wanted = math.degrees(yaw) + (BESIDE if beside else BEHIND)
        self._azimuth += 0.04 * ((wanted - self._azimuth + 180.0) % 360.0 - 180.0)
        self.chase.azimuth = self._azimuth

    def head_frame(self):
        """The duck's-eye view, upright."""
        self.head_renderer.update_scene(
            self.duck.data, camera="head_camera", scene_option=self.head_options
        )
        return np.ascontiguousarray(np.rot90(self.head_renderer.render()))

    def head_segmentation(self):
        self.seg_renderer.update_scene(
            self.duck.data, camera="head_camera", scene_option=self.head_options
        )
        return np.rot90(self.seg_renderer.render())

    def annotate(self, frame, tracked, target_id, births, now):
        """Magenta corners on the ball being fetched, plain boxes on the rest."""
        is_target = np.array(
            [int(tid) == target_id for tid in tracked.tracker_id], dtype=bool
        )
        target, others = tracked[is_target], tracked[~is_target]
        if len(target):
            # Trace only the target: tracing everything just paints the
            # walking head-bob across the whole frame.
            frame = self.trace.annotate(frame, target)
        if len(others):
            frame = self.other_box.annotate(frame, others)
            mature = np.array(
                [now - births.get(int(i), now) > LABEL_AFTER for i in others.tracker_id],
                dtype=bool,
            )
            if mature.any():
                frame = self.other_label.annotate(
                    frame, others[mature],
                    labels=[f"#{self._display_id(i)}" for i in others[mature].tracker_id],
                )
        if len(target):
            frame = self.target_box.annotate(frame, target)
            frame = self.target_label.annotate(
                frame, target, labels=["FETCH" for _ in target.tracker_id]
            )
        return frame

    def write(self, head):
        """Compose the chase view with the POV inset and write one frame."""
        if self.headless:
            return
        frame = self._chase_frame
        inset = cv2.resize(head, (INSET_W, INSET_H), interpolation=cv2.INTER_AREA)
        pad = 12
        frame[pad : pad + INSET_H + 4, MAIN_W - INSET_W - pad - 4 : MAIN_W - pad] = 30
        frame[
            pad + 2 : pad + 2 + INSET_H,
            MAIN_W - INSET_W - pad - 2 : MAIN_W - pad - 2,
        ] = inset
        self.writer.append_data(frame)
        self.frames_written += 1

    def close(self):
        if self.writer is not None:
            self.writer.close()
            print(f"wrote {self.path}: {self.frames_written} frames")
