"""What the duck can see: detections, tracks, and what a box means in metres.

The track says which detection is ours; the detection says where it is. A
track coasting on its Kalman prediction still reports a box, but that box is a
guess, so nothing is ever measured from it.
"""

import math
import os

import mujoco
import numpy as np
import supervision as sv

from trackers import SORTTracker
from trackers.utils.iou import BIoU

from microduck import BALL_DIAMETER, CAMERA_FOV

# COCO classes RF-DETR reaches for on an orange ball: sports ball, apple and
# orange. Which one it picks depends on the background.
ROUND_THINGS = (37, 53, 55)


def focal_px(width):
    """Pixels per radian at the image centre, from the camera's own field of
    view. The head view is rendered portrait and rotated, so the camera's
    vertical fovy spans the width of the image the tracker sees."""
    return (width / 2) / math.tan(math.radians(CAMERA_FOV) / 2)


class BallDetector:
    """Boxes for every ball in view.

    The default reads them straight out of the simulator's segmentation
    buffer, which stands in for a perfect detector. DETECTOR=rfdetr runs
    RF-DETR Nano on the rendered frame instead, so the whole chain from pixels
    to motor commands is real.
    """

    def __init__(self, ball_geom_ids):
        self.ball_geom_ids = ball_geom_ids
        self.model = None
        if os.environ.get("DETECTOR") == "rfdetr":
            from rfdetr import RFDETRNano

            self.model = RFDETRNano()
            self.model.optimize_for_inference()

    def oracle_boxes(self, segmentation):
        """Ball boxes read from the segmentation buffer."""
        is_geom = segmentation[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
        masks = np.stack(
            [(segmentation[..., 0] == gid) & is_geom for gid in self.ball_geom_ids]
        )
        visible = masks.reshape(len(self.ball_geom_ids), -1).sum(axis=1) >= 3
        if not visible.any():
            return []
        boxes = sv.mask_to_xyxy(masks[visible], coordinate_convention="exclusive")
        return boxes.tolist()

    def __call__(self, frame, oracle_boxes):
        if self.model is not None:
            detections = self.model.predict(frame, threshold=0.3)
            return detections[np.isin(detections.class_id, ROUND_THINGS)]
        if not oracle_boxes:
            return sv.Detections.empty()
        count = len(oracle_boxes)
        return sv.Detections(
            xyxy=np.array(oracle_boxes, dtype=float),
            confidence=np.ones(count),
            class_id=np.zeros(count, dtype=int),
        )


def build_tracker(track_every=1):
    """SORT, tuned for small balls seen from a walking robot.

    Buffered IoU is the one non-default choice: the head bobs with every step,
    which moves a distant ball's box further than its own width between
    frames, and plain IoU association breaks there.
    """
    return SORTTracker(
        lost_track_buffer=150 // track_every,
        frame_rate=50.0 / track_every,
        minimum_consecutive_frames=2 if track_every == 1 else 1,
        minimum_iou_threshold=0.05,
        iou=BIoU(buffer_ratio=2.0),
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
