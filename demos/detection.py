"""Finding balls in a frame. Detection only: no identity, no tracking."""

import os

import mujoco
import numpy as np
import supervision as sv

# COCO classes RF-DETR reaches for on an orange ball: sports ball, apple and
# orange. Which one it picks depends on the background.
ROUND_THINGS = (37, 53, 55)


class BallDetector:
    """Boxes for every ball in view.

    The default reads them from the simulator's segmentation buffer, standing
    in for a perfect detector. DETECTOR=rfdetr runs RF-DETR Nano on the
    rendered frame instead, so the whole chain from pixels to motor commands
    is real.
    """

    def __init__(self, ball_geom_ids, confidence=0.3):
        self.ball_geom_ids = ball_geom_ids
        self.confidence = confidence
        self.model = None
        if os.environ.get("DETECTOR") == "rfdetr":
            from rfdetr import RFDETRNano

            self.model = RFDETRNano()
            self.model.optimize_for_inference()

    def oracle_boxes(self, segmentation):
        """Ball boxes read straight out of the segmentation buffer."""
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
            detections = self.model.predict(frame, threshold=self.confidence)
            return detections[np.isin(detections.class_id, ROUND_THINGS)]
        if not oracle_boxes:
            return sv.Detections.empty()
        count = len(oracle_boxes)
        return sv.Detections(
            xyxy=np.array(oracle_boxes, dtype=float),
            confidence=np.ones(count),
            class_id=np.zeros(count, dtype=int),
        )
