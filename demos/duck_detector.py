"""Microduck's shipped duck detector, as an sv.Detections source.

Wraps duck_detect.onnx from pollen-robotics/microduck and downloads the
weights on first use. The crate documents the model in duck-detect/src/lib.rs:
320x320 in, letterboxed and padded with 114 gray, planar [1, 5, N] out
(cx, cy, w, h, score), one class.
"""

import os
import urllib.request

import cv2
import numpy as np
import supervision as sv

PAD_VALUE = 114
URL = (
    "https://raw.githubusercontent.com/pollen-robotics/microduck/main/"
    "duck-detect/models/duck_detect.onnx"
)
DEFAULT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "duck_detect.onnx"
)


class DuckDetector:
    """Run duck_detect.onnx on RGB frames, return boxes in frame coordinates."""

    def __init__(self, path=DEFAULT_PATH, confidence=0.5, iou=0.5):
        import onnxruntime as ort

        if not os.path.exists(path):
            print(f"downloading duck_detect.onnx to {path}")
            urllib.request.urlretrieve(URL, path)
        self.session = ort.InferenceSession(path)
        self.confidence = confidence
        self.iou = iou

    def __call__(self, frame):
        """The crate documents RGB input, but on simulator renders the model
        scores 0.96 in BGR against 0.19 in RGB with identical box placement,
        so frames are flipped to BGR here."""
        scale = 320.0 / max(frame.shape[:2])
        resized = cv2.resize(frame[..., ::-1], None, fx=scale, fy=scale)
        pad_y = (320 - resized.shape[0]) // 2
        pad_x = (320 - resized.shape[1]) // 2
        canvas = np.full((320, 320, 3), PAD_VALUE, np.uint8)
        canvas[pad_y : pad_y + resized.shape[0], pad_x : pad_x + resized.shape[1]] = (
            resized
        )
        inp = (canvas.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        cx, cy, w, h, conf = self.session.run(None, {"images": inp})[0][0]
        keep = conf > self.confidence
        if not keep.any():
            return sv.Detections.empty()
        xyxy = sv.xcycwh_to_xyxy(
            np.stack([cx[keep], cy[keep], w[keep], h[keep]], axis=1)
        )
        detections = sv.Detections(
            xyxy=(xyxy - [pad_x, pad_y, pad_x, pad_y]) / scale,
            confidence=conf[keep],
            class_id=np.zeros(int(keep.sum()), dtype=int),
        )
        return detections.with_nms(threshold=self.iou)
