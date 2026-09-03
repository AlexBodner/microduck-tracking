"""A Roboflow-trained YOLO-Lite model package, as an sv.Detections source.

This is the detector the edge recipe produces: train YOLO-Lite on the Roboflow
platform, take the `weights.onnx` out of its model package, and run it. The
same ONNX is what `edge/convert_rknn.py` compiles for Microduck's NPU, so what
runs here and what runs on the robot are the same graph.

Preprocessing and the output layout are read from `inference_config.json`
rather than assumed, because getting either wrong produces a model that runs
and detects nothing.
"""

import json
import os

import cv2
import numpy as np
import supervision as sv


class YOLOLiteDetector:
    """Run a YOLO-Lite model package on RGB frames.

    The package ships three output tensors, NMS left outside the graph:
    `boxes_xyxy` already in network pixels, plus objectness and class logits.
    Confidence is sigmoid(objectness) * sigmoid(class), matching Roboflow's own
    unfused post-processing.
    """

    def __init__(self, package_dir, confidence=0.5, iou=0.5):
        import onnxruntime as ort

        config_path = os.path.join(package_dir, "inference_config.json")
        weights_path = os.path.join(package_dir, "weights.onnx")
        network_input = json.load(open(config_path))["network_input"]
        size = network_input["training_input_size"]
        self.width, self.height = size["width"], size["height"]
        self.scale = network_input.get("scaling_factor", 255)
        means, stds = network_input["normalization"]
        self.mean = np.array(means, dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array(stds, dtype=np.float32).reshape(3, 1, 1)

        names_path = os.path.join(package_dir, "class_names.txt")
        with open(names_path) as handle:
            self.class_names = [line.strip() for line in handle if line.strip()]

        self.session = ort.InferenceSession(weights_path)
        self.input_name = self.session.get_inputs()[0].name
        self.confidence = confidence
        self.iou = iou

    def __call__(self, frame):
        original_height, original_width = frame.shape[:2]
        # The package says "stretch", so the aspect ratio is not preserved and
        # rescaling the boxes back is a straight per-axis factor.
        resized = cv2.resize(frame, (self.width, self.height))
        blob = resized.astype(np.float32).transpose(2, 0, 1) / self.scale
        blob = ((blob - self.mean) / self.std)[None]

        boxes, obj_logits, cls_logits = self.session.run(None, {self.input_name: blob})
        scores = _sigmoid(obj_logits[0]) * _sigmoid(cls_logits[0])
        class_ids = scores.argmax(axis=1)
        confidences = scores.max(axis=1)

        keep = confidences > self.confidence
        if not keep.any():
            return sv.Detections.empty()

        xyxy = boxes[0][keep] * [
            original_width / self.width, original_height / self.height,
            original_width / self.width, original_height / self.height,
        ]
        detections = sv.Detections(
            xyxy=xyxy.astype(float),
            confidence=confidences[keep].astype(float),
            class_id=class_ids[keep].astype(int),
        )
        return detections.with_nms(threshold=self.iou)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
