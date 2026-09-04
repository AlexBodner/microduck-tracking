#!/usr/bin/env python3
"""Upload a rendered dataset to a Roboflow project.

Reads the YOLO output of `render_dataset.py` and sends each frame with its
boxes. Annotations go up as Pascal VOC, which carries class names inline, so
there is no label map to keep in sync.

Needs ROBOFLOW_API_KEY in the environment, from the workspace that owns the
project.

    python edge/upload_dataset.py --dataset ball_dataset \\
        --project alexbodner/microduck-balls
"""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from xml.etree import ElementTree as ET

import requests

BASE = "https://api.roboflow.com/dataset"
CLASS_NAME = "ball"


def voc_annotation(name, width, height, boxes):
    root = ET.Element("annotation")
    ET.SubElement(root, "filename").text = name
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = "3"
    for x1, y1, x2, y2 in boxes:
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = CLASS_NAME
        box = ET.SubElement(obj, "bndbox")
        for tag, value in (("xmin", x1), ("ymin", y1), ("xmax", x2), ("ymax", y2)):
            ET.SubElement(box, tag).text = str(round(value))
    return ET.tostring(root, encoding="unicode")


def read_boxes(label_path, width, height):
    boxes = []
    with open(label_path) as handle:
        for line in handle:
            parts = line.split()
            if len(parts) != 5:
                continue
            _, cx, cy, bw, bh = (float(v) for v in parts)
            boxes.append((
                (cx - bw / 2) * width, (cy - bh / 2) * height,
                (cx + bw / 2) * width, (cy + bh / 2) * height,
            ))
    return boxes


def split_for(index, scene_length):
    """Split by scene, never by frame.

    `render_dataset.py` keeps a ball arrangement for `scene_length` consecutive
    frames and only the head angle changes between them. Splitting on the frame
    index therefore puts near-identical frames in train and test, and the
    platform reports something close to 100% because the model has memorised
    those arrangements. Whole scenes go to one split or another.
    """
    scene = index // scene_length
    if scene % 6 == 5:
        return "test"
    if scene % 6 == 4:
        return "valid"
    return "train"


def upload_one(args, project, api_key, index, stem):
    image_path = os.path.join(args.dataset, "images", f"{stem}.jpg")
    label_path = os.path.join(args.dataset, "labels", f"{stem}.txt")
    split = split_for(index, args.scene_length)

    with open(image_path, "rb") as handle:
        response = requests.post(
            f"{BASE}/{project}/upload",
            params={"api_key": api_key, "name": f"{stem}.jpg", "split": split},
            files={"file": handle}, timeout=120,
        )
    payload = response.json()
    image_id = payload.get("id")
    if not image_id:
        return f"upload failed for {stem}: {str(payload)[:120]}"

    from PIL import Image

    with Image.open(image_path) as image:
        width, height = image.size
    boxes = read_boxes(label_path, width, height)
    response = requests.post(
        f"{BASE}/{project}/annotate/{image_id}",
        params={"api_key": api_key, "name": f"{stem}.xml"},
        data=voc_annotation(f"{stem}.jpg", width, height, boxes).encode(),
        headers={"Content-Type": "text/plain"}, timeout=120,
    )
    if not response.ok or response.json().get("error"):
        return f"annotate failed for {stem}: {response.text[:120]}"
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--project", required=True, help="workspace/project")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--scene-length", type=int, default=25,
                        help="frames per ball arrangement in render_dataset.py")
    args = parser.parse_args()

    api_key = os.environ["ROBOFLOW_API_KEY"]
    project = args.project.split("/")[-1]
    stems = sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(os.path.join(args.dataset, "images"))
        if f.endswith(".jpg")
    )
    print(f"uploading {len(stems)} frames to {args.project}")

    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(upload_one, args, project, api_key, i, stem)
            for i, stem in enumerate(stems)
        ]
        for done, future in enumerate(futures, start=1):
            problem = future.result()
            if problem:
                failures.append(problem)
            if done % 50 == 0:
                print(f"  {done}/{len(stems)}", flush=True)

    print(f"done: {len(stems) - len(failures)} uploaded, {len(failures)} failed")
    for problem in failures[:5]:
        print("  ", problem)


if __name__ == "__main__":
    main()
