#!/usr/bin/env python3
"""Compile an ONNX detector to RKNN for a Rockchip NPU.

The interesting output is not the .rknn file, it is the compiler's own report
of which operators it could place on the NPU and which fall back to the CPU.
RF-DETR fails here: its attention does not convert. A convolutional detector
should place everything.

rknn-toolkit2 needs Linux x86_64, onnx==1.15.0 (it calls onnx.mapping, removed
in 1.16) and numpy<2.

    python edge/convert_rknn.py --onnx edge_n_320_raw.onnx --target rk3566
"""

import argparse
import json
import os

import numpy as np
from PIL import Image
from rknn.api import RKNN


def write_calibration_set(directory, img_size, count=8):
    """INT8 needs sample inputs to pick quantization ranges. Noise is enough to
    answer whether the graph compiles; a real deployment calibrates on real
    frames, and the accuracy of the quantized model depends on it."""
    os.makedirs(directory, exist_ok=True)
    listing = os.path.join(directory, "calibration.txt")
    paths = []
    for i in range(count):
        path = os.path.join(directory, f"calib_{i}.jpg")
        noise = (np.random.rand(img_size, img_size, 3) * 255).astype(np.uint8)
        Image.fromarray(noise).save(path)
        # rknn resolves these relative to the list file, so store bare names.
        paths.append(os.path.basename(path))
    with open(listing, "w") as handle:
        handle.write("\n".join(paths) + "\n")
    return listing


def normalization_from_config(config_path):
    """RKNN folds normalization into the graph, so it has to match exactly what
    the model was trained with. A Roboflow model package states that in
    inference_config.json: means and stds are fractions of the scaling factor,
    which RKNN wants in pixel units."""
    if config_path is None:
        return [[0, 0, 0]], [[255, 255, 255]]
    network_input = json.load(open(config_path))["network_input"]
    scale = network_input.get("scaling_factor", 255)
    means, stds = network_input["normalization"]
    return [[m * scale for m in means]], [[s * scale for s in stds]]


def build(onnx_path, target, quantize, calibration, out_path, config_path=None):
    mean_values, std_values = normalization_from_config(config_path)
    rknn = RKNN(verbose=True)
    rknn.config(mean_values=mean_values, std_values=std_values,
                target_platform=target)
    if rknn.load_onnx(model=onnx_path) != 0:
        return "load_onnx failed"
    dataset = calibration if quantize else None
    if rknn.build(do_quantization=quantize, dataset=dataset) != 0:
        return "build failed"
    if rknn.export_rknn(out_path) != 0:
        return "export failed"
    rknn.release()
    return f"OK -> {out_path} ({os.path.getsize(out_path) / 1e6:.2f} MB)"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--target", default="rk3566")
    parser.add_argument("--img-size", type=int, default=320)
    parser.add_argument("--workdir", default="rknn_build")
    parser.add_argument("--inference-config", default=None,
                        help="inference_config.json from a Roboflow model package")
    args = parser.parse_args()

    os.makedirs(args.workdir, exist_ok=True)
    calibration = write_calibration_set(args.workdir, args.img_size)
    stem = os.path.splitext(os.path.basename(args.onnx))[0]

    for quantize in (False, True):
        tag = "int8" if quantize else "fp16"
        out = os.path.join(args.workdir, f"{stem}_{args.target}_{tag}.rknn")
        print(f"\n===== {args.target} {tag.upper()} =====", flush=True)
        result = build(args.onnx, args.target, quantize, calibration, out,
                       args.inference_config)
        print(f"RESULT {tag}: {result}", flush=True)


if __name__ == "__main__":
    main()
