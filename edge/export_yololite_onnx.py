#!/usr/bin/env python3
"""Export a YOLO-Lite edge variant to raw ONNX.

Raw, not decoded: NMS never maps to an NPU, so the graph has to stop at the
detection heads and the boxes get decoded on the CPU. Microduck's own
duck-detect ships the same way, emitting a planar [1, 5, N] tensor.

Weights are not the point here. This exports the architecture so the RKNN
compiler can be asked which operators it can place on the NPU.

    python edge/export_yololite_onnx.py --yololite <path to roboflow/yololite>
"""

import argparse
import os
import sys

import torch

VARIANTS = {
    "edge_n": dict(backbone="mobilenetv4_conv_small_050", fpn_channels=160,
                   width_multiple=0.60, depth_multiple=0.65, head_depth=1),
    "edge_s": dict(backbone="mobilenetv4_conv_small_050", fpn_channels=192,
                   width_multiple=0.75, depth_multiple=0.75, head_depth=1),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yololite", required=True, help="clone of roboflow/yololite")
    parser.add_argument("--variant", default="edge_n", choices=sorted(VARIANTS))
    parser.add_argument("--img-size", type=int, default=320)
    parser.add_argument("--num-classes", type=int, default=1)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    sys.path.insert(0, os.path.abspath(args.yololite))
    from yololite.scripts.model.model_v2 import YOLOLiteMS_CPU

    model = YOLOLiteMS_CPU(
        num_classes=args.num_classes,
        num_anchors_per_level=(1, 1, 1),
        **VARIANTS[args.variant],
    ).eval()
    print(f"{args.variant}: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M parameters")

    dummy = torch.zeros(1, 3, args.img_size, args.img_size)
    with torch.no_grad():
        outputs = model(dummy)
    print("raw outputs:", [tuple(o.shape) for o in outputs])

    out = args.out or f"{args.variant}_{args.img_size}_raw.onnx"
    torch.onnx.export(
        model, dummy, out, opset_version=17,
        input_names=["images"], do_constant_folding=True,
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
