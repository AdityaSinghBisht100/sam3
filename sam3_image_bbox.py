import json
import os
import argparse
import numpy as np
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def main(args):
    with open(args.json, "r") as f:
        data = json.load(f)

    frame0 = next(f for f in data["frames"] if f["frame_no"] == 0)

    objects = [
        obj for obj in frame0["objects"]
        if "bbox" in obj and len(obj["bbox"]) == 4
    ]

    if len(objects) == 0:
        raise RuntimeError("No valid objects found in frame 0")

    boxes = [obj["bbox"] for obj in objects]
    labels = [obj.get("label", None) for obj in objects]

    image = Image.open(args.image).convert("RGB")

    model = build_sam3_image_model()
    processor = Sam3Processor(model)

    state = processor.set_image(image)

    output = processor.set_box_prompt(
        state=state,
        boxes=boxes,
        texts=labels,
    )

    masks = output["masks"]  
    combined_mask = np.zeros_like(masks[0], dtype=np.uint8)

    for mask in masks:
        combined_mask = np.logical_or(combined_mask, mask)

    combined_mask = combined_mask.astype(np.uint8) * 255

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "mask.png")
    Image.fromarray(combined_mask).save(out_path)

    print("segmentation mask saved at:", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SAM-3 single-image mask segmentation"
    )

    parser.add_argument("--image", required=True, help="Path to image")
    parser.add_argument("--json", required=True, help="Path to JSON")
    parser.add_argument(
        "--output_dir",
        default="sam3_output",
        help="Directory to save mask",
    )

    args = parser.parse_args()
    main(args)
