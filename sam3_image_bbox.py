import json
import os
import argparse
import numpy as np
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def xyxy_to_cxcywh_normalized(box, img_w, img_h):
    """
    Convert [x_min, y_min, x_max, y_max] (pixels)
    to [cx, cy, w, h] normalized to [0,1] for SAM-3
    """
    x_min, y_min, x_max, y_max = box

    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    w = x_max - x_min
    h = y_max - y_min

    return [
        cx / img_w,
        cy / img_h,
        w / img_w,
        h / img_h,
    ]


def main(args):
    with open(args.json, "r") as f:
        data = json.load(f)

    frame0 = next(f for f in data["frames"] if f["frame_no"] == 0)
    objects = frame0.get("objects", [])

    image = Image.open(args.image).convert("RGB")
    img_w, img_h = image.size
    print(f"[INFO] Image size: {img_w} x {img_h}")

    model = build_sam3_image_model()
    processor = Sam3Processor(model)

    state = processor.set_image(image)
    valid_boxes = 0

    for obj in objects:
        if "bbox" not in obj or len(obj["bbox"]) != 4:
            continue

        x_min, y_min, x_max, y_max = obj["bbox"]
        if x_max <= x_min or y_max <= y_min:
            continue

        box_norm = xyxy_to_cxcywh_normalized(
            obj["bbox"], img_w, img_h
        )

        state = processor.add_geometric_prompt(
            box=box_norm,
            label=True,   
            state=state,
        )

        valid_boxes += 1

    if valid_boxes == 0:
        raise RuntimeError("No valid bounding boxes found")

    print(f"[INFO] Added {valid_boxes} box prompts")

    masks = state["masks"]  
    combined_mask = np.zeros((img_h, img_w), dtype=np.uint8)

    for mask in masks:
        mask_np = mask.squeeze(0).cpu().numpy().astype(np.uint8)
        combined_mask |= mask_np

    combined_mask *= 255

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "frame_mask.png")
    Image.fromarray(combined_mask).save(out_path)

    print("Segmentation mask saved to:", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SAM-3 image segmentation using bounding boxes "
    )
    parser.add_argument("--image", required=True, help="Frame image path")
    parser.add_argument("--json", required=True, help="Annotation JSON path")
    parser.add_argument(
        "--output_dir",
        default="output_masks",
        help="Directory to save combined mask",
    )

    args = parser.parse_args()
    main(args)
