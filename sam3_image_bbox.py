import json
import os
import argparse
import numpy as np
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def xyxy_to_cxcywh_normalized(box, img_w, img_h):
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = x2 - x1
    h = y2 - y1
    return [cx / img_w, cy / img_h, w / img_w, h / img_h]


def clip_mask_to_box(mask, box):
    x1, y1, x2, y2 = map(int, box)
    clipped = np.zeros_like(mask)
    clipped[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
    return clipped


def main(args):
    with open(args.json, "r") as f:
        data = json.load(f)

    frame0 = next(f for f in data["frames"] if f["frame_no"] == 0)
    objects = frame0["objects"]

    image = Image.open(args.image).convert("RGB")
    img_w, img_h = image.size
    print(f"Image size: {img_w} x {img_h}")

    model = build_sam3_image_model()
    processor = Sam3Processor(model)

    state = processor.set_image(image)

    valid_objects = []

    for obj in objects:
        if "bbox" not in obj or len(obj["bbox"]) != 4:
            continue

        box = obj["bbox"]
        if box[2] <= box[0] or box[3] <= box[1]:
            continue

        box_norm = xyxy_to_cxcywh_normalized(box, img_w, img_h)

        state = processor.add_geometric_prompt(
            box=box_norm,
            label=True,
            state=state,
        )

        valid_objects.append(obj)

    if len(valid_objects) == 0:
        raise RuntimeError("No valid objects found")

    masks = state["masks"]   
    os.makedirs(args.output_dir, exist_ok=True)

    for obj, mask in zip(valid_objects, masks):
        mask_np = mask.squeeze(0).cpu().numpy().astype(np.uint8)

        clipped_mask = clip_mask_to_box(mask_np, obj["bbox"]) * 255

        raw_id = obj.get("object_id", "unknown")
        obj_id = raw_id.replace("<", "").replace(">", "")
        out_path = os.path.join(
            args.output_dir, f"mask_{obj_id}.png"
        )

        Image.fromarray(clipped_mask).save(out_path)
        print(f"Saved mask for {obj_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SAM-3: segmentation mask for each object"
    )
    parser.add_argument("--image", required=True)
    parser.add_argument("--json", required=True)
    parser.add_argument(
        "--output_dir",
        default="output_masks",
    )

    args = parser.parse_args()
    main(args)
