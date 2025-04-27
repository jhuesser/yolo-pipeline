import os
import cv2
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from ultralytics import YOLO

# --- FUNCTIONS ---

def load_classifiers(classifiers_dir):
    classifiers = {}
    for model_file in os.listdir(classifiers_dir):
        if model_file.endswith(".pt"):
            key = model_file.replace(".pt", "")
            model_path = os.path.join(classifiers_dir, model_file)
            classifiers[key] = YOLO(model_path)
            print(f"✅ Loaded classifier for {key}")
    return classifiers

def crop_mask(img, mask, bbox):
    x1, y1, x2, y2 = map(int, bbox)

    if x2 <= x1 or y2 <= y1:
        print(f"⚠️ Invalid bbox: {bbox}")
        return None

    img_h, img_w = img.shape[:2]
    x1, y1, x2, y2 = map(lambda v: np.clip(v, 0, img_w if v in [x1, x2] else img_h), (x1, y1, x2, y2))

    cropped_img = img[y1:y2, x1:x2]
    cropped_mask = mask[y1:y2, x1:x2]

    if cropped_img.size == 0 or cropped_mask.size == 0:
        print(f"⚠️ Empty crop, skipping.")
        return None

    cropped_mask = (cropped_mask * 255).astype(np.uint8)

    if cropped_img.shape[:2] != cropped_mask.shape:
        print(f"⚠️ Size mismatch: cropped_img {cropped_img.shape}, cropped_mask {cropped_mask.shape}")
        return cropped_img

    result = cv2.bitwise_and(cropped_img, cropped_img, mask=cropped_mask)
    return result

def draw_label(img, text, bbox, color=(0, 255, 0)):
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    font_scale = 0.5
    font_thickness = 1
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    cv2.rectangle(img, (x1, y1 - h - 4), (x1 + w, y1), color, -1)
    cv2.putText(img, text, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)

def process_image(image_path, seg_model, classifiers, REGION_TO_CLASSIFIERS, device, conf_threshold, output_dir):
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"❌ Error: Cannot read {image_path}")
        return

    img_height, img_width = img.shape[:2]
    annotated_img = img.copy()

    seg_results = seg_model.predict(str(image_path), imgsz=640, device=device, conf=conf_threshold)[0]

    if not seg_results.masks:
        print(f"⚠️ No masks detected for {image_path}")
        return

    final_results = []

    for idx in range(len(seg_results.boxes)):
        box = seg_results.boxes[idx]
        mask = seg_results.masks.data[idx].cpu().numpy()

        label_id = int(box.cls.cpu().numpy().flatten()[0])
        conf = float(box.conf.cpu().numpy().flatten()[0])

        label_name = seg_model.names[label_id]
        bbox = box.xyxy.cpu().numpy().flatten()

        cropped = crop_mask(img, mask, bbox)
        if cropped is None:
            continue

        attributes = {}

        if label_name in REGION_TO_CLASSIFIERS:
            for clf_key in REGION_TO_CLASSIFIERS[label_name]:
                if clf_key in classifiers:
                    clf_model = classifiers[clf_key]
                    pred = clf_model.predict(cropped, imgsz=224, device=device, verbose=False)[0]

                    pred_label_idx = pred.probs.top1
                    pred_label = clf_model.names[pred_label_idx]
                    pred_conf = float(pred.probs.data[pred_label_idx])

                    attributes[clf_key] = {
                        "attribute": pred_label,
                        "confidence": pred_conf
                    }

        # Save result
        final_results.append({
            "region": label_name,
            "bbox": bbox.tolist(),
            "segmentation_confidence": conf,
            "attributes": attributes
        })

        # Draw
        label_to_draw = label_name
        if attributes:
            attr_texts = [f"{k.replace('_', ' ')}: {v['attribute']}" for k, v in attributes.items()]
            label_to_draw += " | " + ", ".join(attr_texts)

        draw_label(annotated_img, label_to_draw, bbox)

    # Save results
    image_stem = Path(image_path).stem
    out_img = os.path.join(output_dir, f"{image_stem}_annotated.jpg")
    out_json = os.path.join(output_dir, f"{image_stem}.json")

    cv2.imwrite(out_img, annotated_img)
    with open(out_json, "w") as f:
        json.dump(final_results, f, indent=4)

    print(f"✅ Saved outputs for {image_path}")

def main():
    parser = argparse.ArgumentParser(description="Segment and classify images.")
    parser.add_argument("--seg-model", required=True, help="Path to segmentation model .pt")
    parser.add_argument("--classifiers-dir", required=True, help="Path to classifiers directory")
    parser.add_argument("--input", required=True, help="Path to input image OR input folder")
    parser.add_argument("--output-dir", default="results", help="Directory to save outputs")
    parser.add_argument("--device", default="mps", help="Device to run on: mps, cuda, cpu")
    parser.add_argument("--conf-threshold", type=float, default=0.5, help="Segmentation confidence threshold")
    args = parser.parse_args()

    print("🚀 Starting batch pipeline...")

    os.makedirs(args.output_dir, exist_ok=True)

    seg_model = YOLO(args.seg_model)
    classifiers = load_classifiers(args.classifiers_dir)

    REGION_TO_CLASSIFIERS = {}
    for clf_name in classifiers.keys():
        prefix = clf_name.split("_")[0].capitalize()
        REGION_TO_CLASSIFIERS.setdefault(prefix, []).append(clf_name)

    input_path = Path(args.input)
    if input_path.is_file():
        images = [input_path]
    elif input_path.is_dir():
        images = list(input_path.glob("*.jpg")) + list(input_path.glob("*.jpeg")) + list(input_path.glob("*.png")) + list(input_path.glob("*.bmp"))
    else:
        print(f"❌ Invalid input: {args.input}")
        return

    for img_path in tqdm(images, desc="Processing images"):
        process_image(img_path, seg_model, classifiers, REGION_TO_CLASSIFIERS, args.device, args.conf_threshold, args.output_dir)

    print("\n🏁 Done processing all images!")

if __name__ == "__main__":
    main()