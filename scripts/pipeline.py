import os
import cv2
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

# --- CONFIGURATION ---
SEG_MODEL_PATH = "models/segmentation/best.pt"    # path to your segmentation model
CLASSIFIERS_DIR = "models/classifiers/"                 # folder containing classifiers (hair_color.pt, eye_color.pt, etc.)
INPUT_IMAGE = "06_6b1u9zg1orje1.jpg.jpg"                        # the image to process
DEVICE = "mps"                                   # mps (Mac) or cuda/cpu
CONFIDENCE_THRESHOLD = 0.5                       # min confidence for segmentation
RESULTS_JSON = "results.json"                    # save prediction results
RESULT_IMAGE = "result_annotated.jpg"             # output image with drawn labels

# --- FUNCTIONS ---

def load_classifiers(classifiers_dir):
    classifiers = {}
    for model_file in os.listdir(classifiers_dir):
        if model_file.endswith(".pt"):
            name = model_file.replace(".pt", "")
            model_path = os.path.join(classifiers_dir, model_file)
            classifiers[name] = YOLO(model_path)
            print(f"✅ Loaded classifier for {name}")
    return classifiers

def crop_mask(img, mask, bbox):
    """Crop masked region using bbox and mask"""
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

    cropped_img = img[y1:y2, x1:x2]
    cropped_mask = mask[y1:y2, x1:x2]

    cropped_mask = (cropped_mask * 255).astype(np.uint8)

    if cropped_img.shape[:2] != cropped_mask.shape:
        print(f"⚠️ Size mismatch: cropped_img {cropped_img.shape}, cropped_mask {cropped_mask.shape}")
        return cropped_img

    result = cv2.bitwise_and(cropped_img, cropped_img, mask=cropped_mask)
    return result

def draw_label(img, text, bbox, color=(0, 255, 0)):
    """Draw bounding box and text label"""
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    font_scale = 0.5
    font_thickness = 1
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    cv2.rectangle(img, (x1, y1 - h - 4), (x1 + w, y1), color, -1)
    cv2.putText(img, text, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)

def main():
    print("🚀 Starting pipeline...")

    # Load models
    seg_model = YOLO(SEG_MODEL_PATH)
    print(f"✅ Loaded segmentation model from {SEG_MODEL_PATH}")

    classifiers = load_classifiers(CLASSIFIERS_DIR)

    # Read input image
    img = cv2.imread(INPUT_IMAGE)
    if img is None:
        print(f"❌ Error: cannot read {INPUT_IMAGE}")
        return

    img_height, img_width = img.shape[:2]
    annotated_img = img.copy()  # Copy for visualization

    # Run segmentation
    seg_results = seg_model.predict(INPUT_IMAGE, imgsz=640, device=DEVICE, conf=CONFIDENCE_THRESHOLD)[0]

    if not seg_results.masks:
        print("⚠️ No masks detected!")
        return

    final_results = []

    # Process each region
    for idx in tqdm(range(len(seg_results.boxes)), desc="Processing detected regions"):
        box = seg_results.boxes[idx]
        mask = seg_results.masks.data[idx].cpu().numpy()

        label_id = int(box.cls.cpu().numpy())
        conf = float(box.conf.cpu().numpy())

        label_name = seg_model.names[label_id]
        bbox = box.xyxy.cpu().numpy().flatten()

        print(f"\n🔎 Detected: {label_name} (conf {conf:.2f})")

        # Crop
        cropped = crop_mask(img, mask, bbox)

        # Classifier matching
        clf_key = label_name.replace(" ", "_")
        attribute = None
        attr_conf = None

        if clf_key in classifiers:
            clf_model = classifiers[clf_key]
            pred = clf_model.predict(cropped, imgsz=224, device=DEVICE, verbose=False)[0]

            pred_label = clf_model.names[int(torch.argmax(pred.probs))]
            attr_conf = float(pred.probs.max())
            attribute = pred_label

            print(f"🧠 {label_name}: Predicted attribute → {pred_label} (conf {attr_conf:.2f})")

            label_to_draw = f"{label_name}: {pred_label}"
        else:
            print(f"⚡ No classifier for {label_name}")
            label_to_draw = label_name

        # Save result
        final_results.append({
            "region": label_name,
            "attribute": attribute,
            "bbox": bbox.tolist(),
            "segmentation_confidence": conf,
            "attribute_confidence": attr_conf
        })

        # Draw
        draw_label(annotated_img, label_to_draw, bbox)

    # Save annotated result
    cv2.imwrite(RESULT_IMAGE, annotated_img)
    print(f"\n✅ Saved annotated image to {RESULT_IMAGE}")

    # Save results to JSON
    with open(RESULTS_JSON, "w") as f:
        json.dump(final_results, f, indent=4)
    print(f"✅ Saved results to {RESULTS_JSON}")

    # Print final output
    print("\n🏁 FINAL RESULTS:")
    for res in final_results:
        print(f"- {res['region']} → {res['attribute']} (conf {res['attribute_confidence']})")

if __name__ == "__main__":
    main()