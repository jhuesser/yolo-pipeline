import os
import argparse
import json
import cv2
import numpy as np
from tqdm import tqdm

def create_mask(points, img_w, img_h):
    pts = np.array([[int(x/100*img_w), int(y/100*img_h)] for x, y in points], np.int32)
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    if pts.shape[0] >= 3:
        cv2.fillPoly(mask, [pts], 255)
    return mask

def save_crop(img, mask, bbox, size=(256, 256)):
    x, y, w, h = bbox
    cropped = cv2.bitwise_and(img, img, mask=mask)[y:y+h, x:x+w]
    resized = cv2.resize(cropped, size, interpolation=cv2.INTER_AREA)
    return resized

def main(args):
    with open(args.labelstudio_json, "r") as f:
        data = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    for item in tqdm(data, desc="Processing annotations"):
        img_rel_path = item['data']['image'].replace('file://', '').lstrip('/')
        img_path = os.path.join(args.images_dir, os.path.basename(img_rel_path))
        
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to read image: {img_path}")
            continue
        
        img_h, img_w = img.shape[:2]
        results = item['annotations'][0]['result']

        # Maps
        shapes = {}
        choices_map = {}

        for r in results:
            if r['type'] in ['polygonlabels', 'rectanglelabels']:
                shapes[r['id']] = r
            elif r['type'] == 'choices':
                choices_map.setdefault(r['id'], []).append(r)

        for shape_id, shape_data in shapes.items():
            shape_type = shape_data['type']
            label_name = shape_data['value']['polygonlabels'][0] if shape_type == 'polygonlabels' else shape_data['value']['rectanglelabels'][0]

            if shape_type == 'polygonlabels':
                points = shape_data['value']['points']
                mask = create_mask(points, img_w, img_h)
                bbox = cv2.boundingRect(np.array([[int(x/100*img_w), int(y/100*img_h)] for x, y in points], np.int32))
            elif shape_type == 'rectanglelabels':
                x = shape_data['value']['x'] / 100 * img_w
                y = shape_data['value']['y'] / 100 * img_h
                w = shape_data['value']['width'] / 100 * img_w
                h = shape_data['value']['height'] / 100 * img_h
                bbox = (int(x), int(y), int(w), int(h))
                mask = np.zeros((img_h, img_w), dtype=np.uint8)
                cv2.rectangle(mask, (int(x), int(y)), (int(x+w), int(y+h)), 255, -1)

            if bbox[2] < 10 or bbox[3] < 10:
                continue  # skip too small

            crop = save_crop(img, mask, bbox, size=(args.crop_size, args.crop_size))

            attached_choices = choices_map.get(shape_id, [])

            if attached_choices:
                for choice in attached_choices:
                    attribute_name = choice['from_name']
                    for choice_value in choice['value']['choices']:
                        save_dir = os.path.join(args.output_dir, attribute_name, choice_value)
                        os.makedirs(save_dir, exist_ok=True)
                        save_path = os.path.join(save_dir, f"{os.path.splitext(os.path.basename(img_path))[0]}_{shape_id}.jpg")
                        cv2.imwrite(save_path, crop)
            else:
                # No attributes attached
                save_dir = os.path.join(args.output_dir, label_name, "unspecified")
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"{os.path.splitext(os.path.basename(img_path))[0]}_{shape_id}.jpg")
                cv2.imwrite(save_path, crop)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract crops from Label Studio annotations correctly.")
    parser.add_argument("--labelstudio-json", required=True, help="Path to Label Studio JSON export file.")
    parser.add_argument("--images-dir", required=True, help="Path to original images.")
    parser.add_argument("--output-dir", required=True, help="Path to output cropped images.")
    parser.add_argument("--crop-size", type=int, default=256, help="Crop output size (default: 256).")

    args = parser.parse_args()
    main(args)