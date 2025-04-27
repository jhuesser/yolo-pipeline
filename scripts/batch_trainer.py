import os
import subprocess
from tqdm import tqdm
import argparse

def find_datasets(output_root):
    datasets = []
    for d in os.listdir(output_root):
        dataset_path = os.path.join(output_root, d)
        train_path = os.path.join(dataset_path, 'train')
        val_path = os.path.join(dataset_path, 'val')
        if os.path.isdir(dataset_path) and os.path.isdir(train_path) and os.path.isdir(val_path):
            datasets.append(dataset_path)
    return datasets

def train_dataset(dataset_path, model, epochs, imgsz, device):
    cmd = [
        "yolo",
        "task=classify",
        "mode=train",
        f"model={model}",
        f"data={dataset_path}",
        f"epochs={epochs}",
        f"imgsz={imgsz}",
        f"device={device}"
    ]
    print(f"\n🚀 Launching training for: {dataset_path}")
    print(f"🔹 Command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def main(args):
    datasets = find_datasets(args.output_root)

    if not datasets:
        print(f"⚠️ No datasets with train/val found in {args.output_root}")
        return

    print(f"🔎 Found {len(datasets)} datasets to train.")

    for dataset_path in tqdm(datasets, desc="Training datasets"):
        try:
            train_dataset(dataset_path, args.model, args.epochs, args.imgsz, args.device)
        except subprocess.CalledProcessError as e:
            print(f"❌ Training failed for {dataset_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch train YOLOv8 classifiers on multiple datasets.")
    parser.add_argument("--output-root", required=True, help="Root output folder (e.g., output/)")
    parser.add_argument("--model", default="yolov8n-cls.pt", help="YOLOv8 classification model to use (default: yolov8n-cls.pt)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs (default: 100)")
    parser.add_argument("--imgsz", type=int, default=224, help="Image size (default: 224)")
    parser.add_argument("--device", default="mps", help="Device to train on (default: mps for Mac)")

    args = parser.parse_args()
    main(args)