import os
import shutil
import random
import argparse
from tqdm import tqdm

def split_images(class_folder, train_class_folder, val_class_folder, split_ratio=0.8):
    images = [f for f in os.listdir(class_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not images:
        return

    random.shuffle(images)

    split_point = int(len(images) * split_ratio)
    train_images = images[:split_point]
    val_images = images[split_point:]

    os.makedirs(train_class_folder, exist_ok=True)
    os.makedirs(val_class_folder, exist_ok=True)

    for img in train_images:
        src = os.path.join(class_folder, img)
        dst = os.path.join(train_class_folder, img)
        shutil.copy2(src, dst)

    for img in val_images:
        src = os.path.join(class_folder, img)
        dst = os.path.join(val_class_folder, img)
        shutil.copy2(src, dst)

def split_single_dataset(dataset_root, split_ratio=0.8):
    class_folders = [
        d for d in os.listdir(dataset_root)
        if os.path.isdir(os.path.join(dataset_root, d)) and d not in ['train', 'val', 'unspecified']
    ]

    if not class_folders:
        print(f"⚠️ Skipping {dataset_root} — no valid classes found.")
        return

    train_root = os.path.join(dataset_root, 'train')
    val_root = os.path.join(dataset_root, 'val')

    os.makedirs(train_root, exist_ok=True)
    os.makedirs(val_root, exist_ok=True)

    for class_name in tqdm(class_folders, desc=f"Splitting {os.path.basename(dataset_root)}", leave=False):
        class_folder = os.path.join(dataset_root, class_name)
        train_class_folder = os.path.join(train_root, class_name)
        val_class_folder = os.path.join(val_root, class_name)

        split_images(class_folder, train_class_folder, val_class_folder, split_ratio)

    print(f"✅ Done splitting {dataset_root}")

def main(args):
    root = os.path.abspath(args.dataset_root)
    print(f"🔎 Scanning all datasets inside: {root}")

    # Find all sub-datasets
    dataset_folders = [
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    ]

    for dataset_name in tqdm(dataset_folders, desc="Processing datasets"):
        dataset_path = os.path.join(root, dataset_name)
        split_single_dataset(dataset_path, args.split_ratio)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto split ALL datasets inside a root folder into train/val.")
    parser.add_argument("--dataset-root", required=True, help="Root folder (e.g., output/)")
    parser.add_argument("--split-ratio", type=float, default=0.8, help="Train/val split ratio (default 0.8)")

    args = parser.parse_args()
    main(args)