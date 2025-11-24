import os
from pathlib import Path
import cv2
import albumentations as A
import shutil

# ----------------------
# Paths (Windows style)
# ----------------------
INPUT_IMG_DIR = r"C:\Users\minthanttin\Desktop\Myanmar_DL_Datasets\datasets\images\train"
INPUT_LABEL_DIR = r"C:\Users\minthanttin\Desktop\Myanmar_DL_Datasets\datasets\labels\train"

OUTPUT_IMG_DIR = r"C:\Users\minthanttin\Desktop\Myanmar_DL_Datasets\datasets\augmented\images"
OUTPUT_LABEL_DIR = r"C:\Users\minthanttin\Desktop\Myanmar_DL_Datasets\datasets\augmented\labels"

os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)

# ----------------------
# Augmentation pipeline
# ----------------------
# NOTE: No flips to keep text readable
transform = A.Compose([
    A.GaussNoise(var_limit=(5, 50), p=0.3),
    A.Rotate(limit=10, p=0.5),
    A.RandomBrightnessContrast(p=0.5),
],
bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# ----------------------
# Process each image
# ----------------------
for img_file in os.listdir(INPUT_IMG_DIR):
    if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    img_path = os.path.join(INPUT_IMG_DIR, img_file)
    label_path = os.path.join(INPUT_LABEL_DIR, Path(img_file).stem + ".txt")

    # Skip if label missing
    if not os.path.exists(label_path):
        print(f"⚠ Label missing for: {img_file}")
        continue

    # Read image
    image = cv2.imread(img_path)
    if image is None:
        continue

    # Read YOLO labels
    bboxes = []
    class_labels = []
    with open(label_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            cls = int(parts[0])
            x_center, y_center, w, h = map(float, parts[1:5])
            bboxes.append([x_center, y_center, w, h])
            class_labels.append(cls)

    # Apply augmentation
    augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
    image_aug = augmented['image']
    bboxes_aug = augmented['bboxes']
    labels_aug = augmented['class_labels']

    # Save augmented image
    aug_img_name = Path(img_file).stem + "_aug.jpg"
    cv2.imwrite(os.path.join(OUTPUT_IMG_DIR, aug_img_name), image_aug)

    # Save augmented labels
    with open(os.path.join(OUTPUT_LABEL_DIR, Path(img_file).stem + "_aug.txt"), 'w') as f:
        for cls, bbox in zip(labels_aug, bboxes_aug):
            f.write(f"{cls} {' '.join(map(str, bbox))}\n")

print("✅ Augmentation completed successfully!")