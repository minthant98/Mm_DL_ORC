import os
import random
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import albumentations as A
from pathlib import Path
import string

# ------------------ CONFIG ------------------

OUTPUT_IMG_DIR = r"C:\Users\minthanttin\Desktop\Myanmar_DL_Datasets\datasets\synthetic_aug\images"
OUTPUT_LABEL_DIR = r"C:\Users\minthanttin\Desktop\Myanmar_DL_Datasets\datasets\synthetic_aug\labels"

os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)

TEMPLATE_PATH = "template.png"
FONT_PATH = "arial.ttf"  # English font
FONT_SIZE = 32

IMG_WIDTH = 1200
IMG_HEIGHT = 600

NUM_IMAGES = 200

CLASSES = ["No.", "Name", "N.R.C. No", "Date of Birth", "Blood Type", "Valid up to"]

# ------------------ RANDOM FIELD GENERATORS ------------------

def random_license_number():
    letter = random.choice(string.ascii_uppercase)
    part1 = random.randint(10000, 99999)
    part2 = random.randint(10, 99)
    return f"{letter}/{part1}/{part2}"

def random_name():
    words = ["August", "Smith", "James", "Taylor", "Michael", "Lee", "Brown", "David", "Johnson", "Robert"]
    num_words = random.choice([2, 3, 4])
    return " ".join(random.choices(words, k=num_words))

def random_nrc():
    number1 = random.randint(1, 14)
    syllables = ["Ka", "La", "Ma", "Ta", "Na", "Pa", "Sa", "Wa"]
    four_syllables = "-".join(random.choices(syllables, k=4))
    number2 = random.randint(100000, 999999)
    return f"{number1}/{four_syllables}/{number2}"

def random_dob():
    day = random.randint(1, 28)
    month = random.randint(1, 12)
    year = random.randint(1970, 2005)
    return f"{day:02d}-{month:02d}-{year}"

def random_blood():
    return random.choice(["A", "B", "AB", "O"])

def random_expiry():
    day = random.randint(1, 28)
    month = random.randint(1, 12)
    year = random.randint(2025, 2035)
    return f"{day:02d}-{month:02d}-{year}"

# ------------------ AUGMENTATION ------------------

augment = A.Compose([
    A.GaussNoise(var_limit=(5, 50), p=0.3),
    A.Rotate(limit=5, p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.MotionBlur(blur_limit=3, p=0.3)
])

# ------------------ UTILITIES ------------------

def make_yolo_bbox(x, y, w, h, img_w, img_h):
    x_center = (x + w/2) / img_w
    y_center = (y + h/2) / img_h
    w /= img_w
    h /= img_h
    return x_center, y_center, w, h

def draw_text_with_shadow(draw, position, text, font, text_color=(0,0,0), shadow_color=(50,50,50)):
    x, y = position
    draw.text((x+2, y+2), text, font=font, fill=shadow_color)  # shadow
    draw.text((x, y), text, font=font, fill=text_color)

# ------------------ MAIN GENERATOR ------------------

def generate_synthetic_id(index):
    template = Image.open(TEMPLATE_PATH).convert("RGB")
    template = template.resize((IMG_WIDTH, IMG_HEIGHT))
    draw = ImageDraw.Draw(template)
    font = ImageFont.truetype(FONT_PATH, FONT_SIZE)

    labels = []
    fields = {
        0: random_license_number(),
        1: random_name(),
        2: random_nrc(),
        3: random_dob(),
        4: random_blood(),
        5: random_expiry(),
    }

    # Base coordinates (adjust to your template)
    base_coords = {
        0: (200, 100),
        1: (200, 160),
        2: (200, 220),
        3: (200, 280),
        4: (200, 340),
        5: (200, 400),
    }

    for cls_id, text in fields.items():
        x_off = random.randint(-5,5)
        y_off = random.randint(-5,5)
        x, y = base_coords[cls_id]
        x += x_off
        y += y_off

        draw_text_with_shadow(draw, (x,y), text, font)

        text_w, text_h = draw.textbbox((x, y), text, font=font)[2:]
        bbox = make_yolo_bbox(x, y, text_w, text_h, IMG_WIDTH, IMG_HEIGHT)
        labels.append((cls_id, bbox))

    img = cv2.cvtColor(np.array(template), cv2.COLOR_RGB2BGR)

    augmented = augment(image=img)
    img_aug = augmented["image"]

    img_name = f"synthetic_{index}.jpg"
    cv2.imwrite(os.path.join(OUTPUT_IMG_DIR, img_name), img_aug)

    label_path = os.path.join(OUTPUT_LABEL_DIR, f"synthetic_{index}.txt")
    with open(label_path, "w") as f:
        for cls_id, bbox in labels:
            x_c, y_c, w, h = bbox
            f.write(f"{cls_id} {x_c} {y_c} {w} {h}\n")

# ------------------ RUN ------------------

if __name__ == "__main__":
    print("Generating augmented synthetic Myanmar DL dataset (English text, updated formats)...")
    for i in range(NUM_IMAGES):
        generate_synthetic_id(i)
    print(f"\n✅ Done! Generated {NUM_IMAGES} synthetic ID images with updated field formats.")