import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import easyocr
import re

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Myanmar Driving License Extractor",
    layout="wide"
)

# -------------------------------------------------
# LOAD YOLO MODEL
# -------------------------------------------------
@st.cache_resource
def load_yolo():
    model_path = r"C:/Users/minthanttin/Desktop/Myanmar_DL_Datasets/runs/train/myanmar_dl/weights/best.pt"
    return YOLO(model_path)

yolo_model = load_yolo()

# -------------------------------------------------
# LOAD EASYOCR
# -------------------------------------------------
@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

ocr = load_ocr()

# -------------------------------------------------
# UTILITY: OCR TEXT CLEANER
# -------------------------------------------------
def clean_text(text):
    return text.replace("\n", " ").strip()

# -------------------------------------------------
# UTILITY: DETECT & CROP USING YOLO
# -------------------------------------------------
def detect_and_crop(image):
    results = yolo_model.predict(image, conf=0.4)
    boxes = results[0].boxes.xyxy.cpu().numpy()

    cropped_images = []
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        crop = image[y1:y2, x1:x2]
        cropped_images.append((i, crop))

    return results[0].plot(), cropped_images

# -------------------------------------------------
# UI HEADER
# -------------------------------------------------
st.title("🪪 Myanmar Driving License Detection + OCR")
st.write("YOLOv8 + EasyOCR | Offline | Cropping + Auto Extraction")

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
st.sidebar.title("Controls")
use_webcam = st.sidebar.checkbox("Use Webcam")
st.sidebar.info("Make sure your trained YOLO model path is correct.")

# -------------------------------------------------
# MAIN APP
# -------------------------------------------------
uploaded_file = None
frame = None

# --- Upload image ---
if not use_webcam:
    uploaded_file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        frame = np.array(img)

# --- Webcam mode ---
else:
    cam = st.camera_input("Take a photo")
    if cam:
        img = Image.open(cam).convert("RGB")
        frame = np.array(img)

# -------------------------------------------------
# PROCESS IMAGE
# -------------------------------------------------
if frame is not None:

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(frame, use_column_width=True)

    # YOLO detection + cropping
    annotated, crops = detect_and_crop(frame)

    with col2:
        st.subheader("YOLO Detection")
        st.image(annotated, use_column_width=True)

    st.markdown("---")
    st.header("📌 Cropped Regions & OCR")

    if len(crops) == 0:
        st.warning("No driving license detected.")
    else:
        for idx, crop in crops:

            st.subheader(f"Crop #{idx+1}")
            st.image(crop, width=400)

            # OCR
            result = ocr.readtext(crop)

            extracted_text = " ".join([clean_text(r[1]) for r in result])
            st.write("**OCR Output:**")
            st.success(extracted_text)

            # Optional extraction rules
            name_match = re.search(r"Name[: ]+([A-Za-z ]+)", extracted_text)
            id_match = re.search(r"(\d{6,12})", extracted_text)

            st.write("**Extracted Fields (Experimental):**")
            st.json({
                "Name": name_match.group(1) if name_match else "",
                "ID Number": id_match.group(1) if id_match else ""
            })