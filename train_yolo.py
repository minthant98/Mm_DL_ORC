from ultralytics import YOLO
import os

# ----------
#1. Dataset path
# ----------

#dataset_path = 'desktop/Myanmar_DL_Datasets/datasets'
#data_yaml = "C:\Users\minthanttin\Desktop\Myanmar_DL_Datasets\datasets\data.yaml"
# Path to dataset folder in Colab
dataset_path = '/content/Mm_DL_ORC/datasets'

# Path to data.yaml in Colab
data_yaml = '/content/Mm_DL_ORC/datasets/data.yaml'

# ----------
#2. Model Selection
# ----------    
# yolov8n = nano (small, good for small datasets)
model_name = "yolov8s.pt"

# -------------------------------
# 3. Training configuration
# -------------------------------
# Adjust epochs, batch size, and image size for small dataset
training_params = {
    "data": data_yaml,
    "model": model_name,
    "epochs": 100,        # more epochs for small datasets
    "batch": 4,           # small batch due to limited images
    "imgsz": 960,         # size of input images
    "patience": 20,       # early stopping if no improvement
    "workers": 2,         # number of data loader workers
    "project": "runs/train",  # folder to save results
    "name": "myanmar_dl_v2",
    "exist_ok": True,     # overwrite existing project folder if exists
    "augment": True       # apply simple augmentations (flip, rotate, brightness)
}

# -------------------------------
# 4. Start training
# -------------------------------
print("Starting YOLOv8 training for Myanmar Driving License fields...")
model = YOLO(training_params["model"])
model.train(
    data=training_params["data"],
    epochs=training_params["epochs"],
    batch=training_params["batch"],
    imgsz=training_params["imgsz"],
    patience=training_params["patience"],
    workers=training_params["workers"],
    project=training_params["project"],
    name=training_params["name"],
    exist_ok=training_params["exist_ok"],
    augment=training_params["augment"]
)
print("Training complete. Model weights saved in runs/train/myanmar_dl/weights/")