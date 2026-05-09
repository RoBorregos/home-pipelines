import os
# os.environ["MKL_THREADING_LAYER"] = "GNU" # for some systems
from ultralytics import YOLO
import argparse
# Argument parser for command line arguments
parser = argparse.ArgumentParser(description="Train YOLOv8 model")
parser.add_argument("--data", type=str, default="data.yaml", help="Path to the data configuration file")
parser.add_argument("--device", type=str, default="0", help="Device to use for training (e.g., '0' for GPU 0, 'cpu' for CPU)")

args = parser.parse_args()
data_path = args.data


model = YOLO("yolo11m.pt")  # "yolo11m.pt" -> start from pretained weights on COCO (.pt not yaml)

# Train the model
model.train(
    data=data_path,
    epochs=100,  # change to 200
    imgsz=640,
    batch=64,  # 32, 64, 128 or -1
    degrees=15,
    translate=0.1,
    shear=5,
    scale=0.75,
    perspective=0.001,
    flipud=0.3,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.2,
    copy_paste=0.1,
    patience=25,
    save_period=10,
    cache=True,
    amp=True,
    device=args.device,  # Use the device specified in command line arguments
    save_dir="./training_results",  # Directory to save training results
)