# Test the model
from ultralytics import YOLO

# Load the best trained model
model = YOLO("content/runs/detect/train/weights/best.pt")

# Run inference on test images
results = model.predict(source="content/ds_final/test/images", save=True, save_txt=True, conf=0.5)
