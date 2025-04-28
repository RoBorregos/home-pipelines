# Test the model
from ultralytics import YOLO
# Load the best trained model
model = YOLO("content/runs/detect/train/weights/epoch75.pt")

# Run inference on test images
results = model.predict(source="content/test", save=True, conf=0.5)
