import os
from PIL import Image
from PIL import UnidentifiedImageError
# base_path = "test2"
# size = 640

# for filename in os.listdir(base_path):
#     file_path = os.path.join(base_path, filename)

#     try:
#         with Image.open(file_path) as img:
#             img = img.resize((size, size))
#             img.save(file_path)
#     except (UnidentifiedImageError, OSError) as e:
#         print(f"Removing corrupt file: {file_path}")
#         os.remove(file_path)  # delete corrupt image

# angle = -90

# def rotate_images_in_directory(directory):
#     count = 0
#     for filename in os.listdir(directory):
#         filepath = os.path.join(directory, filename)
#         if os.path.isfile(filepath):
#             try:
#                 with Image.open(filepath) as img:
#                     rotated = img.rotate(angle, expand=True)  # Rotate right (clockwise)
#                     rotated.save(filepath)
#                     count += 1
#             except Exception as e:
#                 print(f"Failed to process {filename}: {e}")
#     print(f"Rotated {count} images in {directory}")

# # Example usage:
# rotate_images_in_directory("test2")

# Test the model
from ultralytics import YOLO
# Load the best trained model
model = YOLO("content/runs/detect/train11/weights/best.pt")

# Run inference on test images
results = model.predict(source="test2", save=True, conf=0.5)
