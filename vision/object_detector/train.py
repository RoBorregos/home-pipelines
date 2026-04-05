from ultralytics import YOLO

model = YOLO("pipeline_runs/trainnig_abr_2/training/yolo/weights/best.pt")

# Infer single image

results = model("pipeline_runs/trainnig_abr_2/training/images/val/IMG_20230612_122644.jpg")

# Show results
results.show()