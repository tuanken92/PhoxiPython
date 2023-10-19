from ultralytics import YOLO
from ultralytics import settings

print(settings)


model_path = "../model/box1.pt"

# Load a model
model = YOLO(model_path)  # pretrained YOLOv8n model
source = "test1.png"

# Make predictions
results = model.predict(source, save=False, imgsz=320, conf=0.5)

print(results)

# Extract bounding box dimensions
boxes = results[0].boxes.xywh.cpu()
for box in boxes:
    x, y, w, h = box
    print(f"Width of Box: {w}, Height of Box: {h}")
    print(box)
