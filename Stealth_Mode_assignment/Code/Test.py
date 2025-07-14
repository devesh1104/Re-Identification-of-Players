from ultralytics import YOLO

# Load a COCO-pretrained YOLOv8n model
model = YOLO(r"Model\best.pt")

model.info()
