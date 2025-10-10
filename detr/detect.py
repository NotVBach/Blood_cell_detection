from ultralytics import RTDETR
import os

# Load the trained model
model = RTDETR('runs/detect/rtdetr_defect/weights/best.pt')

# Run detection on test images
results = model.predict(
    source='./noaug/test',  # Path to test images dir
    save=True,              # Save annotated images to runs/detect/predict/
    conf=0.25,              # Confidence threshold
    iou=0.45,               # NMS IoU threshold
    device=0                 # GPU/CPU
)

# Print sample detections (for first image)
for r in results[:1]:
    print(r.boxes)  # Boxes, classes, confidences

# View saved predictions in runs/detect/predict/