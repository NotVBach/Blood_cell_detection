- Intsall ultralytics
`pip install ultralytics`

- Training
`yolo train model=yolov8n.pt data=yolo/data.yaml epochs=50 imgsz=640 batch=16 device=0`

- Evaluation
`yolo val model=runs/detect/train/weights/best.pt data=data.yaml`

- Prediction
`yolo predict model=runs/detect/train/weights/best.pt source=yolo/dataset/images/test save=True`