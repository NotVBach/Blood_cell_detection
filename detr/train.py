from ultralytics import RTDETR

# Load a pretrained RT-DETR model
model = RTDETR('rtdetr-l.pt')  # Or 'rtdetr-x.pt' for larger model

# Train on your dataset
results = model.train(
    data='dataset.yaml',  # Path to YAML
    epochs=1,            # Number of epochs (increase for better results)
    imgsz=640,            # Image size (matches your ~360x363 images; pad/resize)
    batch=8,              # Batch size (adjust based on GPU memory)
    device=0,             # GPU ID (use 'cpu' if no GPU)
    name='rtdetr_defect', # Run name (saves to runs/detect/rtdetr_defect/)
    plots=True            # Generate training plots
)

# After training, results include metrics (view in runs/detect/rtdetr_defect/results.csv)
print(results)