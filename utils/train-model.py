from ultralytics import YOLO

# Load the YOLOv11 model (segmentation version)
model = YOLO("../yolo11n-seg.pt")

# Train the model with the dataset
model.train(data='../datasets/card-detect.v11i.yolov11/data.yaml', epochs=60, imgsz=640)
