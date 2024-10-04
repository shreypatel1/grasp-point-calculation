from ultralytics import YOLO

# source = "datasets/card-detect.v11i.yolov11/train/images/IMG_20230907_111450_jpg.rf.9f8ab2f17a632a54b62418e0506319d6.jpg"
source = 0
model = YOLO("../runs/segment/100epochs/weights/best.pt")

results = model(source, show=True, conf=0.94, save=True)