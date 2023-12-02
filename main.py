from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.info()

results = model.train(data="D:\Work\Main_IS_Project\data.yaml", epochs= 100, imgsz=640, patience=200)
metrics = model.val()