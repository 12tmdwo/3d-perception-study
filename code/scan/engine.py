from ultralytics import YOLO

model = YOLO("yolo26n-seg.pt")

model.export(
    format="engine",
    imgsz=640,
    device=0,
    simplify=False
)