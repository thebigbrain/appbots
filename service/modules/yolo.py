from flask import request
from ultralytics import YOLO

from appbots.core.utils import get_model_path
from appbots.core.yolo.config import get_yolo

yolo_model = YOLO(get_model_path("yolov5.pt"), task="detect")


def predict(image_url: str):
    results = yolo_model.predict(source=image_url, conf=0.25)  # return a list of Results objects
    # Process results list
    r = results[0]
    print(f"Detected {len(r)} objects in image")
    return r.boxes or []


def train():
    # Train the model
    args = request.args
    epochs = args.get('epochs', 100)
    imgsz = args.get('imgsz', 640)
    r = yolo_model.train(data=get_yolo("coco"), epochs=epochs, imgsz=imgsz)
    print(r)

    name = yolo_model.model_name.replace(".yaml", ".pt")
    yolo_model.save(name)
