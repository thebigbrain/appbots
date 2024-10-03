from ultralytics import YOLO

from appbots.core.utils import get_assets, get_model_path
from appbots.core.yolo.config import get_yolo

# Load a model
model = YOLO(get_model_path("yolov5nu.pt"))  # pretrained YOLOv8n model


def train():
    # Train the model
    r = model.train(data=get_yolo("coco"), epochs=100, imgsz=640)
    print(r)


def predict():
    # Run batched inference on a list of images
    results = model([get_assets("test2.jpg")])  # return a list of Results objects
    # Process results list
    for r in results:
        print(f"Detected {len(r)} objects in image")
        r.show()  # display to screen


if __name__ == '__main__':
    # train()
    predict()
    pass
