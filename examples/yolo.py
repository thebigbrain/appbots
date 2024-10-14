from ultralytics import YOLO
from ultralytics.engine.model import Model

from appbots.core.utils import get_assets, get_model_path
from appbots.core.yolo.config import get_yolo


def train(model: Model, autosave=True):
    # Train the model
    r = model.train(data=get_yolo("coco"), epochs=100, imgsz=640)
    print(r)

    if autosave:
        name = model.model_name.replace(".yaml", ".pt")
        model.save(name)


def predict(model: Model):
    # Run batched inference on a list of images
    results = model.predict(source=get_assets("test5.jpg"), conf=0.25)  # return a list of Results objects
    # Process results list
    for r in results:
        print(f"Detected {len(r)} objects in image")
        r.show()


def create_yolo_model(name: str):
    model = YOLO(get_model_path(f"{name}.yaml"), task="detect", verbose=True)

    return model


if __name__ == '__main__':
    # yolo_model = create_yolo_model("yolov5")

    # Load a model
    yolo_model = YOLO(get_model_path("yolov5.pt"), task="detect")
    # train(yolo_model)
    predict(yolo_model)
