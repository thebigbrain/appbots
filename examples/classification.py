from torch import Tensor
from torchvision.io import read_image, ImageReadMode
from torchvision.models import resnet50, ResNet50_Weights

from appbots.core.utils import get_path

# Step 1: Initialize model with the best available weights
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
model.eval()

# Step 2: Initialize the inference transforms
preprocess = weights.transforms()


def get_image_tensor(img_path: str) -> Tensor:
    return read_image(get_path(img_path), mode=ImageReadMode.RGB)


def predict(img_path: str):
    img_tensor = get_image_tensor(img_path)
    # Step 3: Apply inference preprocessing transforms
    batch = preprocess(img_tensor).unsqueeze(0)

    # Step 4: Use the model and print the predicted category
    prediction = model(batch).squeeze(0).softmax(0)
    class_id = prediction.argmax().item()
    score = prediction[class_id].item()
    category_name = weights.meta["categories"][class_id]
    print(f"{category_name}: {100 * score:.1f}%")


if __name__ == "__main__":
    predict("assets/test.png")
    predict("assets/bus.jpg")
