import torch
from torchvision.models.detection.anchor_utils import AnchorGenerator

anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
anchor_generator = AnchorGenerator(
    sizes=anchor_sizes,
    aspect_ratios=aspect_ratios
)

if __name__ == "__main__":
    print(anchor_generator.cell_anchors)
