import torch
import torchvision
import cv2

# 加载预训练模型
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# 读取图片
img = 'assets/test.png'
results = model(img)

# 显示结果
results.print()
results.show()
