from ultralytics import YOLO
import torch
import os
from PIL import Image

# load model
model = YOLO('checkpoints/best.pt', task='detect')
model.export(format='onnx', dynamic=True, simplify=True, opset=11)