from ultralytics import YOLO
import shutil
import os

# load model
model = YOLO('checkpoints/best.pt', task='detect')
model.export(format='onnx', dynamic=True, simplify=True, opset=11)

model_path='onnx-inference-app/onnxmodels/'
model_name='yolo.onnx'

# move and rename from 'checkpoints/best.onnx' to 'onnx-inference-app/onnxmodels/yolo.onnx'
if not os.path.exists(model_path):
    os.makedirs(model_path)
shutil.move('checkpoints/best.onnx', model_path + model_name)
print(f"Exported model to: {model_path + model_name}")