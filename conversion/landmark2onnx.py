from __future__ import division

import torch

from mmcv import Config
from mmcv.runner import load_checkpoint
from mmfashion.models import build_landmark_detector

# === CONFIGURATION ===
CONFIG_FILE = 'configs/landmark_detect/landmark_detect_resnet.py'
CHECKPOINT_FILE = 'checkpoints/resnetLandmarkLatest.pth'
ONNX_OUTPUT_FILE = 'onnx-inference-app/onnxmodels/landmark.onnx'

# === BUILD MODEL ===
cfg = Config.fromfile(CONFIG_FILE)
model = build_landmark_detector(cfg.model)
model.eval()
print('Model built.')

# === LOAD CHECKPOINT INTO BASE MODEL (NOT WRAPPER) ===
load_checkpoint(model, CHECKPOINT_FILE, map_location='cpu')
print(f'Loaded checkpoint from: {CHECKPOINT_FILE}')

# === WRAP MODEL FOR ONNX EXPORT ===
class LandmarkWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        result = self.model(x, return_loss=False)
        if result is None:
            raise ValueError("Model returned None. Check if forward supports return_loss=False.")
        pred_vis, pred_lm = result
        return pred_vis, pred_lm

wrapped_model = LandmarkWrapper(model)
wrapped_model.eval()

# === CREATE DUMMY INPUT ===
img_tensor = torch.randn(1, 3, 224, 224, requires_grad=False)

# === DRY RUN FOR DEBUG ===
with torch.no_grad():
    pred_vis, pred_lm = wrapped_model(img_tensor)

print('pred_vis shape:', pred_vis.shape)
print('pred_lm shape:', pred_lm.shape)

# === EXPORT TO ONNX ===
input_names = ['input']
output_names = ['pred_vis', 'pred_lm']
dynamic_axes=   {'input': {0: 'batch_size'},
                 'pred_vis': {0: 'batch_size'},
                 'pred_lm': {0: 'batch_size'}}

torch.onnx.export(
    wrapped_model,
    img_tensor,
    ONNX_OUTPUT_FILE,
    input_names=input_names,
    output_names=output_names,
    dynamic_axes=dynamic_axes,
    opset_version=20,
    do_constant_folding=True,
    training=torch.onnx.TrainingMode.EVAL,
    verbose=False
)

print(f"Exported model to: {ONNX_OUTPUT_FILE}")
