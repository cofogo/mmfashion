from __future__ import division

import torch
import os
import numpy as np

from mmcv import Config
from mmcv.runner import load_checkpoint
from mmfashion.models import build_predictor

# === CONFIGURATION ===
CHECKPOINT_FILE = 'checkpoints/resnetAttrLatest.pth'
CONFIG_FILE = 'configs/attribute_predict_coarse/roi_predictor_resnet_attr.py'
ONNX_OUTPUT_FILE = 'onnx-inference-app/onnxmodels/attributeLayers/attributes.onnx'

# === BUILD MODEL ===
cfg = Config.fromfile(CONFIG_FILE)
model = build_predictor(cfg.model)
model.eval()
print('Model built.')

# === LOAD CHECKPOINT INTO BASE MODEL (NOT WRAPPER) ===
load_checkpoint(model, CHECKPOINT_FILE, map_location='cpu')
print(f'Loaded checkpoint from: {CHECKPOINT_FILE}')

# === WRAP MODEL FOR ONNX EXPORT ===
class AttributeWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, landmarks):
        attr_prob = self.model(x, attr=None, landmark=landmarks, return_loss=False)
        if attr_prob is None:
            raise ValueError("Model returned None. Check if forward supports return_loss=False.")
        return attr_prob

wrapped_model = AttributeWrapper(model)
wrapped_model.eval()

# === CREATE DUMMY INPUT ===
img_tensor = torch.randn(1, 3, 224, 224, requires_grad=False)
landmark_tensor = torch.randn(1, 16, requires_grad=False)

# === DRY RUN FOR DEBUG ===
with torch.no_grad():
    attr_prob = wrapped_model(img_tensor, landmark_tensor)

print('attr_prob shape:', attr_prob.shape)

# === EXPORT TO ONNX ===
input_names = ['image', 'landmarks']
output_names = ['attr_prob']
dynamic_axes=   {'image': {0: 'batch_size'},
                 'landmarks': {0: 'batch_size'},
                 'attr_prob': {0: 'batch_size'}}

torch.onnx.export(
    wrapped_model,
    (img_tensor, landmark_tensor),
    ONNX_OUTPUT_FILE,
    input_names=input_names,
    output_names=output_names,
    dynamic_axes=dynamic_axes,
    opset_version=20,
    do_constant_folding=True
)

print(f"Exported model to: {ONNX_OUTPUT_FILE}")
