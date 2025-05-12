import os
import io
import numpy as np
import torch
import onnxruntime as ort
from flask import Flask, request, jsonify
from PIL import Image
from ultralytics.utils import ops


# --- Configuration ---
MODEL_PATH = 'yoloItem.onnx'
DEFAULT_INPUT_SIZE = (224, 224)
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
MAX_DETECTIONS = 300


# --- Model Loader ---
def load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"ONNX model not found at {path}")

    print(f"Loading ONNX model from {path}...")
    session = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    input_cfg = session.get_inputs()[0]
    input_shape = input_cfg.shape
    input_name = input_cfg.name

    height, width = DEFAULT_INPUT_SIZE
    if isinstance(input_shape[2], int):
        height = input_shape[2]
    if isinstance(input_shape[3], int):
        width = input_shape[3]

    print(f"Model loaded. Input: {input_name}, Shape: {input_shape}, Size: {height}x{width}")
    return session, input_name, (height, width)


# --- Preprocessing ---
def preprocess_image(image, size):
    image = image.resize(size).convert("RGB")
    image_np = np.array(image).astype(np.float32) / 255.0
    image_chw = np.transpose(image_np, (2, 0, 1))
    return np.expand_dims(image_chw, axis=0)


# --- Postprocessing ---
def scale_detections(detections, model_size, original_size):
    scaled = []
    in_w, in_h = model_size
    orig_w, orig_h = original_size

    for det in detections:
        x1, y1, x2, y2 = det[:4]
        x1 = int(x1 / in_w * orig_w)
        y1 = int(y1 / in_h * orig_h)
        x2 = int(x2 / in_w * orig_w)
        y2 = int(y2 / in_h * orig_h)
        scaled.append([x1, y1, x2, y2] + det[4:].tolist())

    return scaled


# --- Initialize Model ---
try:
    session, input_name, input_size = load_model(MODEL_PATH)
except Exception as e:
    print(f"ERROR: {e}")
    session = None


# --- Flask App ---
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    if session is None:
        return jsonify({"error": "Model not loaded"}), 500

    if "image" not in request.files:
        return jsonify({"error": "No image file uploaded"}), 400

    try:
        image = Image.open(io.BytesIO(request.files["image"].read()))
        original_size = image.size
        input_tensor = preprocess_image(image, (input_size[1], input_size[0]))

        outputs = session.run(None, {input_name: input_tensor})
        raw_output = torch.from_numpy(outputs[0])

        detections = ops.non_max_suppression(
            raw_output,
            conf_thres=CONF_THRESHOLD,
            iou_thres=IOU_THRESHOLD,
            classes=None,
            agnostic=False,
            max_det=MAX_DETECTIONS
        )[0]

        if detections is None or len(detections) == 0:
            return jsonify({"detections": []}), 200

        result = scale_detections(detections.cpu().numpy(), input_size[::-1], original_size)
        return jsonify({"detections": result}), 200

    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": "Failed to process image"}), 500


if __name__ == "__main__":
    if session:
        print("Starting Flask server...")
        app.run(host="0.0.0.0", port=5000)
    else:
        print("FATAL: Cannot start server because model failed to load.")
