import os
import io
import numpy as np
import torch
import onnxruntime as ort
from flask import Flask, request, jsonify
from PIL import Image
from ultralytics.utils import ops


import torchvision.transforms as transforms # For landmark preprocessing normalization

# --- Configuration ---
YOLO_MODEL_PATH = 'onnx-inference-app/yoloItem.onnx'
LANDMARK_MODEL_PATH = 'landmark.onnx' # Added landmark model path
DEFAULT_YOLO_INPUT_SIZE = (224, 224)
DEFAULT_LANDMARK_INPUT_SIZE = (224, 224) # Default for landmark model
CROP_BUFFER = 10 # Pixels to add around the bounding box for cropping
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
MAX_DETECTIONS = 300


# --- Model Loader ---
def load_model(path, default_size):
    if not os.path.exists(path):
        raise FileNotFoundError(f"ONNX model not found at {path}")

    print(f"Loading ONNX model from {path}...")
    session = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    input_cfg = session.get_inputs()[0]
    input_shape = input_cfg.shape
    input_name = input_cfg.name

    height, width = default_size
    # Try to get fixed height/width from model definition
    if len(input_shape) == 4: # Check if shape has 4 dimensions (B, C, H, W)
        if isinstance(input_shape[2], int) and input_shape[2] > 0:
            height = input_shape[2]
        if isinstance(input_shape[3], int) and input_shape[3] > 0:
            width = input_shape[3]
    else:
        print(f"Warning: Model input shape {input_shape} is not standard BCHW. Using default size {default_size}.")


    print(f"Model loaded. Input: {input_name}, Shape: {input_shape}, Using Size: {height}x{width}")
    return session, input_name, (height, width)


# --- Preprocessing ---
# Preprocessing for YOLO model
def preprocess_image_yolo(image, size):
    # size is (height, width)
    image = image.resize((size[1], size[0])).convert("RGB") # PIL resize takes (width, height)
    image_np = np.array(image).astype(np.float32) / 255.0
    image_chw = np.transpose(image_np, (2, 0, 1)) # HWC to CHW
    return np.expand_dims(image_chw, axis=0) # Add batch dimension

# Preprocessing for Landmark model (assumes ImageNet normalization)
def preprocess_image_landmark(pil_image, size):
    # size is (height, width)
    img_norm = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    normalize = transforms.Normalize(mean=img_norm['mean'], std=img_norm['std'])
    transform = transforms.Compose([
        transforms.Resize(size), # Resize takes (h, w)
        transforms.ToTensor(), # Converts to [0, 1] and CHW
        normalize,
    ])
    img_tensor = transform(pil_image).unsqueeze(0) # Add batch dimension
    return img_tensor.cpu().numpy() # ONNX runtime expects numpy


# --- Postprocessing ---
# Scale YOLO detections from model input size to original image size
def scale_yolo_detections(detections, model_size, original_size):
    scaled = []
    # model_size is (height, width), original_size is (width, height)
    in_h, in_w = model_size
    orig_w, orig_h = original_size

    for det in detections:
        # det format: [x1, y1, x2, y2, conf, cls]
        x1, y1, x2, y2 = det[:4]
        # Scale coordinates
        x1_s = int(x1 / in_w * orig_w)
        y1_s = int(y1 / in_h * orig_h)
        x2_s = int(x2 / in_w * orig_w)
        y2_s = int(y2 / in_h * orig_h)
        # Append scaled box and original conf/class
        scaled.append([x1_s, y1_s, x2_s, y2_s] + det[4:].tolist())

    return scaled

# Scale Landmark coordinates from model input space to original image space
def scale_landmarks(landmarks_norm, landmark_model_size, crop_size, crop_origin_buffered):
    # landmarks_norm: list of [x, y] normalized to landmark_model_size (e.g., 0-224)
    # landmark_model_size: (height, width) of the landmark model input
    # crop_size: (width, height) of the actual cropped image fed to landmark model
    # crop_origin_buffered: (x1, y1) of the crop in the original image (with buffer)

    lm_h, lm_w = landmark_model_size
    crop_w, crop_h = crop_size
    bx1, by1 = crop_origin_buffered

    scaled_landmarks = []
    for lx_norm, ly_norm in landmarks_norm:
        # Scale from landmark model input space (e.g., 224x224) to crop space
        lx_crop = lx_norm * crop_w / lm_w
        ly_crop = ly_norm * crop_h / lm_h

        # Translate to original image space by adding crop origin
        Lx_orig = int(round(lx_crop + bx1))
        Ly_orig = int(round(ly_crop + by1))
        scaled_landmarks.append([Lx_orig, Ly_orig])

    return scaled_landmarks


# --- Initialize Models ---
try:
    yolo_session, yolo_input_name, yolo_input_size = load_model(YOLO_MODEL_PATH, DEFAULT_YOLO_INPUT_SIZE)
except Exception as e:
    print(f"ERROR loading YOLO model: {e}")
    yolo_session = None

try:
    landmark_session, landmark_input_name, landmark_input_size = load_model(LANDMARK_MODEL_PATH, DEFAULT_LANDMARK_INPUT_SIZE)
except Exception as e:
    print(f"ERROR loading Landmark model: {e}")
    landmark_session = None


# --- Flask App ---
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    if yolo_session is None or landmark_session is None:
        return jsonify({"error": "One or more models not loaded"}), 500

    if "image" not in request.files:
        return jsonify({"error": "No image file uploaded"}), 400

    try:
        image_bytes = request.files["image"].read()
        original_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        original_size = original_image.size # (width, height)
        img_w, img_h = original_size

        # 1. Run YOLO Detection
        yolo_input_tensor = preprocess_image_yolo(original_image, yolo_input_size)
        yolo_outputs = yolo_session.run(None, {yolo_input_name: yolo_input_tensor})
        yolo_raw_output = torch.from_numpy(yolo_outputs[0])

        yolo_detections_nms = ops.non_max_suppression(
            yolo_raw_output,
            conf_thres=CONF_THRESHOLD,
            iou_thres=IOU_THRESHOLD,
            classes=None, # Can filter for specific classes here if needed
            agnostic=False,
            max_det=MAX_DETECTIONS
        )[0] # Get detections for the first image

        if yolo_detections_nms is None or len(yolo_detections_nms) == 0:
            return jsonify({"results": []}), 200 # Return empty list if no detections

        # Scale YOLO boxes to original image coordinates
        scaled_yolo_results = scale_yolo_detections(
            yolo_detections_nms.cpu().numpy(),
            yolo_input_size, # (h, w)
            original_size # (w, h)
        )

        # 2. Process each detection: Crop and run Landmark Model
        final_results = []
        for detection in scaled_yolo_results:
            # detection format: [x1, y1, x2, y2, conf, cls] in original image coords
            x1, y1, x2, y2 = map(int, detection[:4])
            confidence = detection[4]
            class_id = int(detection[5])

            # Create buffered crop coordinates, clamped to image bounds
            bx1 = max(0, x1 - CROP_BUFFER)
            by1 = max(0, y1 - CROP_BUFFER)
            bx2 = min(img_w, x2 + CROP_BUFFER)
            by2 = min(img_h, y2 + CROP_BUFFER)

            if bx1 >= bx2 or by1 >= by2:
                print(f"Warning: Invalid crop coordinates after buffer/clamp for box {detection[:4]}. Skipping.")
                continue

            # Crop the original image using PIL
            try:
                cropped_image = original_image.crop((bx1, by1, bx2, by2))
                crop_size = cropped_image.size # (width, height)
            except Exception as e:
                app.logger.error(f"Error cropping image for box {detection[:4]}: {e}")
                continue

            if crop_size[0] == 0 or crop_size[1] == 0:
                print(f"Warning: Zero-size crop for box {detection[:4]}. Skipping.")
                continue

            # Preprocess crop for landmark model
            landmark_input_tensor = preprocess_image_landmark(cropped_image, landmark_input_size)

            # Run landmark inference
            try:
                landmark_outputs = landmark_session.run(None, {landmark_input_name: landmark_input_tensor})
                # Assuming the first output contains landmark coordinates [batch, num_landmarks, 2]
                # And they are normalized to the landmark model input size (e.g., 0-224)
                landmarks_output = landmark_outputs[0][0] # Get landmarks for the first (only) image in batch
            except Exception as e:
                app.logger.error(f"Error running landmark inference for box {detection[:4]}: {e}")
                landmarks_output = [] # Assign empty list on error

            # Scale landmarks back to original image coordinates
            scaled_landmarks = scale_landmarks(
                landmarks_output,         # List of [x, y] normalized to landmark model input
                landmark_input_size,      # (height, width) of landmark model input
                crop_size,                # (width, height) of the cropped image
                (bx1, by1)                # Top-left corner of the buffered crop in original image
            )

            final_results.append({
                "bbox": [x1, y1, x2, y2], # Original YOLO box (unbuffered)
                "confidence": confidence,
                "class_id": class_id,
                "landmarks": scaled_landmarks # Scaled to original image coords
            })

        return jsonify({"results": final_results}), 200

    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": "Failed to process image"}), 500


if __name__ == "__main__":
    if yolo_session and landmark_session:
        print("Starting Flask server with YOLO and Landmark models...")
        app.run(host="0.0.0.0", port=5000)
    else:
        print("FATAL: Cannot start server because one or more models failed to load.")
        # width = input_shape[3]

    # print(f"Model loaded. Input: {input_name}, Shape: {input_shape}, Size: {height}x{width}")
    # return session, input_name, (height, width)


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


# --- Postprocessing ---
# Scale YOLO detections from model input size to original image size
def scale_yolo_detections(detections, model_size, original_size):
    scaled = []
    # model_size is (height, width), original_size is (width, height)
    in_h, in_w = model_size
    orig_w, orig_h = original_size

    for det in detections:
        # det format: [x1, y1, x2, y2, conf, cls]
        x1, y1, x2, y2 = det[:4]
        # Scale coordinates
        x1_s = int(x1 / in_w * orig_w)
        y1_s = int(y1 / in_h * orig_h)
        x2_s = int(x2 / in_w * orig_w)
        y2_s = int(y2 / in_h * orig_h)
        # Append scaled box and original conf/class
        scaled.append([x1_s, y1_s, x2_s, y2_s] + det[4:].tolist())

    return scaled

# Scale Landmark coordinates from model input space to original image space
def scale_landmarks(landmarks_norm, landmark_model_size, crop_size, crop_origin_buffered):
    # landmarks_norm: list of [x, y] normalized to landmark_model_size (e.g., 0-224)
    # landmark_model_size: (height, width) of the landmark model input
    # crop_size: (width, height) of the actual cropped image fed to landmark model
    # crop_origin_buffered: (x1, y1) of the crop in the original image (with buffer)

    lm_h, lm_w = landmark_model_size
    crop_w, crop_h = crop_size
    bx1, by1 = crop_origin_buffered

    scaled_landmarks = []
    for lx_norm, ly_norm in landmarks_norm:
        # Scale from landmark model input space (e.g., 224x224) to crop space
        lx_crop = lx_norm * crop_w / lm_w
        ly_crop = ly_norm * crop_h / lm_h

        # Translate to original image space by adding crop origin
        Lx_orig = int(round(lx_crop + bx1))
        Ly_orig = int(round(ly_crop + by1))
        scaled_landmarks.append([Lx_orig, Ly_orig])

    return scaled_landmarks


# --- Initialize Models ---
try:
    yolo_session, yolo_input_name, yolo_input_size = load_model(YOLO_MODEL_PATH, DEFAULT_YOLO_INPUT_SIZE)
except Exception as e:
    print(f"ERROR loading YOLO model: {e}")
    yolo_session = None

try:
    landmark_session, landmark_input_name, landmark_input_size = load_model(LANDMARK_MODEL_PATH, DEFAULT_LANDMARK_INPUT_SIZE)
except Exception as e:
    print(f"ERROR loading Landmark model: {e}")
    landmark_session = None


# --- Flask App ---
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    if yolo_session is None or landmark_session is None:
        return jsonify({"error": "One or more models not loaded"}), 500

    if "image" not in request.files:
        return jsonify({"error": "No image file uploaded"}), 400

    try:
        image = Image.open(io.BytesIO(request.files["image"].read()))
        original_size = image.size
        img_w, img_h = original_size

        # 1. Run YOLO Detection
        yolo_input_tensor = preprocess_image_yolo(image, yolo_input_size)
        yolo_outputs = yolo_session.run(None, {yolo_input_name: yolo_input_tensor})
        yolo_raw_output = torch.from_numpy(yolo_outputs[0])

        yolo_detections_nms = ops.non_max_suppression(
            yolo_raw_output,
            conf_thres=CONF_THRESHOLD,
            iou_thres=IOU_THRESHOLD,
            classes=None,
            agnostic=False,
            max_det=MAX_DETECTIONS
        )[0] # Get detections for the first image

        if yolo_detections_nms is None or len(yolo_detections_nms) == 0:
            return jsonify({"results": []}), 200 # Return empty list if no detections

        # Scale YOLO boxes to original image coordinates
        scaled_yolo_results = scale_yolo_detections(
            yolo_detections_nms.cpu().numpy(),
            yolo_input_size, # (h, w)
            original_size # (w, h)
        )

        # 2. Process each detection: Crop and run Landmark Model
        final_results = []
        for detection in scaled_yolo_results:
            # detection format: [x1, y1, x2, y2, conf, cls] in original image coords
            x1, y1, x2, y2 = map(int, detection[:4])
            confidence = detection[4]
            class_id = int(detection[5])

            # Create buffered crop coordinates, clamped to image bounds
            bx1 = max(0, x1 - CROP_BUFFER)
            by1 = max(0, y1 - CROP_BUFFER)
            bx2 = min(img_w, x2 + CROP_BUFFER)
            by2 = min(img_h, y2 + CROP_BUFFER)

            if bx1 >= bx2 or by1 >= by2:
                print(f"Warning: Invalid crop coordinates after buffer/clamp for box {detection[:4]}. Skipping.")
                continue

            # Crop the original image using PIL
            try:
                cropped_image = image.crop((bx1, by1, bx2, by2))
                crop_size = cropped_image.size # (width, height)
            except Exception as e:
                app.logger.error(f"Error cropping image for box {detection[:4]}: {e}")
                continue

            if crop_size[0] == 0 or crop_size[1] == 0:
                print(f"Warning: Zero-size crop for box {detection[:4]}. Skipping.")
                continue

            # Preprocess crop for landmark model
            landmark_input_tensor = preprocess_image_landmark(cropped_image, landmark_input_size)

            # Run landmark inference
            try:
                landmark_outputs = landmark_session.run(None, {landmark_input_name: landmark_input_tensor})
                # Assuming the first output contains landmark coordinates [batch, num_landmarks, 2]
                # And they are normalized to the landmark model input size (e.g., 0-224)
                landmarks_output = landmark_outputs[0][0] # Get landmarks for the first (only) image in batch
            except Exception as e:
                app.logger.error(f"Error running landmark inference for box {detection[:4]}: {e}")
                landmarks_output = [] # Assign empty list on error

            # Scale landmarks back to original image coordinates
            scaled_landmarks = scale_landmarks(
                landmarks_output,         # List of [x, y] normalized to landmark model input
                landmark_input_size,      # (height, width) of landmark model input
                crop_size,                # (width, height) of the cropped image
                (bx1, by1)                # Top-left corner of the buffered crop in original image
            )

            final_results.append({
                "bbox": [x1, y1, x2, y2], # Original YOLO box (unbuffered)
                "confidence": confidence,
                "class_id": class_id,
                "landmarks": scaled_landmarks # Scaled to original image coords
            })

        return jsonify({"results": final_results}), 200

    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": "Failed to process image"}), 500


if __name__ == "__main__":
    if yolo_session and landmark_session:
        print("Starting Flask server with YOLO and Landmark models...")
        app.run(host="0.0.0.0", port=5000)
    else:
        print("FATAL: Cannot start server because one or more models failed to load.")
