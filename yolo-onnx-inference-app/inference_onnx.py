import onnxruntime as ort
import numpy as np
import os
import torch
from flask import Flask, request, jsonify
from PIL import Image
import io
import ultralytics.utils

# Attempt to import Ultralytics components
try:
    from ultralytics.utils import ops
    ULTRALYTICS_AVAILABLE = True
    print("Ultralytics components loaded successfully.")
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("Warning: Ultralytics library not found or not fully installed. NMS post-processing will be unavailable.")
    # Define a dummy function if ultralytics is not available
    class ops:
        @staticmethod
        def non_max_suppression(*args, **kwargs):
            raise ImportError("Ultralytics ops.non_max_suppression function is not available.")


# --- Global Model Loading ---
MODEL_PATH = 'yoloItem.onnx' # Corrected path assuming it's in checkpoints
session = None
input_name = None
model_input_height = 224 # Default
model_input_width = 224 # Default

if not os.path.exists(MODEL_PATH):
    print(f"FATAL ERROR: ONNX model not found at {MODEL_PATH}")
    exit(1)

try:
    print(f"Loading ONNX model from {MODEL_PATH}...")
    # Consider adding more providers like 'CUDAExecutionProvider' if available
    session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
    print("ONNX model loaded successfully.")

    # Get model input details
    input_cfg = session.get_inputs()[0]
    input_name = input_cfg.name
    input_shape = input_cfg.shape # Example: [1, 3, 'height', 'width'] or [1, 3, 640, 640]
    print(f"Model Input Name: {input_name}")
    print(f"Model Input Shape reported by ONNX: {input_shape}")

    # Determine input dimensions, overriding defaults if possible
    if isinstance(input_shape[2], int) and input_shape[2] > 0:
        model_input_height = input_shape[2]
    if isinstance(input_shape[3], int) and input_shape[3] > 0:
        model_input_width = input_shape[3]
    print(f"Using Model Input Size (H, W): ({model_input_height}, {model_input_width})")

except Exception as e:
    print(f"FATAL ERROR: Could not load ONNX model or get details: {e}")
    # Optionally exit or handle appropriately if the server can't start
    session = None # Ensure session is None if loading failed

# --- Flask App ---
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    if session is None:
        return jsonify({"error": "Model not loaded"}), 500

    if "image" not in request.files:
        return jsonify({"error": "No image file uploaded"}), 400

    image_file = request.files["image"]

    try:
        # Load image using PIL
        image_bytes = image_file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_shape = image.size

        # Preprocess image
        # 1. Resize
        image_resized = image.resize((model_input_width, model_input_height))
        # 2. Convert to numpy array HWC
        img_np = np.array(image_resized)
        # 3. Normalize to 0-1
        img_normalized = img_np.astype(np.float32) / 255.0
        # 4. Transpose HWC to CHW (assuming model expects CHW)
        img_chw = np.transpose(img_normalized, (2, 0, 1))
        # 5. Add batch dimension (BCHW)
        input_data = np.expand_dims(img_chw, axis=0)

        # Run inference
        outputs = session.run(None, {input_name: input_data})

        # Assuming the primary output for detection models is the first one
        raw_output_tensor = torch.from_numpy(outputs[0])

        # Apply Non-Maximum Suppression (NMS)
        if not ULTRALYTICS_AVAILABLE:
             return jsonify({"error": "Ultralytics library not available for NMS post-processing"}), 500

        # Use default thresholds or make them configurable
        conf_threshold = 0.25
        iou_threshold = 0.45
        detections = ops.non_max_suppression(
            raw_output_tensor,
            conf_thres=conf_threshold,
            iou_thres=iou_threshold,
            classes=None, # Filter by specific classes if needed
            agnostic=False, # Set True for class-agnostic NMS
            max_det=300 # Maximum number of detections
        )[0] # Get detections for the first (and only) image in the batch

        # Convert NMS results tensor to list for JSON serialization
        nms_results_list = detections.cpu().numpy().tolist()

        scaled_results = []

        for detection in nms_results_list:
            x1, y1, x2, y2 = detection[:4]
            # Scale coordinates
            x1 = int(x1 / model_input_width * image_shape[0])
            y1 = int(y1 / model_input_height * image_shape[1])
            x2 = int(x2 / model_input_width * image_shape[0])
            y2 = int(y2 / model_input_height * image_shape[1])

            # Append scaled detection with optional confidence/class
            scaled_results.append([x1, y1, x2, y2] + detection[4:])

        # Return the NMS results
        return jsonify({"detections": scaled_results}), 200

    except ImportError as e:
        app.logger.error(f"Import error during processing: {str(e)}")
        return jsonify({"error": "Server configuration error: Missing required library"}), 500
    except Exception as e:
        app.logger.error(f"Error processing image or applying NMS: {str(e)}")
        return jsonify({"error": "Failed to process image"}), 500

if __name__ == "__main__":
    if session is None:
        print("ERROR: Cannot start server because the model failed to load.")
    else:
        print("Starting Flask server...")
        # Use host='0.0.0.0' to make it accessible externally, e.g., within Docker
        app.run(host="0.0.0.0", port=5000)
