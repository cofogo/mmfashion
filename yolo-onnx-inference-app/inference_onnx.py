import onnxruntime as ort
import numpy as np
import os
from flask import Flask, request, jsonify
from PIL import Image
import io

# --- Global Model Loading ---
MODEL_PATH = 'yoloItem.onnx'
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

        # Convert output numpy arrays to lists for JSON serialization
        output_lists = [output.tolist() for output in outputs]

        # Return the raw model output
        # Modify this part if specific post-processing is needed
        return jsonify({"model_output": output_lists})

    except Exception as e:
        app.logger.error(f"Error processing image: {str(e)}")
        return jsonify({"error": "Failed to process image"}), 500

if __name__ == "__main__":
    if session is None:
        print("ERROR: Cannot start server because the model failed to load.")
    else:
        print("Starting Flask server...")
        # Use host='0.0.0.0' to make it accessible externally, e.g., within Docker
        app.run(host="0.0.0.0", port=5000)
