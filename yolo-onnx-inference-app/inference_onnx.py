import onnxruntime as ort
import numpy as np
import os

# Define the path to the ONNX model
# Assumes the script is run from the root directory containing 'checkpoints'
onnx_model_path = 'checkpoints/yoloItem.onnx'

# Check if the model file exists
if not os.path.exists(onnx_model_path):
    print(f"Error: ONNX model not found at {onnx_model_path}")
    exit(1)

try:
    # 1. Load the ONNX model
    print(f"Loading ONNX model from {onnx_model_path}...")
    # Consider adding more providers like 'CUDAExecutionProvider' if GPU is available and needed
    session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
    print("ONNX model loaded successfully.")

    # Get model input details
    input_cfg = session.get_inputs()[0]
    input_name = input_cfg.name
    input_shape = input_cfg.shape # Example: [1, 3, 'height', 'width'] or [1, 3, 640, 640]
    print(f"Model Input Name: {input_name}")
    print(f"Model Input Shape reported by ONNX: {input_shape}")

    # Determine input dimensions (handle dynamic axes like 'batch', 'height', 'width')
    # Use defaults (e.g., 640x640) if dimensions are not fixed integers in the model definition
    batch_size = 1 if not isinstance(input_shape[0], int) or input_shape[0] <= 0 else input_shape[0]
    channels = 3 if not isinstance(input_shape[1], int) or input_shape[1] <= 0 else input_shape[1]
    height = 640 if not isinstance(input_shape[2], int) or input_shape[2] <= 0 else input_shape[2] # Default/common value
    width = 640 if not isinstance(input_shape[3], int) or input_shape[3] <= 0 else input_shape[3] # Default/common value

    final_input_shape = (batch_size, channels, height, width)
    print(f"Using Input Shape for Inference: {final_input_shape}")

    # Create dummy input data
    # NOTE: Real inference requires actual image data, preprocessed appropriately
    # (e.g., resized, normalized 0-1, converted HWC to CHW, BGR to RGB if needed)
    # Input type should match model expectation (often float32)
    dummy_input_data = np.random.rand(*final_input_shape).astype(np.float32)
    print(f"Created dummy input data with shape: {dummy_input_data.shape} and dtype: {dummy_input_data.dtype}")

    # 2. Run inference
    print("Running inference with dummy data...")
    # The first argument 'None' means fetch all outputs
    outputs = session.run(None, {input_name: dummy_input_data})
    print("Inference completed.")

    # Print basic information about the outputs
    print(f"Number of outputs generated: {len(outputs)}")
    for i, output in enumerate(outputs):
        print(f"Output {i} shape: {output.shape}")
        print(f"Output {i} dtype: {output.dtype}")
        # Example: Accessing the first few elements of the first output
        # print(f"Output {i} data (first 5 elements): {output.flatten()[:5]}")

except ort.OrtLoadError as e:
    print(f"Error loading the ONNX model: {e}")
    print("Ensure the model file is valid and all necessary operators are supported by the ONNX Runtime build.")
    exit(1)
except Exception as e:
    print(f"An error occurred during inference: {e}")
    exit(1)

print("Script finished.")
