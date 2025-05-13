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
YOLO_MODEL_PATH = 'onnxmodels/yolo.onnx'
LANDMARK_MODEL_PATH = 'onnxmodels/landmark.onnx' # Added landmark model path
CLASSIFICATION_MODEL_PATH = 'onnxmodels/category.onnx' # Added classification model path
DEFAULT_YOLO_INPUT_SIZE = (224, 224)
DEFAULT_LANDMARK_INPUT_SIZE = (224, 224) # Default for landmark model
DEFAULT_CLASSIFICATION_INPUT_SIZE = (224, 224) # Default for classification model
CROP_BUFFER = 10 # Pixels to add around the bounding box for cropping
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
MAX_DETECTIONS = 1 # Only process the first most confident detection
CLOTHING_CLASS_ID = 2 # Class ID for clothing in YOLO model


# --- Model Loader ---
def load_model(path, default_img_size):
    if not os.path.exists(path):
        raise FileNotFoundError(f"ONNX model not found at {path}")

    print(f"Loading ONNX model from {path}...")
    session = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    model_inputs_meta = session.get_inputs()
    
    input_details_list = []

    if path == CLASSIFICATION_MODEL_PATH:
        if len(model_inputs_meta) < 2:
            raise ValueError(f"Classification model at {path} is expected to have at least 2 inputs (image, landmarks), but found {len(model_inputs_meta)}.")
        
        # Input 0: Image (assumed)
        img_input_meta = model_inputs_meta[0]
        img_h, img_w = default_img_size
        if len(img_input_meta.shape) == 4: # BCHW
            if isinstance(img_input_meta.shape[2], int) and img_input_meta.shape[2] > 0: img_h = img_input_meta.shape[2]
            if isinstance(img_input_meta.shape[3], int) and img_input_meta.shape[3] > 0: img_w = img_input_meta.shape[3]
        else:
            print(f"Warning: Classification model image input '{img_input_meta.name}' shape {img_input_meta.shape} is not BCHW. Using default size {default_img_size}.")
        input_details_list.append({'name': img_input_meta.name, 'type': 'image', 'size': (img_h, img_w)})
        
        # Input 1: Landmarks (assumed)
        lm_input_meta = model_inputs_meta[1]
        # Ensure landmark shape is fully defined, e.g. (1, 16)
        lm_shape = []
        for dim_val in lm_input_meta.shape:
            if isinstance(dim_val, int) and dim_val > 0:
                lm_shape.append(dim_val)
            else: # Dynamic dimension or zero/negative, which is problematic for fixed landmark tensor
                raise ValueError(f"Landmark input '{lm_input_meta.name}' for classification model {path} has non-fixed dimensions: {lm_input_meta.shape}. Expected fixed shape (e.g., [1, 16]).")
        lm_shape_tuple = tuple(lm_shape)
        input_details_list.append({'name': lm_input_meta.name, 'type': 'landmarks', 'shape': lm_shape_tuple})
        
        print(f"Classification Model loaded. Image Input: '{input_details_list[0]['name']}' Size: {input_details_list[0]['size']}, Landmark Input: '{input_details_list[1]['name']}' Shape: {input_details_list[1]['shape']}")

    else: # For YOLO and Landmark models (single image input)
        if not model_inputs_meta:
            raise ValueError(f"Model at {path} has no inputs defined.")
        img_input_meta = model_inputs_meta[0]
        img_h, img_w = default_img_size
        if len(img_input_meta.shape) == 4: #BCHW
            if isinstance(img_input_meta.shape[2], int) and img_input_meta.shape[2] > 0: img_h = img_input_meta.shape[2]
            if isinstance(img_input_meta.shape[3], int) and img_input_meta.shape[3] > 0: img_w = img_input_meta.shape[3]
        else:
             print(f"Warning: Model input '{img_input_meta.name}' shape {img_input_meta.shape} is not BCHW. Using default size {default_img_size}.")
        input_details_list.append({'name': img_input_meta.name, 'type': 'image', 'size': (img_h, img_w)})
        print(f"Model loaded. Input: '{input_details_list[0]['name']}', Using Size: {input_details_list[0]['size']}")

    return session, input_details_list


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

# Preprocessing for Classification model (assumes ImageNet normalization)
def preprocess_image_classification(pil_image, size):
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

# Prepare landmark data for ONNX model input
def prepare_landmarks_for_onnx(landmarks_norm_array, target_shape):
    """
    Prepares normalized landmark coordinates for an ONNX model.
    Args:
        landmarks_norm_array (np.ndarray): Array of shape (num_landmarks, 2),
                                           with coordinates normalized (e.g., to 0-224 range).
        target_shape (tuple): The exact shape required by the ONNX model's landmark input,
                              e.g., (1, num_landmarks * 2).
    Returns:
        np.ndarray: Landmark data reshaped and typed for ONNX.
    """
    flat_landmarks = landmarks_norm_array.flatten().astype(np.float32)
    
    expected_elements = 1
    for dim in target_shape:
        expected_elements *= dim
    
    if flat_landmarks.shape[0] != expected_elements:
        raise ValueError(
            f"Mismatch in landmark data size. Expected {expected_elements} elements for shape {target_shape}, "
            f"but got {flat_landmarks.shape[0]} from landmarks array of shape {landmarks_norm_array.shape}."
        )
    return flat_landmarks.reshape(target_shape)


# --- Postprocessing ---
# Scale YOLO detections from model input size to original image size
def scale_yolo_detections(detection, model_size, original_size):
    # model_size is (height, width), original_size is (width, height)
    in_h, in_w = model_size
    orig_w, orig_h = original_size

    # det format: [x1, y1, x2, y2, conf, cls]
    x1, y1, x2, y2 = detection[:4]
    # Scale coordinates
    x1_s = int(x1 / in_w * orig_w)
    y1_s = int(y1 / in_h * orig_h)
    x2_s = int(x2 / in_w * orig_w)
    y2_s = int(y2 / in_h * orig_h)
    # Append scaled box and original conf/class
    return [x1_s, y1_s, x2_s, y2_s] + detection[4:].tolist()

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
yolo_session, landmark_session, classification_session = None, None, None
yolo_input_details, landmark_input_details, classification_input_details = None, None, None

try:
    yolo_session, yolo_input_details = load_model(YOLO_MODEL_PATH, DEFAULT_YOLO_INPUT_SIZE)
except Exception as e:
    print(f"ERROR loading YOLO model: {e}")

try:
    landmark_session, landmark_input_details = load_model(LANDMARK_MODEL_PATH, DEFAULT_LANDMARK_INPUT_SIZE)
except Exception as e:
    print(f"ERROR loading Landmark model: {e}")

try:
    classification_session, classification_input_details = load_model(CLASSIFICATION_MODEL_PATH, DEFAULT_CLASSIFICATION_INPUT_SIZE)
except Exception as e:
    print(f"ERROR loading Classification model: {e}")


# --- Flask App ---
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    if not all([yolo_session, landmark_session, classification_session,
                yolo_input_details, landmark_input_details, classification_input_details]):
        return jsonify({"error": "One or more models or their configurations not loaded"}), 500

    if "image" not in request.files:
        return jsonify({"error": "No image file uploaded"}), 400

    try:
        image_bytes = request.files["image"].read()
        original_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        original_size = original_image.size # (width, height)
        img_w, img_h = original_size

        # 1. Run YOLO Detection
        yolo_input_name = yolo_input_details[0]['name']
        yolo_model_input_size = yolo_input_details[0]['size'] # (h, w) for model
        yolo_input_tensor = preprocess_image_yolo(original_image, yolo_model_input_size)
        yolo_outputs = yolo_session.run(None, {yolo_input_name: yolo_input_tensor})
        yolo_raw_output = torch.from_numpy(yolo_outputs[0])

        yolo_detections_nms = ops.non_max_suppression(
            yolo_raw_output,
            conf_thres=CONF_THRESHOLD,
            iou_thres=IOU_THRESHOLD,
            classes=CLOTHING_CLASS_ID,
            agnostic=False,
            max_det=MAX_DETECTIONS
        )[0] # Get detections for the first image

        if yolo_detections_nms is None or len(yolo_detections_nms) == 0:
            return jsonify({"results": []}), 200 # Return empty list if no detections
        
        assert len(yolo_detections_nms) == MAX_DETECTIONS, "Unexpected number of detections. Expected 1."

        yolo_detections_nms = yolo_detections_nms[0] # Get the first detection

        # Scale YOLO boxes to original image coordinates
        detection = scale_yolo_detections(
            yolo_detections_nms.cpu().numpy(),
            yolo_model_input_size, # (h, w) of yolo model input
            original_size # (w, h) of original image
        )

        # 2. Process each detection: Crop and run Landmark Model
        # detection format: [x1, y1, x2, y2, conf, cls] in original image coords
        x1, y1, x2, y2 = map(int, detection[:4])
        confidence = detection[4]
        class_id = int(detection[5])
        assert class_id == CLOTHING_CLASS_ID, "Class ID mismatch. Expected clothing class."

        # Create buffered crop coordinates, clamped to image bounds
        bx1 = max(0, x1 - CROP_BUFFER)
        by1 = max(0, y1 - CROP_BUFFER)
        bx2 = min(img_w, x2 + CROP_BUFFER)
        by2 = min(img_h, y2 + CROP_BUFFER)

        if bx1 >= bx2 or by1 >= by2:
            print(f"Warning: Invalid crop coordinates after buffer/clamp for box {detection[:4]}. Skipping.")
            return jsonify({"error": "Invalid crop coordinates"}), 400

        # Crop the original image using PIL
        try:
            cropped_image = original_image.crop((bx1, by1, bx2, by2))
            crop_size = cropped_image.size # (width, height)
        except Exception as e:
            app.logger.error(f"Error cropping image for box {detection[:4]}: {e}")
            return jsonify({"error": "Failed to crop image"}), 400

        if crop_size[0] == 0 or crop_size[1] == 0:
            print(f"Warning: Zero-size crop for box {detection[:4]}. Skipping.")
            return jsonify({"error": "Zero-size crop"}), 400

        # Preprocess crop for landmark model
        landmark_input_name = landmark_input_details[0]['name']
        landmark_model_input_size = landmark_input_details[0]['size'] # (h,w) for landmark model
        landmark_input_tensor = preprocess_image_landmark(cropped_image, landmark_model_input_size)

        # Run landmark inference
        # landmarks_output_from_model will be a list [vis_output, lm_output] or just [lm_output]
        # Assuming landmark_session.run returns a list of outputs, and landmarks are in the second element if two, or first if one.
        # The current landmark model seems to return (vis, landmarks)
        try:
            landmark_model_outputs = landmark_session.run(None, {landmark_input_name: landmark_input_tensor})
            # Assuming the actual landmark coordinates array is the second output if multiple exist (e.g. vis, landmarks)
            # Or the first if only one output. For 'landmark.onnx', it's [vis_scores, landmark_coords]
            if len(landmark_model_outputs) == 2: # Assuming (vis, landmarks)
                 landmarks_norm_raw = landmark_model_outputs[1] # Numpy array, e.g. shape (8,2)
            elif len(landmark_model_outputs) == 1: # Assuming only landmarks
                 landmarks_norm_raw = landmark_model_outputs[0]
            else:
                raise ValueError(f"Unexpected number of outputs from landmark model: {len(landmark_model_outputs)}")

        except Exception as e:
            app.logger.error(f"Error running landmark inference for box {detection[:4]}: {e}")
            landmarks_norm_raw = np.array([]) # Assign empty array on error to handle downstream

        # Scale landmarks back to original image coordinates (for JSON response)
        # Only proceed if landmarks_norm_raw is not empty and correctly shaped
        scaled_landmarks = []
        if landmarks_norm_raw.ndim == 2 and landmarks_norm_raw.shape[1] == 2:
            scaled_landmarks = scale_landmarks(
                landmarks_norm_raw,       # Numpy array of [x, y] normalized to landmark model input
                landmark_model_input_size,# (height, width) of landmark model input
                crop_size,                # (width, height) of the cropped image
                (bx1, by1)                # Top-left corner of the buffered crop in original image
            )
        else:
            app.logger.warning(f"Landmark output not in expected format (num_lm, 2). Got shape {landmarks_norm_raw.shape}. Scaled landmarks will be empty.")


        # 4. Run Classification Model on the crop and its landmarks
        classification_img_input_name = classification_input_details[0]['name']
        classification_img_input_size = classification_input_details[0]['size']
        classification_lm_input_name = classification_input_details[1]['name']
        classification_lm_input_shape = classification_input_details[1]['shape']

        classification_img_tensor = preprocess_image_classification(cropped_image, classification_img_input_size)
        
        try:
            if landmarks_norm_raw.ndim != 2 or landmarks_norm_raw.shape[0] == 0 : # Check if landmarks were detected
                 raise ValueError("No valid landmarks detected or landmark output is empty/malformed, cannot run classification.")
            
            classification_lm_tensor = prepare_landmarks_for_onnx(landmarks_norm_raw, classification_lm_input_shape)

            classification_feed_dict = {
                classification_img_input_name: classification_img_tensor,
                classification_lm_input_name: classification_lm_tensor
            }
            classification_outputs = classification_session.run(None, classification_feed_dict)
            # Assuming the output is a list of probabilities for each class
            # Output shape might be (1, num_classes)
            class_probs = classification_outputs[0][0] # Get the probabilities for the first (and only) batch item
            predicted_class_index = int(np.argmax(class_probs))
            predicted_class_confidence = float(class_probs[predicted_class_index])
            classification_result = {
                "predicted_class_index": predicted_class_index,
                "confidence": round(predicted_class_confidence, 3)
            }
        except Exception as e:
            app.logger.error(f"Error running classification inference for box {detection[:4]}: {e}")
            classification_result = {"error": "Classification failed"}


        return jsonify({
            "bbox": [x1, y1, x2, y2], # Original YOLO box (unbuffered)
            "confidence": round(confidence, 3),
            "landmarks": scaled_landmarks, # Scaled to original image coords
            "classification": classification_result
        }), 200


    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": "Failed to process image"}), 500


if __name__ == "__main__":
    if all([yolo_session, landmark_session, classification_session,
            yolo_input_details, landmark_input_details, classification_input_details]):
        print("Starting Flask server with YOLO, Landmark, and Classification models...")
        app.run(host="0.0.0.0", port=5000)
    else:
        print("FATAL: Cannot start server because one or more models or their configurations failed to load.")
