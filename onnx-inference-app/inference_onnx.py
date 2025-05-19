import io
import numpy as np
import torch
from flask import Flask, request, jsonify
from PIL import Image
from ultralytics.utils import ops

from models import load_model, load_classification_model
from processing import *
from labels import CATEGORY_LIST, ATTRIBUTE_LIST, ATTRIBUTE_LIST_COARSE

print("Warning: If the container stops running without error, it most likely means the container ran out of memory.")

# --- Configuration ---
YOLO_MODEL_PATH = 'onnxmodels/yolo.onnx'
LANDMARK_MODEL_PATH = 'onnxmodels/landmark.onnx' # Added landmark model path
CLASSIFICATION_MODEL_PATH = 'onnxmodels/category.onnx' # Added classification model path
ATTRIBUTES_MODEL_PATH = 'onnxmodels/attributeLayers/attributes.onnx' # Added attributes model path

DEFAULT_YOLO_INPUT_SIZE = (224, 224)
DEFAULT_LANDMARK_INPUT_SIZE = (224, 224) # Default for landmark model
DEFAULT_CLASSIFICATION_INPUT_SIZE = (224, 224) # Default for classification model
DEFAULT_ATTRIBUTES_INPUT_SIZE = (224, 224) # Default for attributes model

CROP_BUFFER = 10 # Pixels to add around the bounding box for cropping
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
MAX_DETECTIONS = 1 # Only process the first most confident detection
CLOTHING_CLASS_ID = 2 # Class ID for clothing in YOLO model
ATTRIBUTE_THRESHOLD = 0.4
COARSE_ATTRIBUTE_THRESHOLD = 0.1


# --- Initialize Models ---
yolo_session, landmark_session, classification_session = None, None, None
yolo_input_details, landmark_input_details, classification_input_details = None, None, None

yolo_session, yolo_input_details = load_model(YOLO_MODEL_PATH, DEFAULT_YOLO_INPUT_SIZE)
landmark_session, landmark_input_details = load_model(LANDMARK_MODEL_PATH, DEFAULT_LANDMARK_INPUT_SIZE)
classification_session, classification_input_details = load_classification_model(CLASSIFICATION_MODEL_PATH, DEFAULT_CLASSIFICATION_INPUT_SIZE)
attributes_session, attributes_input_details = load_classification_model(ATTRIBUTES_MODEL_PATH, DEFAULT_ATTRIBUTES_INPUT_SIZE)

# --- Flask App ---
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    if not all([yolo_session, landmark_session, classification_session, attributes_session,
                yolo_input_details, landmark_input_details, classification_input_details, attributes_input_details]):
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
        
        bbox_result = {
            "bbox": [x1, y1, x2, y2],
            "confidence": round(confidence, 5)
        }

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
        landmark_input_tensor = preprocess_image(cropped_image, landmark_model_input_size)

        # Run landmark inference
        try:
            _, landmarks_norm_raw = landmark_session.run(None, {landmark_input_name: landmark_input_tensor})

        except Exception as e:
            app.logger.error(f"Error running landmark inference for box {detection[:4]}: {e}")
            landmarks_norm_raw = np.array([]) # Assign empty array on error to handle downstream

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

        classification_img_tensor = preprocess_image(cropped_image, classification_img_input_size)
        
        classification_result = {
            "predicted_class_index": None,
            "class_confidence": None
        }
        
        attributes_result = {
            "predicted_fine_attributes": None,
            "predicted_coarse_attributes": None
        }
        
        try:
            if landmarks_norm_raw.ndim != 2 or landmarks_norm_raw.shape[0] == 0 : # Check if landmarks were detected
                 raise ValueError("No valid landmarks detected or landmark output is empty/malformed, cannot run classification.")
            
            classification_lm_tensor = prepare_landmarks_for_onnx(landmarks_norm_raw, classification_lm_input_shape)

            classification_feed_dict = {
                classification_img_input_name: classification_img_tensor,
                classification_lm_input_name: classification_lm_tensor
            }
            classification_outputs = classification_session.run(None, classification_feed_dict)
            attr_probs, class_probs = classification_outputs
            predicted_attributes = []
            attr_probs = np.array(attr_probs)  # Ensure it's a NumPy array
            mask = attr_probs >= ATTRIBUTE_THRESHOLD
            predicted_attributes = list(np.array(ATTRIBUTE_LIST)[mask])
            predicted_class_index = int(np.argmax(class_probs))
            predicted_class_confidence = float(class_probs[predicted_class_index])
            classification_result["predicted_class_index" ]= CATEGORY_LIST[predicted_class_index]
            classification_result["class_confidence"] = round(predicted_class_confidence, 5)
            attributes_result["predicted_fine_attributes"] = predicted_attributes
        except Exception as e:
            app.logger.error(f"Error running classification inference for box {detection[:4]}: {e}")
            
        # 5. Run Attributes Model on the crop and its landmarks
        attributes_img_input_name = attributes_input_details[0]['name']
        attributes_img_input_size = attributes_input_details[0]['size']
        attributes_lm_input_name = attributes_input_details[1]['name']
        attributes_lm_input_shape = attributes_input_details[1]['shape']
        
        attributes_img_tensor = preprocess_image(cropped_image, attributes_img_input_size)
        try:
            if landmarks_norm_raw.ndim != 2 or landmarks_norm_raw.shape[0] == 0 : # Check if landmarks were detected
                 raise ValueError("No valid landmarks detected or landmark output is empty/malformed, cannot run attributes.")
            
            attributes_lm_tensor = prepare_landmarks_for_onnx(landmarks_norm_raw, attributes_lm_input_shape)

            attributes_feed_dict = {
                attributes_img_input_name: attributes_img_tensor,
                attributes_lm_input_name: attributes_lm_tensor
            }
            attr_probs = attributes_session.run(None, attributes_feed_dict)
            predicted_attributes = []
            attr_probs = np.array(attr_probs[0])  # Ensure it's a NumPy array
            mask = attr_probs >= COARSE_ATTRIBUTE_THRESHOLD
            predicted_attributes = list(np.array(ATTRIBUTE_LIST_COARSE)[mask])
            attributes_result["predicted_coarse_attributes"] = predicted_attributes
        except Exception as e:
            app.logger.error(f"Error running attribute inference for box {detection[:4]}: {e}")


        return jsonify({
            "bounding_box": bbox_result,
            "landmarks": scaled_landmarks, # Scaled to original image coords
            "classification": classification_result,
            "attributes": attributes_result
        }), 200


    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": "Failed to process image"}), 500


if __name__ == "__main__":
    if all([yolo_session, landmark_session, classification_session, attributes_session,
            yolo_input_details, landmark_input_details, classification_input_details, attributes_input_details]):
        print("Starting Flask server with YOLO, Landmark, and Classification models...")
        app.run(host="0.0.0.0", port=5000)
    else:
        print("FATAL: Cannot start server because one or more models or their configurations failed to load.")
