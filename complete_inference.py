import argparse
import os
import torch
import numpy as np
from PIL import Image # Keep PIL for opening and initial cropping
import cv2 # Import OpenCV
import torchvision.transforms as transforms
from mmcv import Config
from mmcv.runner import load_checkpoint
from ultralytics import YOLO

# MMfashion imports
from mmfashion.models import build_predictor, build_landmark_detector
from mmfashion.core import CatePredictor # Assuming we primarily want category for now
# from mmfashion.core import AttrPredictor # Can add later if needed
# from mmfashion.utils import get_img_tensor # We'll create a specific transform for PIL images


# --- Configuration ---
DEFAULT_INPUT_DIR = 'data/Attr_Predict/img/V-Neck_Sweater' # Example input
DEFAULT_OUTPUT_DIR = 'output_complete_inference'

# Model Paths (Update these paths as needed)
YOLO_CHECKPOINT = 'checkpoints/best.pt'
LANDMARK_CONFIG = 'configs/landmark_detect/landmark_detect_resnet.py'
LANDMARK_CHECKPOINT = 'checkpoints/resnetLandmarkLatest.pth'
PREDICTOR_CONFIG = 'configs/category_attribute_predict/roi_predictor_resnet.py'
PREDICTOR_CHECKPOINT = 'checkpoints/resnetRoiLatest.pth'

# --- Argument Parsing ---
def parse_args():
    parser = argparse.ArgumentParser(description='Complete Fashion Analysis Pipeline')
    parser.add_argument(
        '--input',
        type=str,
        help='Input image path or directory path',
        default=DEFAULT_INPUT_DIR)
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Directory to save cropped images and results',
        default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        '--yolo-checkpoint',
        type=str,
        default=YOLO_CHECKPOINT,
        help='Path to YOLO detection model checkpoint')
    parser.add_argument(
        '--landmark-config',
        type=str,
        default=LANDMARK_CONFIG,
        help='Path to landmark detection config file')
    parser.add_argument(
        '--landmark-checkpoint',
        type=str,
        default=LANDMARK_CHECKPOINT,
        help='Path to landmark detection model checkpoint')
    parser.add_argument(
        '--predictor-config',
        type=str,
        default=PREDICTOR_CONFIG,
        help='Path to category/attribute predictor config file')
    parser.add_argument(
        '--predictor-checkpoint',
        type=str,
        default=PREDICTOR_CHECKPOINT,
        help='Path to category/attribute predictor model checkpoint')
    parser.add_argument(
        '--conf-threshold',
        type=float,
        default=0.25, # Confidence threshold for YOLO detection
        help='Confidence threshold for YOLO object detection')
    parser.add_argument(
        '--use-cuda',
        action='store_true',
        help='Use GPU if available')
    parser.add_argument(
        '--save-crops',
        action='store_true',
        help='Save cropped clothing images')
    parser.add_argument(
        '--draw-landmarks',
        action='store_true',
        help='Draw landmarks on saved cropped images')

    args = parser.parse_args()
    if args.use_cuda and not torch.cuda.is_available():
        print("CUDA requested but not available. Running on CPU.")
        args.use_cuda = False
    return args

# --- Preprocessing for MMfashion Models ---
def preprocess_image_mmfashion(pil_image, use_cuda):
    """Prepares a PIL image for MMfashion landmark/predictor models."""
    # Use normalization parameters consistent with MMfashion training
    img_norm = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # From configs
    normalize = transforms.Normalize(mean=img_norm['mean'], std=img_norm['std'])
    transform = transforms.Compose([
        transforms.Resize((224, 224)), # Resize to the expected input size (e.g., 224x224)
        transforms.ToTensor(),
        normalize,
    ])
    img_tensor = transform(pil_image).unsqueeze(0)
    return img_tensor.cuda() if use_cuda else img_tensor

# --- Landmark Detection on Cropped Image ---
def detect_landmarks_on_crop(pil_crop, landmark_detector, use_cuda):
    """Detects landmarks on a cropped PIL image."""
    img_tensor = preprocess_image_mmfashion(pil_crop, use_cuda)
    img_h, img_w = pil_crop.size[1], pil_crop.size[0] # Get dimensions of the crop

    landmark_detector.eval()
    with torch.no_grad():
        # Assuming the landmark detector returns visibility and landmarks
        pred_vis, pred_lm = landmark_detector(img_tensor, return_loss=False)

    pred_lm = pred_lm.data.cpu().numpy().squeeze() # Shape (landmark_num, 2) e.g., (8, 2)
    vis = pred_vis.data.cpu().numpy().squeeze() # Shape (landmark_num,) e.g., (8,)

    visible_landmarks_coords = [] # For drawing on the original crop dimensions

    # MMfashion landmarks output by the model are typically normalized to the input size (e.g., 224x224)
    # The predictor model (RoI predictor) expects landmarks in this 224x224 space.
    num_landmarks = pred_lm.shape[0]
    landmark_tensor_np = np.zeros((num_landmarks * 2,), dtype=np.float32)

    # Scale factors for drawing landmarks on the *original* crop size
    scale_x = img_w / 224.0
    scale_y = img_h / 224.0

    for i in range(num_landmarks):
        # Store the 224x224 coordinates for the predictor tensor
        x_norm = pred_lm[i, 0]# * 224.0
        y_norm = pred_lm[i, 1]# * 224.0
        landmark_tensor_np[i * 2] = x_norm
        landmark_tensor_np[i * 2 + 1] = y_norm

        # Store visible landmarks scaled to original crop size for drawing
        if vis[i] >= 0.5: # Visibility threshold
             visible_landmarks_coords.append((x_norm * scale_x, y_norm * scale_y))

    landmark_tensor = torch.from_numpy(landmark_tensor_np).view(1, -1) # Shape (1, num_landmarks * 2)
    return landmark_tensor.cuda() if use_cuda else landmark_tensor, visible_landmarks_coords


# --- Draw Landmarks using OpenCV ---
def draw_landmarks_cv2(cv_image, landmarks, radius=3, color=(0, 0, 255)): # BGR color for red
    """Draws landmarks on an OpenCV image (numpy array)."""
    for x, y in landmarks:
        # Ensure coordinates are within image bounds for drawing
        x_draw, y_draw = int(round(x)), int(round(y))
        # Draw a circle for each landmark
        cv2.circle(cv_image, (x_draw, y_draw), radius, color, -1) # -1 thickness fills the circle
    return cv_image

# --- Utility to convert PIL to OpenCV format ---
def pil_to_cv2(pil_image):
    """Converts a PIL image (RGB) to an OpenCV image (BGR)."""
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

# --- Utility to display image with waitkey ---
def display_image(window_name, image, wait_time=0):
    """Displays an image using cv2.imshow and waits."""
    cv2.imshow(window_name, image)
    key = cv2.waitKey(wait_time)
    return key # Return the pressed key

# --- Main Function ---
def main():
    args = parse_args()
    device = torch.device("cuda" if args.use_cuda else "cpu")

    # --- Load Models ---
    print("Loading models...")
    # 1. YOLO Item Detector
    try:
        yolo_model = YOLO(args.yolo_checkpoint)
        yolo_model.to(device)
        print(f"Loaded YOLO detector from: {args.yolo_checkpoint}")
    except Exception as e:
        print(f"Error loading YOLO model from {args.yolo_checkpoint}: {e}")
        return

    # 2. Landmark Detector
    try:
        lm_cfg = Config.fromfile(args.landmark_config)
        landmark_detector = build_landmark_detector(lm_cfg.model)
        load_checkpoint(landmark_detector, args.landmark_checkpoint, map_location='cpu')
        landmark_detector.to(device)
        landmark_detector.eval()
        print(f"Loaded landmark detector from: {args.landmark_checkpoint}")
    except Exception as e:
        print(f"Error loading landmark detector from {args.landmark_checkpoint} using {args.landmark_config}: {e}")
        return

    # 3. Category/Attribute Predictor
    try:
        pred_cfg = Config.fromfile(args.predictor_config)
        predictor = build_predictor(pred_cfg.model)
        load_checkpoint(predictor, args.predictor_checkpoint, map_location='cpu')
        predictor.to(device)
        predictor.eval()
        print(f"Loaded predictor from: {args.predictor_checkpoint}")
    except Exception as e:
        print(f"Error loading predictor from {args.predictor_checkpoint} using {args.predictor_config}: {e}")
        return

    # Helper for showing category names
    # Ensure the test data config path in predictor_config is correct
    cate_predictor_display = None
    try:
        # Check if the necessary keys exist in the loaded config
        if hasattr(pred_cfg, 'data') and hasattr(pred_cfg.data, 'test') and hasattr(pred_cfg.data.test, 'cate_cloth_file'):
             # Check if the file exists before initializing
             if os.path.exists(pred_cfg.data.test.cate_cloth_file):
                 cate_predictor_display = CatePredictor(pred_cfg.data.test)
                 print(f"Initialized CatePredictor using: {pred_cfg.data.test.cate_cloth_file}")
             else:
                 print(f"Warning: Category list file not found at {pred_cfg.data.test.cate_cloth_file}. Category names will not be displayed.")
        else:
            print("Warning: Config structure for CatePredictor not found (pred_cfg.data.test.cate_cloth_file). Category names will not be displayed.")

    except Exception as e:
        print(f"Warning: Could not initialize CatePredictor for display names: {e}")


    # --- Prepare Output Directory ---
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output will be saved to: {args.output_dir}")

    # --- Process Input ---
    if os.path.isdir(args.input):
        image_files = []
        for f in os.listdir(args.input):
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                image_files.append(os.path.join(args.input, f))
    elif os.path.isfile(args.input):
        image_files = [args.input]
    else:
        print(f"Error: Input path {args.input} is not a valid file or directory.")
        return

    if not image_files:
        print(f"No image files found in {args.input}")
        return

    print(f"Found {len(image_files)} images to process.")

    # --- Main Processing Loop ---
    for img_path in image_files:
        print(f"\n--- Processing: {os.path.basename(img_path)} ---")
        try:
            original_pil_image = Image.open(img_path).convert("RGB")
            # Convert original PIL image to OpenCV format for drawing
            original_cv_image = pil_to_cv2(original_pil_image)
        except Exception as e:
            print(f"  Error opening image: {e}")
            continue

        # 1. Detect Items using YOLO
        print("  Step 1: Running YOLO object detection...")
        try:
            yolo_results = yolo_model.predict(img_path, conf=args.conf_threshold, verbose=False)
        except Exception as e:
            print(f"    Error during YOLO prediction: {e}")
            continue

        if len(yolo_results) == 0 or len(yolo_results[0].boxes) == 0:
            print("    No objects detected by YOLO.")
            continue

        # --- Visualization: Show all YOLO detections ---
        yolo_vis_image = original_cv_image.copy()
        boxes = yolo_results[0].boxes
        print(f"    Found {len(boxes)} objects in total.")
        for box in boxes:
            try:
                coords = list(map(int, box.xyxy[0].tolist()))
                conf = box.conf[0].item()
                cls_idx = int(box.cls[0])
                label = f"{yolo_model.names[cls_idx]}: {conf:.2f}"
                # Draw rectangle (BGR color - green for clothing, blue otherwise)
                color = (0, 255, 0) if yolo_model.names[cls_idx] == 'clothing' else (255, 0, 0)
                cv2.rectangle(yolo_vis_image, (coords[0], coords[1]), (coords[2], coords[3]), color, 2)
                # Put label
                cv2.putText(yolo_vis_image, label, (coords[0], coords[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            except Exception as e:
                print(f"    Warning: Error processing a YOLO box for visualization: {e}")
        print("    Displaying all YOLO detections. Press any key to continue...")
        display_image(f"YOLO Detections: {os.path.basename(img_path)}", yolo_vis_image)
        cv2.destroyWindow(f"YOLO Detections: {os.path.basename(img_path)}") # Close the window
        # --- End Visualization ---


        crop_counter = 0
        processed_clothing_item = False
        print("\n  Processing detected 'clothing' items:")
        for i, box in enumerate(boxes):
            try:
                class_id_idx = int(box.cls[0])
                class_name = yolo_model.names[class_id_idx]
            except Exception as e:
                print(f"  Error accessing YOLO box data: {e}")
                continue # Skip this box

            # 2. Filter for 'clothing' and Crop
            if class_name == 'clothing':
                processed_clothing_item = True
                coords = box.xyxy[0].tolist() # [x1, y1, x2, y2]
                conf = box.conf[0].item()
                print(f"  Detected 'clothing' item {crop_counter} (Confidence: {conf:.2f}) at {coords}")

                # Crop the original image using PIL, adding a 10px boundary
                img_w, img_h = original_pil_image.size
                raw_x1, raw_y1, raw_x2, raw_y2 = map(int, coords)

                # Add 10px boundary
                boundary = 10
                x1_adj = raw_x1 - boundary
                y1_adj = raw_y1 - boundary
                x2_adj = raw_x2 + boundary
                y2_adj = raw_y2 + boundary

                # Clip to image dimensions
                x1 = max(0, x1_adj)
                y1 = max(0, y1_adj)
                x2 = min(img_w, x2_adj)
                y2 = min(img_h, y2_adj)

                if x1 >= x2 or y1 >= y2:
                    print(f"    Invalid or zero-area crop coordinates after boundary and clamping: ({x1},{y1},{x2},{y2}). Original: ({raw_x1},{raw_y1},{raw_x2},{raw_y2}). Skipping.")
                    continue
                min_crop_size = 50
                if x2-x1 < min_crop_size or y2-y1 < min_crop_size:
                    print(f"    Crop size too small after boundary and clamping: ({x2-x1},{y2-y1}). Minimum: {min_crop_size}. Skipping.")
                    continue
                crop_coords_valid = (x1, y1, x2, y2)

                try:
                    cropped_pil_image = original_pil_image.crop(crop_coords_valid)
                except Exception as e:
                     print(f"    Error cropping image: {e}")
                     continue

                if cropped_pil_image.size[0] == 0 or cropped_pil_image.size[1] == 0:
                    print("    Warning: Cropped image has zero size after PIL crop. Skipping.")
                    continue

                # Convert crop to OpenCV format for processing and display
                cropped_cv_image = pil_to_cv2(cropped_pil_image)

                # --- Visualization: Show Cropped Image ---
                print("    Step 2: Displaying cropped clothing item. Press any key...")
                display_image(f"Crop {crop_counter}", cropped_cv_image)
                cv2.destroyWindow(f"Crop {crop_counter}")
                # --- End Visualization ---

                # 3. Detect Landmarks on the Crop
                '''
                For upper-body clothes, landmark annotations are listed in the order of
                ["left collar", "right collar", "left sleeve", "right sleeve", "left hem", "right hem"]
                
                For lower-body clothes, landmark annotations are listed in the order of
                ["left waistline", "right waistline", "left hem", "right hem"]
                
                For upper-body clothes, landmark annotations are listed in the order of
                ["left collar", "right collar", "left sleeve", "right sleeve", "left waistline", "right waistline", "left hem", "right hem"].
                '''
                print(f"    Step 3: Detecting landmarks for item {crop_counter}...")
                try:
                    landmark_tensor, visible_landmarks = detect_landmarks_on_crop(
                        cropped_pil_image, landmark_detector, args.use_cuda # Use PIL image for tensor input
                    )
                    print(f"      Detected {len(visible_landmarks)} visible landmarks.")

                    # --- Visualization: Show Landmarks on Crop ---
                    landmarks_vis_image = cropped_cv_image.copy() # Use CV image for drawing
                    landmarks_vis_image = draw_landmarks_cv2(landmarks_vis_image, visible_landmarks)
                    print("      Displaying landmarks on crop. Press any key...")
                    display_image(f"Landmarks {crop_counter}", landmarks_vis_image)
                    cv2.destroyWindow(f"Landmarks {crop_counter}")
                    # --- End Visualization ---

                except Exception as e:
                    print(f"      Error detecting landmarks for item {crop_counter}: {e}")
                    # For RoI predictor, landmarks are usually required.
                    print(f"    Skipping prediction for item {crop_counter} due to landmark error.")
                    print(f"      Skipping prediction for item {crop_counter} due to landmark error.")
                    continue # Skip prediction if landmarks failed

                # 4. Predict Category/Attributes using Crop and Landmarks
                print(f"    Step 4: Predicting category/attributes for item {crop_counter}...")
                pred_cate_name = "unknown_category" # Default name
                final_vis_image = cropped_cv_image.copy() # Start with the crop
                if visible_landmarks: # Draw landmarks if available
                    final_vis_image = draw_landmarks_cv2(final_vis_image, visible_landmarks)

                try:
                    # Preprocess the PIL crop specifically for the predictor
                    img_tensor_pred = preprocess_image_mmfashion(cropped_pil_image, args.use_cuda)

                    with torch.no_grad():
                        # Pass image tensor and landmark tensor to the predictor
                        attr_prob, cate_prob = predictor(img_tensor_pred, attr=None, landmark=landmark_tensor, return_loss=False)

                    # Process and display category prediction
                    if cate_prob is not None:
                        pred_cate_idx = cate_prob.argmax().item()
                        if cate_predictor_display and pred_cate_idx in cate_predictor_display.cate_idx2name:
                            pred_cate_name = cate_predictor_display.cate_idx2name[pred_cate_idx]
                            print(f"      Predicted Category: {pred_cate_name} (Index: {pred_cate_idx})")
                        else:
                            pred_cate_name = f"category_index_{pred_cate_idx}"
                            print(f"      Predicted Category Index: {pred_cate_idx} (Name lookup unavailable)")
                    else:
                        print("      Category prediction not available.")
                        pred_cate_name = "category_NA"

                    # Add attribute prediction processing here if needed

                    # --- Visualization: Show Final Result (Crop + Landmarks + Prediction) ---
                    # Add predicted category text to the image
                    cv2.putText(final_vis_image, pred_cate_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2) # Green text at top-left
                    print("      Displaying final prediction. Press any key to validate and continue...")
                    display_image(f"Prediction {crop_counter}: {pred_cate_name}", final_vis_image)
                    cv2.destroyWindow(f"Prediction {crop_counter}: {pred_cate_name}") # Close the window
                    # --- End Visualization ---

                except Exception as e:
                    print(f"      Error during category/attribute prediction: {e}")
                    # Display the image before prediction failure if possible
                    print("      Displaying image before prediction error. Press any key...")
                    display_image(f"Error Before Prediction {crop_counter}", final_vis_image) # Show crop+landmarks if available
                    cv2.destroyWindow(f"Error Before Prediction {crop_counter}")
                    # Continue to next item or save without prediction name

                # 5. Save Cropped Image (Optional) - Keep this functionality
                if args.save_crops:
                    # Use the final visualized image (crop + landmarks) for saving if landmarks were drawn
                    img_to_save_cv = final_vis_image if (args.draw_landmarks and visible_landmarks) else cropped_cv_image

                    # Sanitize category name for filename
                    safe_cate_name = "".join(c if c.isalnum() or c in ['_','-'] else "_" for c in pred_cate_name)
                    base_filename = os.path.splitext(os.path.basename(img_path))[0]
                    output_filename = f"{base_filename}_crop_{crop_counter}_{safe_cate_name}.jpg"
                    output_filename = f"{base_filename}_crop_{crop_counter}_{safe_cate_name}.jpg"
                    output_path = os.path.join(args.output_dir, output_filename)
                    try:
                        # Save the OpenCV image
                        cv2.imwrite(output_path, img_to_save_cv)
                        print(f"    Saved final image to: {output_path}")
                    except Exception as e:
                        print(f"    Error saving final image {output_path}: {e}")

                crop_counter += 1
                print("-" * 20) # Separator between clothing items
            # End if class_name == 'clothing'
        # End loop over boxes for one image

        if not processed_clothing_item:
            print("  No 'clothing' items found in this image.")

    print("\nProcessing finished.")


if __name__ == '__main__':
    main()
