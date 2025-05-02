import argparse
import os
import torch
import numpy as np
from PIL import Image, ImageDraw
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
DEFAULT_INPUT_DIR = 'data/Attr_Predict/img/Woven_Suit_Joggers' # Example input
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
        x_norm = pred_lm[i, 0] * 224.0
        y_norm = pred_lm[i, 1] * 224.0
        landmark_tensor_np[i * 2] = x_norm
        landmark_tensor_np[i * 2 + 1] = y_norm

        # Store visible landmarks scaled to original crop size for drawing
        if vis[i] >= 0.5: # Visibility threshold
             visible_landmarks_coords.append((x_norm * scale_x, y_norm * scale_y))

    landmark_tensor = torch.from_numpy(landmark_tensor_np).view(1, -1) # Shape (1, num_landmarks * 2)
    return landmark_tensor.cuda() if use_cuda else landmark_tensor, visible_landmarks_coords


# --- Draw Landmarks ---
def draw_landmarks_on_pil(pil_image, landmarks, radius=3, color='red'):
    """Draws landmarks on a PIL image."""
    draw = ImageDraw.Draw(pil_image)
    for x, y in landmarks:
        # Ensure coordinates are within image bounds for drawing
        x_draw, y_draw = int(round(x)), int(round(y))
        draw.ellipse((x_draw - radius, y_draw - radius, x_draw + radius, y_draw + radius), fill=color)
    return pil_image

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
        except Exception as e:
            print(f"  Error opening image: {e}")
            continue

        # 1. Detect Items using YOLO
        try:
            yolo_results = yolo_model.predict(img_path, conf=args.conf_threshold, verbose=False) # verbose=False to reduce clutter
        except Exception as e:
            print(f"  Error during YOLO prediction: {e}")
            continue

        if len(yolo_results) == 0 or len(yolo_results[0].boxes) == 0:
            print("  No objects detected by YOLO.")
            continue

        boxes = yolo_results[0].boxes
        crop_counter = 0
        processed_clothing_item = False
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

                # Crop the original image
                crop_coords_int = tuple(map(int, coords))
                # Ensure coordinates are valid (x1 < x2, y1 < y2)
                if crop_coords_int[0] >= crop_coords_int[2] or crop_coords_int[1] >= crop_coords_int[3]:
                    print(f"    Invalid crop coordinates: {crop_coords_int}. Skipping.")
                    continue
                try:
                    cropped_pil_image = original_pil_image.crop(crop_coords_int)
                except Exception as e:
                     print(f"    Error cropping image: {e}")
                     continue

                if cropped_pil_image.size[0] == 0 or cropped_pil_image.size[1] == 0:
                    print("    Warning: Cropped image has zero size. Skipping.")
                    continue

                # 3. Detect Landmarks on the Crop
                try:
                    landmark_tensor, visible_landmarks = detect_landmarks_on_crop(
                        cropped_pil_image, landmark_detector, args.use_cuda
                    )
                    print(f"    Detected {len(visible_landmarks)} visible landmarks for item {crop_counter}.")
                except Exception as e:
                    print(f"    Error detecting landmarks for item {crop_counter}: {e}")
                    # Decide whether to continue without landmarks or skip prediction
                    # For RoI predictor, landmarks are usually required.
                    print(f"    Skipping prediction for item {crop_counter} due to landmark error.")
                    continue

                # 4. Predict Category/Attributes using Crop and Landmarks
                pred_cate_name = "unknown_category" # Default name if prediction fails or no display helper
                try:
                    # Preprocess the crop specifically for the predictor
                    img_tensor_pred = preprocess_image_mmfashion(cropped_pil_image, args.use_cuda)

                    with torch.no_grad():
                        # Pass image tensor and landmark tensor to the predictor
                        # Ensure the keys ('img', 'landmark') match what the predictor's forward method expects
                        # Based on typical MMfashion structure, it might just take tensors directly or a dict
                        # Assuming direct tensor input based on common patterns:
                        attr_prob, cate_prob = predictor(img_tensor_pred, attr=None, landmark=landmark_tensor, return_loss=False)
                        # If it expects a dict:
                        # results = predictor(dict(img=img_tensor_pred, landmark=landmark_tensor), return_loss=False)
                        # attr_prob, cate_prob = results['attr_prob'], results['cate_prob'] # Adjust keys as needed

                    # Process and display predictions
                    if cate_prob is not None:
                        pred_cate_idx = cate_prob.argmax().item()
                        if cate_predictor_display and pred_cate_idx in cate_predictor_display.cate_idx2name:
                            pred_cate_name = cate_predictor_display.cate_idx2name[pred_cate_idx]
                            print(f"    Predicted Category for item {crop_counter}: {pred_cate_name} (Index: {pred_cate_idx})")
                        else:
                            pred_cate_name = f"category_index_{pred_cate_idx}"
                            print(f"    Predicted Category Index for item {crop_counter}: {pred_cate_idx} (Name lookup unavailable)")
                        # Optional: Show top-k probabilities
                        # if cate_predictor_display:
                        #    cate_predictor_display.show_prediction(cate_prob)
                    else:
                        print(f"    Category prediction not available for item {crop_counter}.")

                    # Add attribute prediction display here if needed
                    # if attr_prob is not None:
                    #    # Process attr_prob (e.g., find top attributes based on threshold)
                    #    print(f"    Attribute probabilities (first 5): {attr_prob.squeeze()[:5].tolist()}")


                except Exception as e:
                    print(f"    Error during category/attribute prediction for item {crop_counter}: {e}")
                    # Continue to next item, but maybe save crop without prediction name

                # 5. Save Cropped Image (Optional)
                if args.save_crops:
                    img_to_save = cropped_pil_image.copy()
                    if args.draw_landmarks and visible_landmarks:
                        try:
                            img_to_save = draw_landmarks_on_pil(img_to_save, visible_landmarks)
                        except Exception as e:
                            print(f"    Warning: Could not draw landmarks on saved crop: {e}")

                    # Sanitize category name for filename
                    safe_cate_name = "".join(c if c.isalnum() else "_" for c in pred_cate_name)
                    base_filename = os.path.splitext(os.path.basename(img_path))[0]
                    output_filename = f"{base_filename}_crop_{crop_counter}_{safe_cate_name}.jpg"
                    output_path = os.path.join(args.output_dir, output_filename)
                    try:
                        img_to_save.save(output_path)
                        print(f"    Saved cropped image to: {output_path}")
                    except Exception as e:
                        print(f"    Error saving cropped image {output_path}: {e}")

                crop_counter += 1
            # End if class_name == 'clothing'
        # End loop over boxes

        if not processed_clothing_item:
            print("  No 'clothing' items found in this image.")

    print("\nProcessing finished.")


if __name__ == '__main__':
    main()
