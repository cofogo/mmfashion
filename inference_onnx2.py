import argparse
import os
import cv2
import numpy as np
import onnxruntime as ort
import torch
from PIL import Image

# Attempt to import Ultralytics components
try:
    from ultralytics.utils import ops
    from ultralytics.data.augment import letterbox
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("Warning: Ultralytics library not found or not fully installed. Some functionalities might be limited.")
    # Define dummy functions if ultralytics is not available, to allow basic script structure
    def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
        raise ImportError("Ultralytics letterbox function is not available.")
    class ops:
        @staticmethod
        def non_max_suppression(*args, **kwargs):
            raise ImportError("Ultralytics ops.non_max_suppression function is not available.")
        @staticmethod
        def scale_boxes(*args, **kwargs):
            raise ImportError("Ultralytics ops.scale_boxes function is not available.")


# --- Helper: Box class to mimic Ultralytics output ---
class Box:
    def __init__(self, xyxy_tensor, conf_tensor, cls_tensor, orig_shape):
        self.xyxy = xyxy_tensor
        self.conf = conf_tensor
        self.cls = cls_tensor
        self.orig_shape = orig_shape

class Boxes:
    def __init__(self, xyxy, conf, cls, orig_shape):
        self.xyxy = torch.as_tensor(xyxy)
        self.conf = torch.as_tensor(conf)
        self.cls = torch.as_tensor(cls)
        self.orig_shape = orig_shape
        # Concatenate data if all inputs are valid tensors and have items
        if self.xyxy.numel() > 0 and self.conf.numel() > 0 and self.cls.numel() > 0:
             self.data = torch.cat((self.xyxy, self.conf.unsqueeze(1), self.cls.unsqueeze(1)), dim=1)
        else:
            self.data = torch.empty((0,6), device=self.xyxy.device)


    def __len__(self):
        return self.xyxy.shape[0]

    def __getitem__(self, idx):
        return Box(self.xyxy[idx], self.conf[idx], self.cls[idx], self.orig_shape)

class ONNXResults:
    def __init__(self, boxes_obj, names_map, orig_img_path, orig_img_shape):
        self.boxes = boxes_obj
        self.names = names_map
        self.path = orig_img_path
        self.orig_shape = orig_img_shape # H, W of original image
        # Add other attributes if needed, like .speed, .orig_img

# --- Image Preprocessing ---
def preprocess_image_for_onnx(img_path, input_hw):
    """Loads and preprocesses an image for ONNX YOLO model."""
    if not ULTRALYTICS_AVAILABLE:
        raise ImportError("Ultralytics components (like letterbox) are required for preprocessing.")

    img0 = cv2.imread(img_path)  # BGR
    assert img0 is not None, f"Image Not Found {img_path}"
    original_shape_hw = img0.shape[:2] # H, W

    # Letterbox
    img = letterbox(img0, new_shape=input_hw, auto=False, scaleup=False)[0] # input_hw is (H,W)

    # Convert HWC to CHW, BGR to RGB
    img = img.transpose((2, 0, 1))[::-1]
    img = np.as_contiguousarray(img)

    # Normalize and to tensor
    img_tensor = torch.from_numpy(img).float()
    img_tensor /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img_tensor.ndimension() == 3:
        img_tensor = img_tensor.unsqueeze(0)  # add batch dim

    return img_tensor, img0, original_shape_hw

# --- Main ONNX Predict Function ---
def predict_with_onnx(onnx_model_path, image_path, conf_threshold=0.25, iou_threshold=0.45, model_input_size_wh=(224,224)):
    if not ULTRALYTICS_AVAILABLE:
        raise ImportError("Ultralytics components (like ops.non_max_suppression) are required for postprocessing.")

    # 1. Load ONNX session
    session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider']) # Or ['CUDAExecutionProvider']
    input_cfg = session.get_inputs()[0]
    input_name = input_cfg.name
    onnx_input_shape = input_cfg.shape # e.g., [1, 3, 'height', 'width'] or [1, 3, 224, 224]

    # Determine model input H, W
    # Use provided model_input_size_wh by default, can be overridden if ONNX shape is fixed
    current_model_input_h, current_model_input_w = model_input_size_wh[1], model_input_size_wh[0]

    if isinstance(onnx_input_shape[2], int) and onnx_input_shape[2] > 0:
        current_model_input_h = onnx_input_shape[2]
    if isinstance(onnx_input_shape[3], int) and onnx_input_shape[3] > 0:
        current_model_input_w = onnx_input_shape[3]
    
    print(f"Using model input size (H, W): ({current_model_input_h}, {current_model_input_w})")

    # 2. Preprocess image
    img_tensor, _, original_hw = preprocess_image_for_onnx(image_path, (current_model_input_h, current_model_input_w))
    img_np = img_tensor.cpu().numpy()

    # 3. Run inference
    raw_outputs = session.run(None, {input_name: img_np})
    preds_tensor = torch.from_numpy(raw_outputs[0])

    # 4. Post-process (NMS, scaling)
    detections = ops.non_max_suppression(
        preds_tensor,
        conf_thres=conf_threshold,
        iou_thres=iou_threshold,
        classes=None,
        agnostic=False,
        max_det=300
    )[0]  # Get detections for the first (and only) image

    # Scale boxes from model input size to original image size
    if detections.shape[0] > 0:
        detections[:, :4] = ops.scale_boxes(img_tensor.shape[2:], detections[:, :4], original_hw).round()

    # 5. Format results
    if detections.shape[0] > 0:
        boxes_obj = Boxes(
            xyxy=detections[:, :4],
            conf=detections[:, 4],
            cls=detections[:, 5],
            orig_shape=original_hw
        )
    else:
        boxes_obj = Boxes(
            xyxy=torch.empty((0, 4)),
            conf=torch.empty((0,)),
            cls=torch.empty((0,)),
            orig_shape=original_hw
        )

    # Load class names
    current_class_names = {0: 'object'} # Default placeholder
    if ULTRALYTICS_AVAILABLE:
        try:
            pt_model_path = onnx_model_path.replace(".onnx", ".pt")
            if os.path.exists(pt_model_path):
                yolo_model_for_names = YOLO(pt_model_path)
                current_class_names = yolo_model_for_names.names
                print(f"Loaded class names from {pt_model_path}")
            else:
                print(f"Warning: Corresponding .pt model ({pt_model_path}) not found. Using placeholder class names: {current_class_names}")
        except Exception as e:
            print(f"Warning: Could not load class names from .pt model ({e}). Using placeholder class names: {current_class_names}")
    else:
        print(f"Warning: Ultralytics not available to load class names from .pt model. Using placeholder: {current_class_names}")


    results_obj = ONNXResults(boxes_obj, current_class_names, image_path, original_hw)
    return [results_obj] # Return as a list to match yolo_results structure

# --- Main Execution ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ONNX YOLO Inference')
    parser.add_argument('--onnx-model', type=str, default='checkpoints/yoloItem.onnx', help='Path to ONNX model')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--conf-threshold', type=float, default=0.25, help='Confidence threshold for NMS')
    parser.add_argument('--iou-threshold', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--model-input-width', type=int, default=224, help='Model input width (e.g., 224, 640)')
    parser.add_argument('--model-input-height', type=int, default=224, help='Model input height (e.g., 224, 640)')
    parser.add_argument('--output-dir', type=str, default='output_onnx_inference', help='Directory to save output image')

    args = parser.parse_args()

    if not os.path.exists(args.onnx_model):
        print(f"Error: ONNX model not found at {args.onnx_model}")
        exit(1)
    if not os.path.exists(args.image):
        print(f"Error: Image not found at {args.image}")
        exit(1)
    
    if not ULTRALYTICS_AVAILABLE:
        print("Error: Ultralytics library is required for this script to run. Please install it.")
        exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    yolo_results_onnx = predict_with_onnx(
        args.onnx_model,
        args.image,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
        model_input_size_wh=(args.model_input_width, args.model_input_height)
    )

    if yolo_results_onnx and yolo_results_onnx[0].boxes and len(yolo_results_onnx[0].boxes) > 0:
        results = yolo_results_onnx[0]
        print(f"Found {len(results.boxes)} objects in {os.path.basename(args.image)}.")

        img_to_draw = cv2.imread(args.image)

        for i in range(len(results.boxes)):
            # Access data directly from results.boxes.data for efficiency
            box_data = results.boxes.data[i] # xyxy, conf, cls
            xyxy = box_data[:4].cpu().numpy().astype(int)
            conf = box_data[4].item()
            cls_idx = int(box_data[5].item())
            
            class_name = results.names.get(cls_idx, f"class_{cls_idx}")
            label = f"{class_name}: {conf:.2f}"
            print(f"  Detection {i}: Class='{class_name}' (ID {cls_idx}), Confidence={conf:.2f}, Box={xyxy.tolist()}")

            color = (0, 255, 0) 
            cv2.rectangle(img_to_draw, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 2)
            cv2.putText(img_to_draw, label, (xyxy[0], xyxy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        output_image_path = os.path.join(args.output_dir, f"onnx_out_{os.path.basename(args.image)}")
        cv2.imwrite(output_image_path, img_to_draw)
        print(f"Output image saved to: {output_image_path}")
    else:
        print(f"No objects detected in {os.path.basename(args.image)} with current thresholds.")

    print("ONNX inference finished.")
