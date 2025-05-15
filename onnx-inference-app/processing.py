import numpy as np
from torchvision import transforms
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