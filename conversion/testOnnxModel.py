import onnxruntime as ort
import numpy as np
import cv2

onnx_model_path = "onnxmodels/vgglandmark.onnx"  # Replace with your model path
image_path = "/Users/ties/Pictures/shirt.jpg"        # Replace with your image path

def draw_landmarks_cv2(cv_image, landmarks, radius=3, color=(0, 0, 255)): # BGR color for red
    """Draws landmarks on an OpenCV image (numpy array)."""
    for x, y in landmarks:
        # Ensure coordinates are within image bounds for drawing
        x_draw, y_draw = int(round(x)), int(round(y))
        # Draw a circle for each landmark
        cv2.circle(cv_image, (x_draw, y_draw), radius, color, -1) # -1 thickness fills the circle
    return cv_image


def load_model(onnx_path):
    """Load the ONNX model"""
    session = ort.InferenceSession(onnx_path)
    print("Model loaded successfully.")
    return session

def preprocess_image(image_path, input_shape):
    """Preprocess image to fit model input shape"""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Optional: if model expects RGB
    img_resized = cv2.resize(img, (input_shape[3], input_shape[2]))
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_transposed = np.transpose(img_normalized, (2, 0, 1))  # HWC to CHW
    img_batch = np.expand_dims(img_transposed, axis=0)  # Add batch dimension
    return img_batch, img.shape

def run_inference(session, input_tensor):
    """Run inference and return output"""
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_tensor})
    return outputs[1]

def scale_landmarks(landmarks_norm, landmark_model_size, crop_size):
    # landmarks_norm: list of [x, y] normalized to landmark_model_size (e.g., 0-224)
    # landmark_model_size: (height, width) of the landmark model input
    # crop_size: (width, height) of the actual cropped image fed to landmark model
    # crop_origin_buffered: (x1, y1) of the crop in the original image (with buffer)

    lm_h, lm_w = landmark_model_size
    crop_w, crop_h = crop_size

    scaled_landmarks = []
    for lx_norm, ly_norm in landmarks_norm:
        # Scale from landmark model input space (e.g., 224x224) to crop space
        Lx_orig = int(round(lx_norm * crop_w / lm_w))
        Ly_orig = int(round(ly_norm * crop_h / lm_h))
        scaled_landmarks.append([Lx_orig, Ly_orig])

    return scaled_landmarks

if __name__ == "__main__":

    session = load_model(onnx_model_path)

    # Get input shape (e.g., (1, 3, 224, 224))
    input_shape = session.get_inputs()[0].shape
    input_tensor, img_shape = preprocess_image(image_path, input_shape)

    output = run_inference(session, input_tensor)
    landmarks = scale_landmarks(output, (input_shape[2], input_shape[3]), (img_shape[0], img_shape[1]))
    draw = draw_landmarks_cv2(cv2.imread(image_path), landmarks)  # Draw landmarks on the image
    cv2.imshow("Landmarks", draw)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("Inference Output:", output)
