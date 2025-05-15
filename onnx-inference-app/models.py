import os
import onnxruntime as ort


def load_model(path, default_img_size):
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"ONNX model not found at {path}")

        print(f"Loading ONNX model from {path}...")
        session = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
        model_inputs_meta = session.get_inputs()
        
        input_details_list = []
        
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
    
    except Exception as e:
        print(f"ERROR loading model from {path}: {e}")

def load_classification_model(path, default_img_size):
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"ONNX model not found at {path}")

        print(f"Loading ONNX model from {path}...")
        session = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
        model_inputs_meta = session.get_inputs()
        
        input_details_list = []

        if len(model_inputs_meta) < 2:
            raise ValueError(f"Classification model at {path} is expected to have at least 2 inputs (image, landmarks), but found {len(model_inputs_meta)}.")
        
        # Input 0: Image
        img_input_meta = model_inputs_meta[0]
        img_h, img_w = default_img_size
        if len(img_input_meta.shape) == 4: # BCHW
            if isinstance(img_input_meta.shape[2], int) and img_input_meta.shape[2] > 0: img_h = img_input_meta.shape[2]
            if isinstance(img_input_meta.shape[3], int) and img_input_meta.shape[3] > 0: img_w = img_input_meta.shape[3]
        else:
            print(f"Warning: Classification model image input '{img_input_meta.name}' shape {img_input_meta.shape} is not BCHW. Using default size {default_img_size}.")
        input_details_list.append({'name': img_input_meta.name, 'type': 'image', 'size': (img_h, img_w)})
        
        # Input 1: Landmarks
        lm_input_meta = model_inputs_meta[1]
        lm_shape_tuple = (1, 16)
        input_details_list.append({'name': lm_input_meta.name, 'type': 'landmarks', 'shape': lm_shape_tuple})
        
        print(f"Classification Model loaded. Image Input: '{input_details_list[0]['name']}' Size: {input_details_list[0]['size']}, Landmark Input: '{input_details_list[1]['name']}' Shape: {input_details_list[1]['shape']}")

        return session, input_details_list
    
    except Exception as e:
        print(f"ERROR loading model from {path}: {e}")
