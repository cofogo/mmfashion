from ultralytics import YOLO
# from ultralyticsplus import render_result # No longer needed for rendering all boxes
import torch
import os
from PIL import Image

input_dir = 'data/Attr_Predict/img/Woven_Suit_Joggers'
output_dir = 'output_cropped' # Directory to save cropped images
os.makedirs(output_dir, exist_ok=True) # Create output directory if it doesn't exist

# load model
model = YOLO('checkpoints/best.pt', task='detect')
# checkpoint = torch.load('checkpoints/best.pt', weights_only=False)
# model.load_state_dict(checkpoint, strict=False)
model.model.eval()

# set image
# image = 'demo/imgs/01_4_full.jpg'

for image in os.listdir(input_dir):
    if not image.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue
    image = os.path.join(input_dir, image)

    # perform inference
    results = model.predict(image, conf=0.1, imgsz=640, show=True)

    # Check if any boxes were detected
    if len(results[0].boxes) == 0:
        print(f'No objects detected in {os.path.basename(image)}')
        continue

    # Load the original image using PIL for cropping
    try:
        pil_image = Image.open(image).convert("RGB")
    except Exception as e:
        print(f"Error opening image {image} with PIL: {e}")
        continue

    # Process and crop 'clothing' bounding boxes
    print(f"--- Processing 'clothing' boxes for {os.path.basename(image)} ---")
    crop_counter = 0
    for box in results[0].boxes:
        class_id = model.names[int(box.cls[0])]

        # Filter for 'clothing' class
        if class_id == 'clothing':
            coords = box.xyxy[0].tolist() # [x1, y1, x2, y2]
            conf = box.conf[0].item()
            print(f"  Found 'clothing': Confidence: {conf:.2f}, Coords (xyxy): {coords}")

            # Crop the image using PIL
            # Ensure coordinates are integers for cropping
            crop_coords = tuple(map(int, coords))
            cropped_img = pil_image.crop(crop_coords)

            # Save the cropped image
            base_filename = os.path.splitext(os.path.basename(image))[0]
            output_filename = f"{base_filename}_clothing_{crop_counter}.jpg"
            output_path = os.path.join(output_dir, output_filename)
            try:
                cropped_img.save(output_path)
                print(f"    Saved cropped image to: {output_path}")
                crop_counter += 1
            except Exception as e:
                print(f"    Error saving cropped image {output_path}: {e}")

    if crop_counter == 0:
        print(f"  No 'clothing' items found or saved for {os.path.basename(image)}.")
    print("------------------------------------\n")

    # The rendering part is removed as we are now saving cropped images instead.
    # render = render_result(model=model, image=image, result=results[0])
    # render.show()
