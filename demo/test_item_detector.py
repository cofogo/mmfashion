from ultralytics import YOLO
from ultralyticsplus import render_result
import torch
import os

input_dir = 'data/Attr_Predict/img/Woven_Suit_Joggers'

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
    results = model.predict(image, conf=0.01, imgsz=640, show=True)

    if results[0].boxes.id is None:
        print('No object detected')
        continue
    # observe results
    print("Detected boxes object:", results[0].boxes)
    
    # Explicitly print bounding box coordinates for each detection
    print(f"--- Bounding Boxes for {os.path.basename(image)} ---")
    for box in results[0].boxes:
        class_id = model.names[int(box.cls[0])]
        coords = box.xyxy[0].tolist()
        conf = box.conf[0].item()
        print(f"  Class: {class_id}, Confidence: {conf:.2f}, Coords (xyxy): {coords}")
    print("------------------------------------\n")

    render = render_result(model=model, image=image, result=results[0])
    render.show()
