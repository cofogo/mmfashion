from __future__ import division
import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

import torchvision.transforms as transforms
import torchvision.transforms.functional as F

from mmcv import Config
from mmcv.runner import load_checkpoint

from mmfashion.core import CatePredictor
from mmfashion.models import build_predictor, build_landmark_detector
from mmfashion.utils import get_img_tensor, draw_landmarks


image_dir = '/Users/ties/Pictures/img/'
test_img = '/Users/ties/Pictures/img/img_00000012.jpg'

ONLY_FRONT = False


def parse_args():
    parser = argparse.ArgumentParser(description='Fashion Inference Demo')

    # Category/Attribute Prediction
    parser.add_argument('--input', type=str, help='input image path',
                        default=test_img)
    parser.add_argument('--predictor_config', type=str,
                        default='configs/category_attribute_predict/roi_predictor_resnet.py')
    parser.add_argument('--predictor_checkpoint', type=str,
                        default='checkpoints/resnetRoiLatest.pth')

    # Landmark Detection
    parser.add_argument('--landmark_config', type=str,
                        default='configs/landmark_detect/landmark_detect_resnet.py')
    parser.add_argument('--landmark_checkpoint', type=str,
                        default='checkpoints/resnetLandmarkLatest.pth')
    parser.add_argument('--run_landmark', type=bool, default=True)

    parser.add_argument('--use_cuda', type=bool, default=False)
    return parser.parse_args()


def get_img_tensor_inference(img_path, use_cuda, rotate=True, get_size=False):
    img = Image.open(img_path).convert('RGB')
    dim = min(img.size)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.CenterCrop(dim),
        transforms.Lambda(lambda img: F.rotate(img, 90)) if rotate else lambda img: img,
        transforms.ToTensor(),
        normalize,
    ])
    img_tensor = transform(img).unsqueeze(0)
    img_tensor = img_tensor.cuda() if use_cuda else img_tensor
    if get_size:
        return img_tensor, img.size[0], img.size[1]
    else:
        return img_tensor


def show_img(img_tensor, cat):
    tensor = img_tensor.squeeze(0).cpu()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_tensor = (tensor * std + mean).clamp(0, 1)
    img_np = img_tensor.permute(1, 2, 0).numpy()
    plt.figure()
    plt.title(cat)
    plt.imshow(img_np)
    plt.axis('off')
    plt.show()

def draw_landmarks_on_transformed(image_path, rotate, vis_lms):
    from PIL import ImageDraw

    img = Image.open(image_path).convert('RGB')
    dim = min(img.size)
    img = transforms.CenterCrop(dim)(img)
    if rotate:
        img = F.rotate(img, 90)

    draw = ImageDraw.Draw(img)
    for x, y in vis_lms:
        r = 3
        draw.ellipse((x - r, y - r, x + r, y + r), fill='red')

    img.show()  # or save with img.save('output.jpg')

def detect_landmarks(image_path, model, use_cuda, rotate=True):
    img_tensor, w, h = get_img_tensor_inference(
        image_path, use_cuda=use_cuda, rotate=rotate, get_size=True
    )

    model.eval()
    if use_cuda:
        model = model.cuda()
        img_tensor = img_tensor.cuda()

    # Run landmark detection
    pred_vis, pred_lm = model(img_tensor, return_loss=False)
    pred_lm = pred_lm.data.cpu().numpy()
    vis = pred_vis.data.cpu().numpy()

    landmark_tensor = np.zeros((16,))
    vis_lms = []

    for i, (v, lm) in enumerate(zip(vis, pred_lm)):
        if v >= 0.5:
            # Landmarks already in 224x224 space; scale to transformed image size
            x = lm[0] * w / 224.0
            y = lm[1] * h / 224.0
            landmark_tensor[i * 2] = x
            landmark_tensor[i * 2 + 1] = y
            vis_lms.append((x, y))

    # Draw on the transformed image (same as used in inference)
    draw_landmarks_on_transformed(image_path, rotate, vis_lms)

    landmark_tensor = torch.tensor(landmark_tensor).view(1, -1).float()
    return landmark_tensor.cuda() if use_cuda else landmark_tensor




def run_combined_inference(args):
    pred_cfg = Config.fromfile(args.predictor_config)
    lm_cfg = Config.fromfile(args.landmark_config)

    predictor = build_predictor(pred_cfg.model)
    load_checkpoint(predictor, args.predictor_checkpoint, map_location='cpu')
    if args.use_cuda:
        predictor = predictor.cuda()
    predictor.eval()
    print(f'Loaded predictor from: {args.predictor_checkpoint}')

    landmark_detector = build_landmark_detector(lm_cfg.model)
    load_checkpoint(landmark_detector, args.landmark_checkpoint, map_location='cpu')
    if args.use_cuda:
        landmark_detector = landmark_detector.cuda()
    landmark_detector.eval()
    print(f'Loaded landmark detector from: {args.landmark_checkpoint}')

    cate_predictor = CatePredictor(pred_cfg.data.test, tops_type=[5])

    # Test image
    img_tensor = get_img_tensor_inference(args.input, args.use_cuda, rotate=False)
    landmark_tensor = detect_landmarks(args.input, landmark_detector, args.use_cuda, rotate=False)
    attr_prob, cate_prob = predictor(img_tensor, attr=None, landmark=landmark_tensor, return_loss=False)
    cate_predictor.show_prediction(cate_prob)
    category = cate_predictor.cate_idx2name[cate_prob.argmax().item()]
    # show_img(img_tensor, category)

    # Folder batch
    for image_name in os.listdir(image_dir):
        if not image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        if ONLY_FRONT and not image_name.startswith('front'):
            continue
        image_path = os.path.join(image_dir, image_name)
        img_tensor = get_img_tensor_inference(image_path, args.use_cuda, rotate=False)
        landmark_tensor = detect_landmarks(image_path, landmark_detector, args.use_cuda, rotate=False)
        attr_prob, cate_prob = predictor(img_tensor, attr=None, landmark=landmark_tensor, return_loss=False)
        cate_predictor.show_prediction(cate_prob)
        category = cate_predictor.cate_idx2name[cate_prob.argmax().item()]
        # show_img(img_tensor, category)


def main():
    args = parse_args()
    if args.use_cuda and not torch.cuda.is_available():
        print("CUDA not available. Running on CPU.")
        args.use_cuda = False
    run_combined_inference(args)


if __name__ == '__main__':
    main()
