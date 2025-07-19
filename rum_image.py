import cv2
import argparse
import numpy as np
import torch

from lib.network.networks import get_model
from lib.utils.common import draw_humans
from lib.utils.paf_to_pose import paf_to_pose_cpp
from lib.config import cfg
from lib.evaluate.estimator import load_ckpt, get_outputs, get_using_device

# TODO: 入力データ、出力データの親パスを設定


parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, required=True, default='vgg2016')
parser.add_argument('-c', '--ckpt_path', type=str, default='pose_model.pth')
parser.add_argument('-i', '--input_image', type=str, default='coco')
parser.add_argument('-o', '--output_image', type=str, default='result.png')
parser.add_argument('-d', '--device', type=str, default=None, choices=['cpu', 'cuda', 'mps'], help='使用するデバイス')
args = parser.parse_args()

device = get_using_device(args.device)

model = get_model(args.model)
model = load_ckpt(model, args.ckpt_path, device)

oriImg = cv2.imread(args.input_image) # B,G,R

# Get results of original image
with torch.no_grad():
    paf, heatmap, im_scale = get_outputs(oriImg, model, 'vgg', device)

humans = paf_to_pose_cpp(heatmap, paf, cfg)

out = draw_humans(oriImg, humans)
cv2.imwrite(args.output_image, out)

