import cv2
import argparse
import numpy as np
import torch
from tqdm import tqdm

from lib.network.networks import get_model
from lib.evaluate.estimator import load_ckpt, get_outputs, get_using_device
from lib.config import cfg
from lib.utils.common import draw_humans
from lib.utils.paf_to_pose import paf_to_pose_cpp

CKPT_DIR = './checkpoints/'
INPUT_DIR = './data/'
OUTPUT_DIR = './results/'


parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, required=True, default='vgg2016')
parser.add_argument('-c', '--ckpt_path', type=str, default='pose_model.pth')
parser.add_argument('-i', '--input_video', type=str, default='demo.mp4')
parser.add_argument('-o', '--output_video', type=str, default='demo.mp4')
parser.add_argument('-d', '--device', type=str, default=None, choices=['cpu', 'cuda', 'mps'], help='使用するデバイス')
args = parser.parse_args()

ckpt_path = CKPT_DIR + args.ckpt_path
input_path = INPUT_DIR + args.input_video
output_path = OUTPUT_DIR + args.output_video

device = get_using_device(args.device)

model = get_model(args.model)
model = load_ckpt(model, args.ckpt_path, device)

video_path = input_path
video_capture_dummy = cv2.VideoCapture(video_path)
fps = video_capture_dummy.get(cv2.CAP_PROP_FPS)
ret, oriImg = video_capture_dummy.read()
shape_tuple = tuple(oriImg.shape[1::-1])
video_capture_dummy.release()

video_capture = cv2.VideoCapture(video_path)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
vid_out = cv2.VideoWriter(output_path, fourcc, fps, shape_tuple)

proc_frame_list = []
oriImg_list = []
while True:
    try:
        ret, oriImg = video_capture.read()
        if not ret:
            break
        oriImg_list.append(oriImg)
    except :
        break
video_capture.release()

print("Number of frames",len(oriImg_list))

count = 0
for oriImg in tqdm(oriImg_list):
    with torch.no_grad():
        paf, heatmap, imscale = get_outputs(oriImg, model, 'rtpose', device)
                    
    humans = paf_to_pose_cpp(heatmap, paf, cfg)             
    out = draw_humans(oriImg, humans)

    vid_out.write(out)

print("Video saved to", output_path)