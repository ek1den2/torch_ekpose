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

INPUTDIR = './demo/'
OUTPUTDIR = './demo/outputs/'

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, required=True, default='vgg2016')
parser.add_argument('-c', '--ckpt', type=str, default='./checkpoints/vgg2016/best_epoch.pth')
parser.add_argument('-v', '--video', type=str, default='demo.mp4')
parser.add_argument('-d', '--device', type=str, default=None, choices=['cpu', 'cuda', 'mps'], help='使用するデバイス')
args = parser.parse_args()

device = get_using_device(args.device)

input_video = INPUTDIR + args.video
output_video = OUTPUTDIR + args.video

model = get_model(args.model)
model = load_ckpt(model, args.ckpt, device)

video_capture_dummy = cv2.VideoCapture(input_video)
fps = video_capture_dummy.get(cv2.CAP_PROP_FPS)
ret, oriImg = video_capture_dummy.read()
shape_tuple = tuple(oriImg.shape[1::-1])
video_capture_dummy.release()

video_capture = cv2.VideoCapture(input_video)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
vid_out = cv2.VideoWriter(output_video, fourcc, fps, shape_tuple)

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

print("フレーム数:",len(oriImg_list))

count = 0
for oriImg in tqdm(oriImg_list):
    with torch.no_grad():
        paf, heatmap, imscale = get_outputs(oriImg, model, 'vgg', device)
                    
    humans = paf_to_pose_cpp(heatmap, paf, cfg)             
    out = draw_humans(oriImg, humans)

    vid_out.write(out)

print(">>> 終了 <<<")