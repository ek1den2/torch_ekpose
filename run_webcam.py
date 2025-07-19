import cv2
import argparse
import numpy as np
import torch
from time import time
from collections import deque

from lib.network.networks import get_model
from lib.config import cfg
from lib.evaluate.estimator import load_ckpt, get_outputs, get_using_device
from lib.utils.common import draw_humans
from lib.utils.paf_to_pose import paf_to_pose_cpp


parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, required=True, default='vgg2016')
parser.add_argument('-c', '--ckpt_path', type=str, default='pose_model.pth')
parser.add_argument('--camera', type=int, default=0)
parser.add_argument('-d', '--device', type=str, default=None, choices=['cpu', 'cuda', 'mps'], help='使用するデバイス')
args = parser.parse_args()

device = get_using_device(args.device)

model = get_model(args.model)     
model = load_ckpt(model, args.ckpt_path, device)

video_capture = cv2.VideoCapture(args.camera, cv2.CAP_AVFOUNDATION)

frame_times = deque(maxlen=60)
while True:
    ret, frame = video_capture.read()

    start = time()
    with torch.no_grad():
        paf, heatmap, imscale = get_outputs(
            frame, model, 'vgg', device)
    humans = paf_to_pose_cpp(heatmap, paf, cfg)
    out = draw_humans(frame, humans)
    end = time()

    frame_times.append(end - start)
    if len(frame_times) > 0:
        avg = sum(frame_times) / len(frame_times)
        fps = 1.0 / avg
    else:
        fps = 0.0

    # Display the resulting frame
    cv2.putText(out, f"FPS:{fps:3.2f}", (0, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_4)
    cv2.namedWindow('Video', cv2.WINDOW_NORMAL)  # サイズ変更可能なウィンドウ
    cv2.imshow('Video', out)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()

fps_values = [1.0 / t for t in frame_times if t > 0]  # 0除算防止
if len(fps_values) > 0:
    print("\n--- FPS Report ---")
    print(f"Max FPS: {max(fps_values):.2f}")
    print(f"Avg FPS: {sum(fps_values) / len(fps_values):.2f}")
    print(f"Min FPS: {min(fps_values):.2f}")
