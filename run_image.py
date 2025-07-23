import cv2
import argparse
import numpy as np
import torch
import os
from tqdm import tqdm

from lib.network.networks import get_model
from lib.utils.common import draw_humans
from lib.utils.paf_to_pose import paf_to_pose_cpp
from lib.config import cfg
from lib.evaluate.estimator import load_ckpt, get_outputs, get_using_device

# デフォルトの入力と出力ディレクトリ
INPUTDIR = './demo/'
OUTPUTDIR = './demo/outputs/'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, required=True, default='vgg2016', help='モデル名')
    parser.add_argument('-c', '--ckpt', type=str, default='./checkpoints/vgg2016/best_epoch.pth', help='チェックポイントのパス')
    parser.add_argument('-i', '--image', type=str, default=None, help='入力画像名')
    parser.add_argument('-d', '--device', type=str, default=None, choices=['cpu', 'cuda', 'mps'], help='使用するデバイス')
    args = parser.parse_args()

    device = get_using_device(args.device)
    model = get_model(args.model)
    model = load_ckpt(model, args.ckpt, device)

    if args.image:
        input_path = INPUTDIR + args.image
        output_path = OUTPUTDIR + args.image
        process_image(model, device, input_path, output_path)
    else:
        for filename in tqdm(os.listdir(INPUTDIR)):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                input_path = os.path.join(INPUTDIR, filename)
                output_path = os.path.join(OUTPUTDIR, filename)
                process_image(model, device, input_path, output_path)

    print(">>> 終了 <<<")


def process_image(model, device, inimg, outimg):
    oriImg = cv2.imread(inimg)

    with torch.no_grad():
        paf, heatmap, im_scale = get_outputs(oriImg, model, 'vgg', device)

    humans = paf_to_pose_cpp(heatmap, paf, cfg)

    out = draw_humans(oriImg, humans)
    cv2.imwrite(outimg, out)


if __name__ == '__main__':
    main()
