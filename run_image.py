import cv2
import argparse
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
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
    parser.add_argument('-a', '--analyze', action='store_true', help='heatmap, PAFsを可視化')
    args = parser.parse_args()

    device = get_using_device(args.device)
    model = get_model(args.model)
    model = load_ckpt(model, args.ckpt, device)

    if args.image:
        if not args.analyze:
            input_path = INPUTDIR + args.image
            output_path = OUTPUTDIR + args.image
            process_image(model, device, input_path, output_path)
        else:
            input_path = INPUTDIR + args.image
            output_path = OUTPUTDIR + args.image
            process_image_analyze(model, device, input_path, output_path)


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

def process_image_analyze(model, device, inimg, outimg):
    oriImg = cv2.imread(inimg)
    oriImgRGB = cv2.cvtColor(oriImg, cv2.COLOR_BGR2RGB)

    with torch.no_grad():
        paf, heatmap, im_scale = get_outputs(oriImg, model, 'vgg', device)

    humans = paf_to_pose_cpp(heatmap, paf, cfg)
    out = draw_humans(oriImg, humans)

    h, w, _ = oriImg.shape
    resized_heatmap = cv2.resize(heatmap[:, :, :-1], (w, h))
    res_heatmap = np.amax(resized_heatmap, axis=2)

    resized_paf_x = cv2.resize(paf[:, :, ::2], (w, h))
    res_paf_x = np.amax(resized_paf_x, axis=2)

    resized_paf_y = cv2.resize(paf[:, :, 1::2], (w, h))
    res_paf_y = np.amax(resized_paf_y, axis=2)


    # 姿勢推定画像
    plt.subplot(2, 2, 1)
    plt.title('Pose Estimation')
    plt.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))

    plt.subplot(2, 2, 2)
    plt.title('HeatMap')
    plt.imshow(oriImgRGB)
    plt.imshow(res_heatmap, alpha=0.5, cmap='jet')
    plt.colorbar()

    plt.subplot(2, 2, 3)
    plt.title('PAFs_x')
    plt.imshow(oriImgRGB)
    plt.imshow(res_paf_x, alpha=0.5, cmap='jet')
    plt.colorbar()

    plt.subplot(2, 2, 4)
    plt.title('PAFs_y')
    plt.imshow(oriImgRGB)
    plt.imshow(res_paf_y, alpha=0.5,cmap='jet')
    plt.colorbar()

    plt.savefig(outimg, dpi=300) 
    plt.show()

if __name__ == '__main__':
    main()
