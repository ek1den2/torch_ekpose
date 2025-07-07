import torch
from collections import OrderedDict
import numpy as np
from lib.datasets.preprocessing import vgg_preprocess, rtpose_preprocess
import cv2

# モデルのインポート
from lib.network import VGG19


def load_ckpt(model, ckpt_path):
    """チェックポイントをロード"""

    print("INFO: Loading Checkpoint...")
    with torch.autograd.no_grad():
        state_dict = torch.load(ckpt_path)

        # nn.parallelによるマルチGPUのモデル情報をシングルGPU用に変換
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # 例えば module. が先頭にある場合は7文字削除
            new_state_dict[name] = v

        # ロード
        model.load_state_dict(new_state_dict, strict=True)

        model.eval()
        model.float()
        model = model.cuda()
    
    return model


def _factor_closest(num, factor, is_ceil=True):
    """factorの倍数の中で最もnumに近い値をnumとする"""
    num = np.ceil(float(num) / factor) if is_ceil else np.floor(float(num) / factor)
    num = int(num) * factor
    return num


def padding(im, dest_size, factor=8, is_ceil=True):
    """画像をパディングしてリサイズする"""
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    # im_scale = 1.
    # if max_size is not None and im_size_min > max_size:
    im_scale = float(dest_size) / im_size_max
    im = cv2.resize(im, None, fx=im_scale, fy=im_scale)

    h, w, c = im.shape
    new_h = _factor_closest(h, factor=factor, is_ceil=is_ceil)
    new_w = _factor_closest(w, factor=factor, is_ceil=is_ceil)
    im_pad = np.zeros([new_h, new_w, c], dtype=im.dtype)
    im_pad[0:h, 0:w, :] = im

    return im_pad, im_scale, im.shape


def get_outputs(image, model, preprocess):

    im_cloped, im_scale, pad = padding(image, 160, factor=8, is_ceil=True)
    
    if preprocess == 'vgg':
        im_data = vgg_preprocess(im_cloped)
    elif preprocess == 'rtpose':
        im_data = rtpose_preprocess(im_cloped)

    batch_image = np.expand_dims(im_data, 0)

    batch_var = torch.from_numpy(batch_image).float().cuda()
    predicted, _ = model(batch_var)
    output1, output2 = predicted[-2], predicted[-1]
    pafs = output1.cpu().data.numpy().transpose(0, 2, 3, 1)[0]
    heatmaps = output2.cpu().data.numpy().transpose(0, 2, 3, 1)[0]

    return pafs, heatmaps, im_scale