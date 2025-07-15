import torch
from collections import OrderedDict
import numpy as np
from lib.datasets.preprocessing import vgg_preprocess, rtpose_preprocess
import cv2

# モデルのインポート
from lib.network.VGG2016 import get_model


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


def padding(im, target_size=None, factor=32, is_ceil=True):
    """画像をパディングしてリサイズする"""
    # im_shape = img.shape
    # im_size_min = np.min(im_shape[0:2])     # 短辺を取得
    # im_size_max = np.max(im_shape[0:2])     # 長編を取得

    # im_scale = float(target_size) / im_size_min
    # img = cv2.resize(img, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_AREA)

    # h, w, c = img.shape
    # new_h = _factor_closest(h, factor, is_ceil)
    # new_w = _factor_closest(w, factor, is_ceil)

    # # 背景を黒でパディング
    # im_pad = np.zeros([new_h, new_w, c], dtype=img.dtype)

    # top_pad = (new_h - h) // 2
    # left_pad = (new_w - w) // 2

    # im_pad[top_pad:top_pad + h, left_pad:left_pad + w, :] = img

    # return im_pad, im_scale, im_shape
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    # im_scale = 1.
    # if max_size is not None and im_size_min > max_size:
    im_scale = float(target_size) / im_size_max
    im = cv2.resize(im, None, fx=im_scale, fy=im_scale)

    h, w, c = im.shape
    new_h = _factor_closest(h, factor=factor, is_ceil=is_ceil)
    new_w = _factor_closest(w, factor=factor, is_ceil=is_ceil)
    im_croped = np.full([new_h, new_w, c], fill_value=255, dtype=im.dtype)
    im_croped[0:h, 0:w, :] = im

    return im_croped, im_scale, im.shape


def get_outputs(image, model, preprocess):

    im_cloped, im_scale, real_shape = padding(image, 160, factor=8, is_ceil=True)
    print("im_scale:", im_scale)
    
    if preprocess == 'vgg':
        im_data = vgg_preprocess(im_cloped)
    elif preprocess == 'rtpose':
        im_data = rtpose_preprocess(im_cloped)

    batch_image = np.expand_dims(im_data, axis=0)

    batch_var = torch.from_numpy(batch_image).float().cuda()
    predicted, _ = model(batch_var)
    output1, output2 = predicted[-2], predicted[-1]
    pafs = output1.cpu().data.numpy().transpose(0, 2, 3, 1)[0]
    heatmaps = output2.cpu().data.numpy().transpose(0, 2, 3, 1)[0]

    return heatmaps, pafs, im_data, im_scale



if __name__ == "__main__":
    import cv2
    import matplotlib.pyplot as plt
    img_path = "data/cocoOriginal/images/val/000000009483.jpg"
    ckpt_path = "checkpoints/cocoBest/best_epoch.pth"

    img  = cv2.imread(img_path)

    model = get_model()
    model = load_ckpt(model, ckpt_path)
    heatmaps, pafs, im_data, im_scale = get_outputs(img, model, 'vgg')
    print("im_data shape:", im_data.shape)
    print("im_scale:", im_scale)
    print("heatmap shape:", heatmaps.shape)
    print("paf shape:", pafs.shape)
    im_data = im_data.transpose(1, 2, 0)  # CHW -> HWC


    # plt.subplot(121)
    # plt.imshow(im_data)
    # plt.subplot(122)
    # plt.imshow(heatmaps[:, :, 1], cmap='jet')
    # plt.show()
    

    plt.figure(figsize=(20, 4)) 
    for i in range(15):
        plt.subplot(3, 5, i+1)
        plt.imshow(heatmaps[:, :, i], cmap='jet', vmin=0, vmax=1)

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(20, 4)) 
    for i in range(30):
        plt.subplot(5, 6, i+1)
        plt.imshow(pafs[:, :, i], cmap='jet', vmin=0, vmax=1)

    plt.tight_layout()
    plt.show()
