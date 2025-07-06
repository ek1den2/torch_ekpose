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


def padding(img, target_size, factor=8, is_ceil=True):
    """画像をパディングしてリサイズする"""
    im_shape = img.shape
    im_size_min = np.min(im_shape[0:2])     # 短辺を取得
    im_size_max = np.max(im_shape[0:2])     # 長編を取得

    im_scale = float(target_size) / im_size_max
    img = cv2.resize(img, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_AREA)

    h, w, c = img.shape
    new_h = _factor_closest(h, factor, is_ceil)
    new_w = _factor_closest(w, factor, is_ceil)

    # 背景を黒でパディング
    im_pad = np.zeros([new_h, new_w, c], dtype=img.dtype)

    top_pad = (new_h - h) // 2
    left_pad = (new_w - w) // 2

    im_pad[top_pad:top_pad + h, left_pad:left_pad + w, :] = img

    return im_pad, im_scale, im_shape


def get_outputs(image, model, preprocess):

    im_cloped, im_scale, real_shape = padding(image, 160, factor=8, is_ceil=True)
    
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

    return heatmaps, pafs, im_data



if __name__ == "__main__":
    import cv2
    import matplotlib.pyplot as plt
    img_path = "results/futaba_017.png"
    ckpt_path = "checkpoints/20250704_20-53-50/epoch_180.pth"

    img  = cv2.imread(img_path)

    model = VGG19.get_model()
    model = load_ckpt(model, ckpt_path)
    heatmaps, pafs, im_data = get_outputs(img, model, 'vgg')

    print("heatmap shape:", heatmaps.shape)
    print("paf shape:", pafs.shape)
    im_data = im_data.transpose(1, 2, 0)  # CHW -> HWC


    # plt.subplot(121)
    # plt.imshow(im_data)
    # plt.subplot(122)
    # plt.imshow(heatmaps[:, :, 1], cmap='jet')
    # plt.show()
    

    for i in range(heatmaps.shape[2]):

        plt.subplot(221)
        plt.imshow(im_data)
        
        plt.subplot(222)
        plt.imshow(heatmaps[:, :, i], cmap='jet')

        plt.subplot(223)
        plt.imshow(pafs[:, :, i], cmap='jet')
        plt.show()