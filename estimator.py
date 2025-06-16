import cv2
import numpy as np
import time
import torch
import torch.nn.functional as F

from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.ndimage.morphology import generate_binary_structure
from lib.estimator.slidingwindow.SlidingWindow import DimOrder, generate

from collections import OrderedDict
from lib.estimator.pafprocess import pafprocess
from lib.dataloader.preprocess import vgg_preprocess

from lib.estimator.common import Human, BodyPart, OriginalPart, OriginalColors, OriginalPairRender


def _factor_closest(num, factor, is_ceil=True):
    """factorの倍数の中で最もnumに近い値をnumとする"""
    num = np.ceil(float(num) / factor) if is_ceil else np.floor(float(num) / factor)
    num = int(num) * factor
    print(f"factor_closest: {num}")
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

def find_peaks(param, img):
    """
    （グレースケールの）画像が与えられたとき、指定されたしきい値（param['thre1']）を超える
    局所的な最大値（ピーク）を検出する。

    :param img: ピークを検出したい入力画像（2次元配列）
    :return: 検出された各ピークの [x, y] 座標を格納した 2 次元の np.array を返す
    """

    peaks_binary = (maximum_filter(img, footprint=generate_binary_structure(
        2, 1)) == img) * (img > param)
    # Note reverse ([::-1]): we return [[x y], [x y]...] instead of [[y x], [y
    # x]...]
    return np.array(np.nonzero(peaks_binary)[::-1]).T

def compute_resized_coords(coords, resizeFactor):
    """
    ある入力配列（例：画像）内のセルのインデックス／座標が与えられたときに、
    その配列が resizeFactor 倍にリサイズされた場合の新しい座標を返す。
    例：3x3 サイズの画像が 6x6 にリサイズされた（resizeFactor=2）とすると、
    セル [1,2] の新しい座標は [2.5, 4.5] となる。
    :param coords: 入力配列内のセルの座標（インデックス）
    :param resizeFactor: リサイズ係数（= 出力配列サイズ / 入力配列サイズ）。
    例：resizeFactor=2 の場合、出力配列は元の配列の2倍の大きさになる。
    :return: 配列のサイズが shape_dest = resizeFactor × shape_source に
    リサイズされたときの座標を返す。この座標は、元の配列の 'coords' に最も近い
    点のインデックスを表す。
    """

    # 1) coords に 0.5 を加えることで、ピクセルの中心座標を得る（例：
    # インデックス [0,0] は位置 [0.5, 0.5] のピクセルを表す）
    # 2) その座標を resizeFactor を掛けて shape_dest に変換する
    # 3) 得られた数値は新しい配列におけるピクセル中心の位置を表すので、
    # そこから 0.5 を引いて配列インデックスの座標に変換する（ステップ1の逆を行う）
    return (np.array(coords, dtype=float) + 0.5) * resizeFactor - 0.5


def NMS(heatmaps, upsampFactor=1., bool_refine_center=True, bool_gaussian_filt=False, config=None):
    """
    NonMaximumSuppression (NMS): グレースケール画像の中から局所最大値（ピーク）を見つける
    :param heatmaps: グレースケール画像のセット（3次元のnp.array、 img_h x img_w x num_heatmaps）
    :param upsampFactor: ヒートマップ出力と入力画像サイズの比率（例: 元画像 480x640、ヒートマップ30x40xNならupsampFactor=16）
    :param bool_refine_center: 中心位置を微調整するかどうかを示すフラグ
     - False: 単に低解像度のピークをupsampFactorでアップサンプリングして返す（グリッドスナップあり）
     - True: （推奨、非常に高精度）各低解像度ピークの周辺パッチをアップサンプルし、  
   入力画像と同じ解像度でピーク位置を微調整
    :param bool_gaussian_filt: 1次元ガウシアンフィルタ（平滑化）をアップサンプルされたパッチに適用するかどうかのフラグ
    :return: NUM_JOINTS x 4のnp.array。各行は関節タイプ（0=頭、1=左肩...）を表し、列は{x,y}位置、スコア（確率）、ユニークID（カウンター）を示す
    """

    # CARLOSによる変更: heatmap_avgにヒートマップ全体をアップサンプリングしてから  
    # NMSでピークを探す代わりに、このステップは以下のように25〜50倍高速化可能：  
    # （RoG上での処理時間：ガウシアンフィルタありで9〜10ms、なしで5〜6ms → 通常は250〜280ms）  
    # 1. CPMの出力解像度（低解像度）でNMSを実行  
    # 1.1. scipy.ndimage.filters.maximum_filter を使ってピークを検出  
    # 2. ピークが見つかったら、その周囲の5x5パッチを取り出してアップサンプルし、  
    # 実際の最大値の位置を微調整  
    #  → これはheatmap_avgでピークを見つけるのと同等だが、  
    #     全体（例：480x640）をアップサンプル・スキャンする代わりに、5x5パッチだけを扱うためはるかに高速  

    joint_list_per_joint_type = []
    cnt_total_joints = 0

    # 検出された各ピークに対して、win_size はピークの中心から各方向に何ピクセル取るかを指定する。
    # この範囲のパッチを取り出してアップサンプリングする。
    # 例：win_size=1 → パッチは 3x3、win_size=2 → パッチは 5x5
    # （BICUBIC 補間で高精度に補間するには、win_size は 2 以上である必要がある！）
    win_size = 2

    for joint in range(config.MODEL.NUM_KEYPOINTS):
        map_orig = heatmaps[:, :, joint]
        peak_coords = find_peaks(config.TEST.THRESH_HEATMAP, map_orig)
        peaks = np.zeros((len(peak_coords), 4))
        for i, peak in enumerate(peak_coords):
            if bool_refine_center:
                x_min, y_min = np.maximum(0, peak - win_size)
                x_max, y_max = np.minimum(
                    np.array(map_orig.T.shape) - 1, peak + win_size)

                # 各ピークの周囲の小さな領域だけを取り、
                # その小さな部分だけをアップサンプリング
                patch = map_orig[y_min:y_max + 1, x_min:x_max + 1]
                map_upsamp = cv2.resize(
                    patch, None, fx=upsampFactor, fy=upsampFactor, interpolation=cv2.INTER_CUBIC)

                # ガウシアンフィルタリングは、1ピークあたり平均0.8ミリ秒かかる
                # （しかも1つの関節に複数のピークがある可能性もある！）
                # → 現時点ではスキップする（十分に高精度だから）。
                map_upsamp = gaussian_filter(map_upsamp, sigma=3) if bool_gaussian_filt else map_upsamp

                # パッチ内で最大値の座標を取得する。
                location_of_max = np.unravel_index(map_upsamp.argmax(), map_upsamp.shape)
                # ピークは [x, y] を示していることに注意
                # → [y, x] に反転する必要がある。
                location_of_patch_center = compute_resized_coords(
                    peak[::-1] - [y_min, x_min], upsampFactor)
                
                # 実際の最大値が存在するパッチの中心に対するオフセットを計算する。
                refined_center = (location_of_max - location_of_patch_center)
                peak_score = map_upsamp[location_of_max]
            else:
                refined_center = [0, 0]
                # ピークの座標は [y, x] ではなく [x, y] になっているため、座標を反転する。
                peak_score = map_orig[tuple(peak[::-1])]
            peaks[i, :] = tuple(
                x for x in compute_resized_coords(peak_coords[i], upsampFactor) + refined_center[::-1]) + (
                              peak_score, cnt_total_joints)
            cnt_total_joints += 1
        joint_list_per_joint_type.append(peaks)

    return joint_list_per_joint_type


def get_outputs(img, model, preprocess):
    """リサイズしpafとheatmapを取得"""
    input_size = 224

    # パディング
    # im_cloped, im_scale, real_shape = padding(img, input_size, factor=input_size, is_ceil=True)

    if preprocess == 'vgg':
        im_data = vgg_preprocess(img)

    batch_images = np.expand_dims(im_data, axis=0)

    batch_var = torch.from_numpy(batch_images).cuda().float()
    predicted_outputs, _ = model(batch_var)
    output1, output2 = predicted_outputs[-2], predicted_outputs[-1]
    heatmap = output2.cpu().data.numpy().transpose(0, 2, 3, 1)[0]
    paf = output1.cpu().data.numpy().transpose(0, 2, 3, 1)[0]

    return heatmap, paf, im_data


def model_load(model, ckpt_path):
    """モデルと重みを読み込む"""
    print("INFO: Loading model and weights...")
    with torch.autograd.no_grad():
        state_dict = torch.load(ckpt_path)
        
        # マルチGPU対応の重みをシングルGPU用に変換
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict, strict=True)
        model.eval()
        model.float()
        model.cuda()        

        return model


class PoseEstimator:
    def __init__(self):
        pass

    @staticmethod
    def estimate_paf(peaks, heat_up, paf_up):
        """peaks, heatmap, pafから骨格を推定
        peaks: 各関節のピーク座標とスコアのリスト
        heat_mat: アップサンプリングされたヒートマップ
        paf_mat: アップサンプリングされたPAFマップ
        """
        
        pafprocess.precess_paf(peaks, heat_up, paf_up)

        humans = []
        for human_id in range(pafprocess.get_num_humans()):
            human = Human([])
            is_added = False

            for part_idx in range(13):
                c_idx = int(pafprocess.get_part_cid(human_id, part_idx))
                if c_idx < 0:
                    continue

                is_added = True
                human.body_parts[part_idx] = BodyPart(
                    '%d-%d' % (human_id, part_idx), part_idx,
                    float(pafprocess.get_part_x(c_idx)) / heat_up.shape[1],
                    float(pafprocess.get_part_y(c_idx)) / heat_up.shape[0],
                    pafprocess.get_part_score(c_idx)
                )
            
            if is_added:
                score = pafprocess.get_score(human_id)
                human.score = score
                humans.append(human)
        
        return humans

class EKPoseEstimator:
    def __init__(self, model, ckpt_path, target_size=(160, 120), cfg=None):
        self.target_size = target_size

        # モデルの読み込み
        self.model = model_load(model, ckpt_path)

        # ウォームアップ
        #TODO: ここでウォームアップを行う
    

    def _get_scaled_img(self, npimg, scale):
        # スケール係数を計算するlambda関数
        get_base_scale = lambda s, w, h: max(self.target_size[0] / float(h), self.target_size[1] / float(w)) * s    
        img_h, img_w = npimg.shape[:2]

        if scale is None:
            if npimg.shape[:2] != (self.target_size[1], self.target_size[0]):
                # リサイズ
                npimg = cv2.resize(npimg, self.target_size, interpolation=cv2.INTER_CUBIC)
            return [npimg], [(0.0, 0.0, 1.0, 1.0)]
        elif isinstance(scale, float):
            # センタークロップによる拡縮
            base_scale = get_base_scale(scale, img_w, img_h)
            npimg = cv2.resize(npimg, desize=None, fx=base_scale, fy=base_scale, interpolation=cv2.INTER_CUBIC)

            o_size_h, o_size_w = npimg.shape[:2]
            if npimg.shape[0] < self.target_size[1] or npimg.shape[1] < self.target_size[0]:
                newimg = np.zeros(
                    (max(self.target_size[1], npimg.shape[0]),
                     max(self.target_size[0], npimg.shape[1]), 3), dtype=np.uint8)
                newimg[:npimg.shape[0], :npimg.shape[1], :] = npimg
                npimg = newimg
            
            window_step = scale[1]

            windows = generate(npimg, DimOrder.HeightWidthChannel, self.target_size[0], self.target_size[1], window_step)

            rois = []
            ratios = []
            for window in windows:
                indices = window.indices()
                roi = npimg[indices]
                rois.append(roi)
                ratio_x, ratio_y = float(indices[1].start) / o_size_w, float(indices[0].start) / o_size_h
                ratio_w, ratio_h = float(indices[1].stop - indices[1].start) / o_size_w, float(
                    indices[0].stop - indices[0].start) / o_size_h
                ratios.append((ratio_x, ratio_y, ratio_w, ratio_h))
            
            return rois, ratios
        
        elif isinstance(list, tuple) and len(scale) ==3:
            # ROIによるスケーリング : (want_x, want_y, scale_ratio)
            base_scale = get_base_scale(scale[2], img_w, img_h)
            npimg = cv2.resize(npimg, dsize=None, fx=base_scale, fy=base_scale, interpolation=cv2.INTER_CUBIC)
            ratio_w = self.target_size[0] / float(npimg.shape[1])
            ratio_h = self.target_size[1] / float(npimg.shape[0])

            want_x, want_y = scale[:2]
            ratio_x = want_x - ratio_w / 2.
            ratio_y = want_y - ratio_h / 2.
            ratio_x = max(ratio_x, 0.0)
            ratio_y = max(ratio_y, 0.0)
            if ratio_x + ratio_w > 1.0:
                ratio_x = 1. - ratio_w
            if ratio_y + ratio_h > 1.0:
                ratio_y = 1. - ratio_h

            roi = self._crop_roi(npimg, ratio_x, ratio_y)
            return [roi], [(ratio_x, ratio_y, ratio_w, ratio_h)]
    
    def _crop_roi(self, npimg, ratio_x, ratio_y):
        target_w, target_h = self.target_size
        h, w = npimg.shape[:2]
        x = max(int(w * ratio_x - .5), 0)
        y = max(int(h * ratio_y - .5), 0)
        cropped = npimg[y:y + target_h, x:x + target_w]

        cropped_h, cropped_w = cropped.shape[:2]
        if cropped_w < target_w or cropped_h < target_h:
            npblank = np.zeros((self.target_size[1], self.target_size[0], 3), dtype=np.uint8)

            copy_x, copy_y = (target_w - cropped_w) // 2, (target_h - cropped_h) // 2
            npblank[copy_y:copy_y + cropped_h, copy_x:copy_x + cropped_w] = cropped
        else:
            return cropped
    
    def inference(self, npimg, resize_to_default=True, upsample_size=1.0, preprocess='vgg'):
        if npimg is None:
            raise Exception('ERROR: 画像が読み込めません！')
        if resize_to_default:
            upsample_size = [int(self.target_size[1] / 8 * upsample_size), int(self.target_size[0] / 8 * upsample_size)]
        else:
            upsample_size = [int(npimg.shape[0] / 8 * upsample_size), int(npimg.shape[1] / 8 * upsample_size)]


        # logger.debug('inference+ original shape=%dx%d' % (npimg.shape[1], npimg.shape[0]))
        img = npimg
        # 画像のリサイズとスケーリング
        if resize_to_default:
            img = self._get_scaled_img(npimg, None)[0][0]

        heatmaps, pafs, im_scale = get_outputs(img, self.model, preprocess)

        # peaks = NMS(heatmaps, upsampFactor=8, config)

        



        heat_up = F.interpolate(heatmaps, size=upsample_size, mode='nearest', align_corners=False)
        paf_up = F.interpolate(pafs, size=upsample_size, mode='nearest', align_corners=False)

        # logger.debug('inference- heatMat=%dx%d pafMat=%dx%d' % (
        #     self.heatMat.shape[1], self.heatMat.shape[0], self.pafMat.shape[1], self.pafMat.shape[0]))

        t = time.time()
        humans = PoseEstimator.estimate_paf(peaks, heat_up, paf_up)
        elapsed = time.time() - t
        return humans


# pafprocess.prcess_paf(joint_list, heat_up, paf_up)

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # path
    img_path = '/home/masuryui/2DHPE/IRisPose/data/coco/images/train/image2.jpg'
    ckpt_path = 'checkpoints/20250613_00-03-12/best_epoch.pth'

    # テスト用の画像を読み込む
    img = cv2.imread(img_path)
    print(img.shape)


    from lib.networks.OpenPoseVGG19 import OpenPose

    model = OpenPose(pretrained=False, num_stages=6)

    model = model_load(model, ckpt_path)

    heatmap, paf, im_data = get_outputs(img, model, 'vgg')
    # EKPoseEstimator(model, '/home/masuryui/2DHPE/IRisPose/checkpoints/openpose_vgg19.pth', target_size=(224, 224))

    # EKPoseEstimator.inference

    print("heatmap shape:", heatmap.shape)
    print("paf shape:", paf.shape)
    im_data = im_data.transpose(1, 2, 0)  # CHW -> HWC

    for i in range(heatmap.shape[2]):

        plt.subplot(221)
        plt.imshow(im_data)
        plt.subplot(222)
        plt.imshow(heatmap[:, :, i], cmap='jet')

        plt.subplot(223)
        plt.imshow(paf[:, :, i], cmap='jet')
        plt.show()