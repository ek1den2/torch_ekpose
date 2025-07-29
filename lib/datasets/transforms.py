from abc import ABCMeta, abstractmethod
import copy
import io
import logging
import math
import numpy as np
import PIL
import scipy
import random
import cv2
import torch
import torchvision
from functools import partial, reduce
from .utils import horizontal_swap_coco


# jpeg圧縮によるデータ拡張
def jpeg_compression_augmentation(im):
    f = io.BytesIO()
    im.save(f, 'jpeg', quality=50)
    return PIL.Image.open(f)

# ガウシアンブラー（ぼかし）によるデータ拡張
def blur_augmentation(im, max_sigma=5.0):
    im_np = np.asarray(im)
    sigma = max_sigma * float(torch.rand(1).item())
    im_np = scipy.ndimage.filters.gaussian_filter(im_np, sigma=(sigma, sigma, 0))
    return PIL.Image.fromarray(im_np)

# 画像の正規化（RGB）
normalize = torchvision.transforms.Normalize(  # pylint: disable=invalid-name
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

# 画像の正規化（Gray）
normalize_Gray = torchvision.transforms.Normalize(  # pylint: disable=invalid-name
    mean=[0.5],
    std=[0.5]
)

# テンソル変換
image_transform = torchvision.transforms.Compose([  # pylint: disable=invalid-name
    torchvision.transforms.ToTensor(),
    normalize,
])

# 学習用画像変換まとめ
image_transform_train = torchvision.transforms.Compose([  # pylint: disable=invalid-name
    # 色のジッター（揺らぎ）を追加
    torchvision.transforms.ColorJitter(brightness=0.1,
                                       contrast=0.1,
                                       saturation=0.1,
                                       hue=0.1),

    # JPEG圧縮による拡張を確率的に適用 
    torchvision.transforms.RandomApply([
        # cocoデータセットではあまり効果がないらしい
        torchvision.transforms.Lambda(jpeg_compression_augmentation),
    ], p=0.1),

    # ランダムグレースケール化
    torchvision.transforms.RandomGrayscale(p=0.01),
    torchvision.transforms.ToTensor(),
    normalize,
])


class Preprocess(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, image, mask, anns, meta):
        """前処理操作の実装"""

    @staticmethod
    def keypoint_sets_inverse(keypoint_sets, meta):
        keypoint_sets = keypoint_sets.copy()

        keypoint_sets[:, :, 0] += meta['offset'][0]
        keypoint_sets[:, :, 1] += meta['offset'][1]

        keypoint_sets[:, :, 0] = (keypoint_sets[:, :, 0] + 0.5) / meta['scale'][0] - 0.5
        keypoint_sets[:, :, 1] = (keypoint_sets[:, :, 1] + 0.5) / meta['scale'][1] - 0.5

        if meta['hflip']:
            w = meta['width_height'][0]
            keypoint_sets[:, :, 0] = -keypoint_sets[:, :, 0] - 1.0 + w
            for keypoints in keypoint_sets:
                if meta.get('horizontal_swap'):
                    keypoints[:] = meta['horizontal_swap'](keypoints)

        return keypoint_sets


# アノテーションの正規化
class Normalize(Preprocess):
    @staticmethod
    def normalize_annotations(anns):
        anns = copy.deepcopy(anns)

        # データのnumpy配列化
        # これにより各float型の値が個別のtorch.tensorに変換されるを避ける
        for ann in anns:
            ann['keypoints'] = np.asarray(ann['keypoints'], dtype=np.float32).reshape(-1, 3)
            ann['bbox'] = np.asarray(ann['bbox'], dtype=np.float32)
            ann['bbox_original'] = np.copy(ann['bbox'])
            del ann['segmentation']

        return anns

    def __call__(self, image, mask, anns, meta):
        anns = self.normalize_annotations(anns)

        if meta is None:
            w, h = image.size
            meta = {
                'offset': np.array((0.0, 0.0)),
                'scale': np.array((1.0, 1.0)),
                'valid_area': np.array((0.0, 0.0, w, h)),
                'hflip': False,
                'width_height': np.array((w, h)),
            }

        return image, mask, anns, meta


# 複数の処理を１連の処理としてまとめ、再利用するクラス
class Compose(Preprocess):
    def __init__(self, preprocess_list):
        self.preprocess_list = preprocess_list

    def __call__(self, image, mask, anns, meta):
    
        augmentations = [partial(aug_meth) for aug_meth in self.preprocess_list]
        image, mask, anns, meta = reduce(
            lambda md_i_mm, f: f(*md_i_mm),
            augmentations,
            (image, mask, anns, meta)
        )
        return image, mask, anns, meta


class MultiScale(Preprocess):
    def __init__(self, preprocess_list):
        """前処理の処理ステップをリストとして作成

        この関数は、すべての前処理の一番外側（最初）に位置する必要がある
        作成するリストには、transforms.Compose() を使った前処理の組み合わせも含めることができる
        """
        self.preprocess_list = preprocess_list

    def __call__(self, image, mask, anns, meta):
        image_list, mask_list, anns_list, meta_list = [], [], [], []
        for p in self.preprocess_list:
            this_image, this_mask, this_anns, this_meta = p(image, mask, anns, meta)
            image_list.append(this_image)
            mask_list.append(this_mask)
            anns_list.append(this_anns)
            meta_list.append(this_meta)

        return image_list, mask_list, anns_list, meta_list


# リスケール
class RescaleRelative(Preprocess):
    def __init__(self, scale_range=(0.5, 1.0), *, resample=PIL.Image.BICUBIC):
        self.log = logging.getLogger(self.__class__.__name__)
        self.scale_range = scale_range
        self.resample = resample

    def __call__(self, image, mask, anns, meta):
        meta = copy.deepcopy(meta)
        anns = copy.deepcopy(anns)

        if isinstance(self.scale_range, tuple):
            scale_factor = (
                self.scale_range[0] +
                torch.rand(1).item() * (self.scale_range[1] - self.scale_range[0])
            )
        else:
            scale_factor = self.scale_range

        image, mask, anns, scale_factors = self.scale(image, mask, anns, scale_factor)
        self.log.debug('meta before: %s', meta)
        meta['offset'] *= scale_factors
        meta['scale'] *= scale_factors
        meta['valid_area'][:2] *= scale_factors
        meta['valid_area'][2:] *= scale_factors
        self.log.debug('meta after: %s', meta)

        for ann in anns:
            ann['valid_area'] = meta['valid_area']

        return image, mask, anns, meta

    def scale(self, image, mask, anns, factor):
        # scale image
        w, h = image.size
        new_size = (int(w * factor), int(h * factor))
        image = image.resize(new_size, self.resample)
        if mask:
            mask = mask.resize(new_size, PIL.Image.NEAREST)

        self.log.debug('before resize = (%f, %f), after = %s', w, h, image.size)

        # rescale keypoints
        x_scale = image.size[0] / w
        y_scale = image.size[1] / h
        for ann in anns:
            ann['keypoints'][:, 0] = (ann['keypoints'][:, 0] + 0.5) * x_scale - 0.5
            ann['keypoints'][:, 1] = (ann['keypoints'][:, 1] + 0.5) * y_scale - 0.5
            ann['bbox'][0] *= x_scale
            ann['bbox'][1] *= y_scale
            ann['bbox'][2] *= x_scale
            ann['bbox'][3] *= y_scale

        return image, mask, anns, np.array((x_scale, y_scale))

# 長辺に合わせてリスケール
class RescaleAbsolute(Preprocess):
    def __init__(self, long_edge, *, resample=PIL.Image.BICUBIC):
        self.log = logging.getLogger(self.__class__.__name__)
        self.long_edge = long_edge
        self.resample = resample

    def __call__(self, image, mask, anns, meta):
        meta = copy.deepcopy(meta)
        anns = copy.deepcopy(anns)

        image, mask, anns, scale_factors = self.scale(image, mask, anns)
        self.log.debug('meta before: %s', meta)
        meta['offset'] *= scale_factors
        meta['scale'] *= scale_factors
        meta['valid_area'][:2] *= scale_factors
        meta['valid_area'][2:] *= scale_factors
        self.log.debug('meta after: %s', meta)

        for ann in anns:
            ann['valid_area'] = meta['valid_area']

        return image, mask, anns, meta

    def scale(self, image, mask, anns):
        # scale image
        w, h = image.size

        this_long_edge = self.long_edge
        if isinstance(this_long_edge, (tuple, list)):
            this_long_edge = int(torch.randint(this_long_edge[0], this_long_edge[1], (1,)).item())

        s = this_long_edge / max(h, w)
        if h > w:
            new_size = (int(w * s), this_long_edge)
        else:
            new_size = (this_long_edge, int(h * s))

        image = image.resize(new_size, self.resample)
        if mask:
            mask = mask.resize(new_size, PIL.Image.NEAREST)


        self.log.debug('before resize = (%f, %f), scale factor = %f, after = %s',
                       w, h, s, image.size)

        # rescale keypoints
        x_scale = image.size[0] / w
        y_scale = image.size[1] / h
        for ann in anns:
            ann['keypoints'][:, 0] = (ann['keypoints'][:, 0] + 0.5) * x_scale - 0.5
            ann['keypoints'][:, 1] = (ann['keypoints'][:, 1] + 0.5) * y_scale - 0.5
            ann['bbox'][0] *= x_scale
            ann['bbox'][1] *= y_scale
            ann['bbox'][2] *= x_scale
            ann['bbox'][3] *= y_scale

        return image, mask, anns, np.array((x_scale, y_scale))


# 画像のクロップ
class Crop(Preprocess):
    def __init__(self, long_edge):
        self.log = logging.getLogger(self.__class__.__name__)
        self.long_edge = long_edge

    def __call__(self, image, mask, anns, meta):
        meta = copy.deepcopy(meta)
        anns = copy.deepcopy(anns)

        image, mask, anns, ltrb = self.crop(image, mask, anns)
        meta['offset'] += ltrb[:2]

        self.log.debug('valid area before crop of %s: %s', ltrb, meta['valid_area'])
        # process crops from left and top
        meta['valid_area'][:2] = np.maximum(0.0, meta['valid_area'][:2] - ltrb[:2])
        meta['valid_area'][2:] = np.maximum(0.0, meta['valid_area'][2:] - ltrb[:2])
        # process cropps from right and bottom
        meta['valid_area'][2:] = np.minimum(meta['valid_area'][2:], ltrb[2:] - ltrb[:2])
        self.log.debug('valid area after crop: %s', meta['valid_area'])

        for ann in anns:
            ann['valid_area'] = meta['valid_area']

        return image, mask, anns, meta

    def crop(self, image, mask, anns):
        w, h = image.size
        padding = int(self.long_edge / 2.0)
        x_offset, y_offset = 0, 0
        if w > self.long_edge:
            x_offset = torch.randint(-padding, w - self.long_edge + padding, (1,))
            x_offset = torch.clamp(x_offset, min=0, max=w - self.long_edge).item()
        if h > self.long_edge:
            y_offset = torch.randint(-padding, h - self.long_edge + padding, (1,))
            y_offset = torch.clamp(y_offset, min=0, max=h - self.long_edge).item()
        self.log.debug('crop offsets (%d, %d)', x_offset, y_offset)

        # crop image
        new_w = min(self.long_edge, w - x_offset)
        new_h = min(self.long_edge, h - y_offset)
        ltrb = (x_offset, y_offset, x_offset + new_w, y_offset + new_h)
        image = image.crop(ltrb)
        if mask:
            mask = mask.crop(ltrb)

        # crop keypoints
        for ann in anns:
            ann['keypoints'][:, 0] -= x_offset
            ann['keypoints'][:, 1] -= y_offset
            ann['bbox'][0] -= x_offset
            ann['bbox'][1] -= y_offset

        return image, mask, anns, np.array(ltrb)

# ターゲットサイズで中央配置のパディング
class CenterPad(Preprocess):
    def __init__(self, target_size):
        self.log = logging.getLogger(self.__class__.__name__)

        if isinstance(target_size, int):
            target_size = (target_size, target_size)
        self.target_size = target_size

    def __call__(self, image, mask, anns, meta):
        meta = copy.deepcopy(meta)
        anns = copy.deepcopy(anns)

        image, mask, anns, ltrb = self.center_pad(image, mask, anns)
        meta['offset'] -= ltrb[:2]

        self.log.debug('valid area before pad with %s: %s', ltrb, meta['valid_area'])
        meta['valid_area'][:2] += ltrb[:2]
        self.log.debug('valid area after pad: %s', meta['valid_area'])

        for ann in anns:
            ann['valid_area'] = meta['valid_area']

        return image, mask, anns, meta

    def center_pad(self, image, mask, anns):
        w, h = image.size
        left = int((self.target_size[0] - w) / 2.0)
        top = int((self.target_size[1] - h) / 2.0)
        ltrb = (
            left,
            top,
            self.target_size[0] - w - left,
            self.target_size[1] - h - top,
        )

        # pad image
        image = torchvision.transforms.functional.pad(
            image, ltrb, fill=(124, 116, 104))
        if mask:
            mask = torchvision.transforms.functional.pad(mask, ltrb, fill=0)

        # pad annotations
        for ann in anns:
            ann['keypoints'][:, 0] += ltrb[0]
            ann['keypoints'][:, 1] += ltrb[1]
            ann['bbox'][0] += ltrb[0]
            ann['bbox'][1] += ltrb[1]

        return image, mask, anns, ltrb


# 水平方向の反転
class HFlip(Preprocess):
    def __init__(self, *, swap=horizontal_swap_coco):
        self.swap = swap

    def __call__(self, image, mask, anns, meta):
        meta = copy.deepcopy(meta)
        anns = copy.deepcopy(anns)

        w, _ = image.size
        image = image.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        if mask:
            mask = mask.transpose(PIL.Image.FLIP_LEFT_RIGHT)

        for ann in anns:
            ann['keypoints'][:, 0] = -ann['keypoints'][:, 0] - 1.0 + w
            if self.swap is not None:
                ann['keypoints'] = self.swap(ann['keypoints'])
                meta['horizontal_swap'] = self.swap
            ann['bbox'][0] = -(ann['bbox'][0] + ann['bbox'][2]) - 1.0 + w

        assert meta['hflip'] is False
        meta['hflip'] = True

        meta['valid_area'][0] = -(meta['valid_area'][0] + meta['valid_area'][2]) + w
        for ann in anns:
            ann['valid_area'] = meta['valid_area']

        return image, mask, anns, meta

# ランダム適用の際に利用
class RandomApply(Preprocess):
    def __init__(self, transform, probability):
        self.transform = transform
        self.probability = probability

    def __call__(self, image, mask, anns, meta):
        if float(torch.rand(1).item()) > self.probability:
            return image, mask, anns, meta
        return self.transform(image, mask, anns, meta)


# ランダム回転
class RandomRotate(Preprocess):
    def __init__(self, max_rotate_degree=25):   # 最大回転角度
        super().__init__()
        self.log = logging.getLogger(self.__class__.__name__)

        self.max_rotate_degree =  max_rotate_degree

    def __call__(self, image, mask, anns, meta):

        meta = copy.deepcopy(meta)
        anns = copy.deepcopy(anns)
        w, h = image.size
        
        dice = random.random()
        degree = (dice - 0.5) * 2 * self.max_rotate_degree  # 角度 -40,40 の範囲で

        img_rot, R = self.rotate_bound(np.asarray(image), np.copy(degree), (128, 128, 128))
        image = PIL.Image.fromarray(img_rot)

        if mask:
            mask_rot, _ = self.rotate_bound(np.asarray(mask), np.copy(degree), 0)
            mask = PIL.Image.fromarray(mask_rot)

        for j,ann in enumerate(anns):
            for k in range(17):
                xy = ann['keypoints'][k, :2]     
                new_xy = self.rotatepoint(xy, R)
                anns[j]['keypoints'][k, :2] = new_xy
                
            ann['bbox'] = self.rotate_box(ann['bbox'], R)

        self.log.debug('meta before: %s', meta)
        meta['valid_area'] = self.rotate_box(meta['valid_area'], R)
        self.log.debug('meta after: %s', meta)

        for ann in anns:
            ann['valid_area'] = meta['valid_area']

        return image, mask, anns, meta

    @staticmethod
    def rotatepoint(p, R):
        point = np.zeros((3, 1))
        point[0] = p[0]
        point[1] = p[1]
        point[2] = 1

        new_point = R.dot(point)

        p[0] = new_point[0]

        p[1] = new_point[1]
        return p
    
    # 画像の正しい回転方法
    # http://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/
   
    def rotate_bound(self, image, angle, bordervalue):
        # 画像の寸法から中心点を計算
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)

        # 回転行列の取得（時計回りに回転させる角度にマイナスを適用）
        # sin, cosは行列の回転成分
        M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        # 画像の新しい寸法を計算
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        # 平行移動を考慮して回転行列を更新
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY

        # 回転を適用して画像を返す
        return cv2.warpAffine(image, M, (nW, nH), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,
                              borderValue=bordervalue), M    
    
    # bboxの整形
    def rotate_box(self, bbox, R):
        """入力されるbboxの形式は x, y, width, height"""
        four_corners = np.array([
            [bbox[0], bbox[1]],
            [bbox[0] + bbox[2], bbox[1]],
            [bbox[0], bbox[1] + bbox[3]],
            [bbox[0] + bbox[2], bbox[1] + bbox[3]],
        ])
        
        new_four_corners = []
        for i in range(4):
            xy = self.rotatepoint(four_corners[i], R)
            new_four_corners.append(xy)
        
        new_four_corners = np.array(new_four_corners)

        x = np.min(new_four_corners[:, 0])
        y = np.min(new_four_corners[:, 1])
        xmax = np.max(new_four_corners[:, 0])
        ymax = np.max(new_four_corners[:, 1])

        return np.array([x, y, xmax - x, ymax - y])        
