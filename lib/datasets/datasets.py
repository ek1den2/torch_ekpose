import copy
import io
import logging
import os
import numpy as np
import torch.utils.data
import torchvision
from PIL import Image

from .heatmap import putGaussianMaps
from .paf import putVecMaps
from . import transforms, utils

def kp_connections(keypoints):
    kp_lines = [
        [keypoints.index('neck'), keypoints.index('head')],             # (0, 1)
        [keypoints.index('neck'), keypoints.index('l_shoulder')],       # (2, 3)
        [keypoints.index('l_shoulder'), keypoints.index('l_elbow')],    # (4, 5)
        [keypoints.index('l_elbow'), keypoints.index('l_wrist')],       # (6, 7)
        [keypoints.index('neck'), keypoints.index('r_shoulder')],       # (8, 9)
        [keypoints.index('r_shoulder'), keypoints.index('r_elbow')],    # (10, 11)
        [keypoints.index('r_elbow'), keypoints.index('r_wrist')],       # (12, 13)
        [keypoints.index('neck'), keypoints.index('l_hip')],            # (14, 15)
        [keypoints.index('l_hip'), keypoints.index('l_knee')],          # (16, 17)
        [keypoints.index('l_knee'), keypoints.index('l_ankle')],        # (18, 19)
        [keypoints.index('neck'), keypoints.index('r_hip')],            # (20, 21)
        [keypoints.index('r_hip'), keypoints.index('r_knee')],          # (22, 23)
        [keypoints.index('r_knee'), keypoints.index('r_ankle')],        # (24, 25)
        [keypoints.index('head'), keypoints.index('l_shoulder')],       # (26, 27)
        [keypoints.index('head'), keypoints.index('r_shoulder')]        # (28, 29)
    ]
    return kp_lines
    
def get_keypoints():
    """COCOキーポイントとその左右反転対応マップを取得"""

    keypoints = [
        'head',       # 1
        'neck',       # 2
        'l_shoulder', # 3
        'l_elbow',    # 4
        'l_wrist',    # 5
        'r_shoulder', # 6
        'r_elbow',    # 7
        'r_wrist',    # 8
        'l_hip',      # 9
        'l_knee',     # 10
        'l_ankle',    # 11
        'r_hip',      # 12
        'r_knee',     # 13
        'r_ankle'     # 14
    ]

    return keypoints
    
def collate_images_anns_meta(batch):
    images = torch.utils.data.dataloader.default_collate([b[0] for b in batch])
    anns = [b[1] for b in batch]
    metas = [b[2] for b in batch]
    return images, anns, metas


def collate_multiscale_images_anns_meta(batch):
    """マルチスケールのために照合"""
    
    n_scales = len(batch[0][0])
    images = [torch.utils.data.dataloader.default_collate([b[0][i] for b in batch])
              for i in range(n_scales)]
    anns = [[b[1][i] for b in batch] for i in range(n_scales)]
    metas = [b[2] for b in batch]
    return images, anns, metas


def collate_images_targets_meta(batch):

    images = torch.utils.data.dataloader.default_collate([b[0] for b in batch])
    targets1 = torch.utils.data.dataloader.default_collate([b[1] for b in batch])
    targets2 = torch.utils.data.dataloader.default_collate([b[2] for b in batch])    

    return images, targets1, targets2


class CustomKeypoints(torch.utils.data.Dataset):
    """独自データクラス"""

    def __init__(self, root, annFile, image_transform=None, target_transforms=None,
                 n_images=None, preprocess=None, all_images=False, all_persons=False,
                 input_y=368, input_x=368, stride=8):
        from pycocotools.coco import COCO
        from contextlib import redirect_stdout

        self.root = root
        with redirect_stdout(io.StringIO()):
            self.coco = COCO(annFile)

        self.cat_ids = self.coco.getCatIds(catNms=['person'])
        if all_images:
            self.ids = self.coco.getImgIds()
        elif all_persons:
            self.ids = self.coco.getImgIds(catIds=self.cat_ids)
        else:
            self.ids = self.coco.getImgIds(catIds=self.cat_ids)
            self.filter_for_keypoint_annotations()
        if n_images:
            self.ids = self.ids[:n_images]

        self.preprocess = preprocess or transforms.Normalize()
        self.image_transform = image_transform or transforms.image_transform
        self.target_transforms = target_transforms
        
        self.HEATMAP_COUNT = len(get_keypoints())
        self.LIMB_IDS = kp_connections(get_keypoints())
        self.input_y = input_y
        self.input_x = input_x        
        self.stride = stride
        self.log = logging.getLogger(self.__class__.__name__)

    def filter_for_keypoint_annotations(self):
        """キーポイントがある画像のみをフィルタリング"""
        def has_keypoint_annotation(image_id):
            ann_ids = self.coco.getAnnIds(imgIds=image_id, catIds=self.cat_ids)
            anns = self.coco.loadAnns(ann_ids)
            for ann in anns:
                if 'keypoints' not in ann:
                    continue
                if any(v > 0.0 for v in ann['keypoints'][2::3]):
                    return True
            return False

        self.ids = [image_id for image_id in self.ids
                    if has_keypoint_annotation(image_id)]


    def __getitem__(self, index):
        """データセットを取得"""
        image_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=image_id, catIds=self.cat_ids)
        anns = self.coco.loadAnns(ann_ids)
        anns = copy.deepcopy(anns)

        image_info = self.coco.loadImgs(image_id)[0]
        self.log.debug(image_info)
        with open(os.path.join(self.root, image_info['file_name']), 'rb') as f:
            image = Image.open(f).convert('RGB')

        meta_init = {
            'dataset_index': index,
            'image_id': image_id,
            'file_name': image_info['file_name'],
        }

        image, anns, meta = self.preprocess(image, anns, None)
             
        if isinstance(image, list):
            return self.multi_image_processing(image, anns, meta, meta_init)

        return self.single_image_processing(image, anns, meta, meta_init)

    def multi_image_processing(self, image_list, anns_list, meta_list, meta_init):
        return list(zip(*[
            self.single_image_processing(image, anns, meta, meta_init)
            for image, anns, meta in zip(image_list, anns_list, meta_list)
        ]))

    def single_image_processing(self, image, anns, meta, meta_init):
        """単一の画像を処理"""

        meta.update(meta_init)

        # transform image
        original_size = image.size
        image = self.image_transform(image)
        assert image.size(2) == original_size[0]
        assert image.size(1) == original_size[1]

        # 有効領域のマスク
        valid_area = meta['valid_area']
        utils.mask_valid_area(image, valid_area)

        self.log.debug(meta)

        # GTの生成
        heatmaps, pafs = self.get_ground_truth(anns)
        
        heatmaps = torch.from_numpy(
            heatmaps.transpose((2, 0, 1)).astype(np.float32))
            
        pafs = torch.from_numpy(pafs.transpose((2, 0, 1)).astype(np.float32))       
        return image, heatmaps, pafs

    def remove_illegal_joint(self, keypoints):
        """画面外のキーポイントを除去"""

        MAGIC_CONSTANT = (-1, -1, 0)
        mask = np.logical_or.reduce((keypoints[:, :, 0] >= self.input_x,
                                     keypoints[:, :, 0] < 0,
                                     keypoints[:, :, 1] >= self.input_y,
                                     keypoints[:, :, 1] < 0))
        keypoints[mask] = MAGIC_CONSTANT

        return keypoints
        
    def add_neck(self, keypoint):
        """キーポイントにneckを追加"""
        our_order = [0, 13, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

        # 1が左肩、4が右肩
        left_shoulder = keypoint[1, :]
        right_shoulder = keypoint[4, :]
        neck = (right_shoulder + left_shoulder) / 2
        if right_shoulder[2] == 2 and left_shoulder[2] == 2:
            neck[2] = 2
        else:
            neck[2] = right_shoulder[2] * left_shoulder[2]

        neck = neck.reshape(1, len(neck))
        neck = np.round(neck)
        keypoint = np.vstack((keypoint, neck))
        keypoint = keypoint[our_order, :]

        return keypoint
                
    def get_ground_truth(self, anns):
        """heatmapとPAFsの正解データを取得"""

        # 出力チャネル数を計算
        grid_y = int(self.input_y / self.stride)
        grid_x = int(self.input_x / self.stride)
        channels_heat = (self.HEATMAP_COUNT + 1)
        channels_paf = 2 * len(self.LIMB_IDS)
        heatmaps = np.zeros((int(grid_y), int(grid_x), channels_heat))
        pafs = np.zeros((int(grid_y), int(grid_x), channels_paf))

        keypoints = []
        for ann in anns:
            single_keypoints = np.array(ann['keypoints']).reshape(13,3) # 13キーポイント
            single_keypoints = self.add_neck(single_keypoints)
            keypoints.append(single_keypoints)
        keypoints = np.array(keypoints)
        keypoints = self.remove_illegal_joint(keypoints)

        # Confidence Maps
        for i in range(self.HEATMAP_COUNT):
            joints = [jo[i] for jo in keypoints]
            for joint in joints:
                if joint[2] > 0.5:
                    center = joint[:2]
                    gaussian_map = heatmaps[:, :, i]
                    heatmaps[:, :, i] = putGaussianMaps(
                        center, gaussian_map,
                        7.0, grid_y, grid_x, self.stride)

        # Parts Affinity Fields (PAFs)
        for i, (k1, k2) in enumerate(self.LIMB_IDS):
            # limb
            count = np.zeros((int(grid_y), int(grid_x)), dtype=np.uint32)
            for joint in keypoints:
                if joint[k1, 2] > 0.5 and joint[k2, 2] > 0.5:
                    centerA = joint[k1, :2]
                    centerB = joint[k2, :2]
                    vec_map = pafs[:, :, 2 * i:2 * (i + 1)]

                    pafs[:, :, 2 * i:2 * (i + 1)], count = putVecMaps(
                        centerA=centerA,
                        centerB=centerB,
                        accumulate_vec_map=vec_map,
                        count=count, grid_y=grid_y, grid_x=grid_x, stride=self.stride
                    )

        # 背景チャネル（14 "+1"の部分）。どのキーポイントでもない領域
        heatmaps[:, :, -1] = np.maximum(
            1 - np.max(heatmaps[:, :, :self.HEATMAP_COUNT], axis=2),
            0.
        )
        return heatmaps, pafs
        
    def __len__(self):
        return len(self.ids)


class ImageList(torch.utils.data.Dataset):
    def __init__(self, image_paths, preprocess=None, image_transform=None):
        self.image_paths = image_paths
        self.image_transform = image_transform or transforms.image_transform
        self.preprocess = preprocess

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        with open(image_path, 'rb') as f:
            image = Image.open(f).convert('RGB')

        if self.preprocess is not None:
            image = self.preprocess(image, [], None)[0]

        original_image = torchvision.transforms.functional.to_tensor(image)
        image = self.image_transform(image)

        return image_path, original_image, image

    def __len__(self):
        return len(self.image_paths)


class PilImageList(torch.utils.data.Dataset):
    def __init__(self, images, image_transform=None):
        self.images = images
        self.image_transform = image_transform or transforms.image_transform

    def __getitem__(self, index):
        pil_image = self.images[index].copy().convert('RGB')
        original_image = torchvision.transforms.functional.to_tensor(pil_image)
        image = self.image_transform(pil_image)

        return index, original_image, image

    def __len__(self):
        return len(self.images)

