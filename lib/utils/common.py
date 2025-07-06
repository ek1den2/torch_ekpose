import cv2
from enum import Enum
import numpy as np
import math

class CustomPart(Enum):
    """カスタムキーポイントの定義"""
    Head = 0
    LShoulder = 1
    LElbow = 2
    LWrist = 3
    RShoulder = 4
    RElbow = 5
    RWrist = 6
    LHip = 7
    LKnee = 8
    LAnkle = 9
    RHip = 10
    RKnee = 11
    RAnkle = 12
    Neck = 13
    Background = 14

CustomPairs = [
    [0, 13], [13, 1], [1, 2], [2, 3], [4, 13], [4, 5], [5, 6],
    [13, 7], [7, 8], [8, 9], [13, 10], [10, 11], [11, 12]
]

CustomColors = [
    [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], 
    [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255],
    [0, 170, 255], [0, 85, 255], [0, 0, 255], [255, 0, 255]
]

CustomPairsRender = CustomPairs


def _round(value):
    """小数点以下を四捨五入して整数に変換"""
    return int(round(value))

def _include_part(part_list, part_idx):
    """part_listからpart_idxに対応するBodyPartを取得"""
    for part in part_list:
        if part_idx == part.part_idx:
            return True, part
    return False, None


class Human:
    """body_parts: ボディ部分のlist"""

    __slots__ = ('body_parts', 'pairs', 'uidx_list', 'score')

    def __init__(self, pairs):
        self.pairs = []
        self.uidx_list = set()
        self.body_parts = {}
        for pair in pairs:
            self.add_pair(pair)
        self.score = 0.0

    @staticmethod
    def _get_uidx(part_idx, idx):
        return '%d-%d' % (part_idx, idx)

    def add_pair(self, pair):
        self.pairs.append(pair)
        self.body_parts[pair.part_idx1] = BodyPart(Human._get_uidx(pair.part_idx1, pair.idx1),
                                                   pair.part_idx1,
                                                   pair.coord1[0], pair.coord1[1], pair.score)
        self.body_parts[pair.part_idx2] = BodyPart(Human._get_uidx(pair.part_idx2, pair.idx2),
                                                   pair.part_idx2,
                                                   pair.coord2[0], pair.coord2[1], pair.score)
        self.uidx_list.add(Human._get_uidx(pair.part_idx1, pair.idx1))
        self.uidx_list.add(Human._get_uidx(pair.part_idx2, pair.idx2))

    def is_connected(self, other):
        return len(self.uidx_list & other.uidx_list) > 0

    def merge(self, other):
        for pair in other.pairs:
            self.add_pair(pair)

    def part_count(self):
        return len(self.body_parts.keys())

    def get_max_score(self):
        return max([x.score for _, x in self.body_parts.items()])

    def get_face_box(self, img_w, img_h, mode=0):
        """画像サイズ(w, h)と比較した顔のboxを取得"""
        # SEE : https://github.com/ildoonet/tf-pose-estimation/blob/master/tf_pose/common.py#L13
        _HEAD = CustomPart.Head.value
        _NECK = CustomPart.Neck.value
        _LSHO = CustomPart.LShoulder.value
        _RSHO = CustomPart.RShoulder.value

        _THRESHOLD_PART_CONFIDENCE = 0.2
        parts = [part for idx, part in self.body_parts.items() if part.score > _THRESHOLD_PART_CONFIDENCE]

        # headがない場合はNone
        is_head, part_head = _include_part(parts, _HEAD)
        if not is_head:
            return None

        size = 0
        is_neck, part_neck = _include_part(parts, _NECK)
        if is_neck:
            size = max(size, img_h * (part_neck.y - part_head.y) * 0.8)

        is_lsho, part_lsho = _include_part(parts, _LSHO)
        is_rsho, part_rsho = _include_part(parts, _RSHO)
        if is_rsho and is_lsho:
            size = max(size, img_w * (part_rsho.x - part_lsho.x) * 0.5)
            size = max(size,
                       img_w * math.sqrt((part_rsho.x - part_lsho.x) ** 2 + (part_rsho.y - part_lsho.y) ** 2) * 0.5)

        if size <= 0:
            return None

        # 顔のboxを計算
        x = max(0, x)
        y = max(0, y)
        x2 = min(img_w - x, x2 - x) + x
        y2 = min(img_h - y, y2 - y) + y

        if _round(x2 - x) == 0.0 or _round(y2 - y) == 0.0:
            return None
        if mode == 0:
            return {"x": _round((x + x2) / 2),  # 中心x
                    "y": _round((y + y2) / 2),  # 中心y
                    "w": _round(x2 - x),        # 幅
                    "h": _round(y2 - y)}        # 高さ
        else:
            return {"x": _round(x),             # 左上x
                    "y": _round(y),             # 左上y  
                    "w": _round(x2 - x),        # 幅
                    "h": _round(y2 - y)}        # 高さ


    def get_upper_body_box(self, img_w, img_h):
        """画像サイズ(w, h)と比較した上半身のboxを取得"""

        if not (img_w > 0 and img_h > 0):
            raise Exception("img size should be positive")

        _HEAD = CustomPart.Head.value
        _NECK = CustomPart.Neck.value
        _RSHO = CustomPart.RShoulder.value
        _LSHO = CustomPart.LShoulder.value
        _THRESHOLD_PART_CONFIDENCE = 0.3
        parts = [part for idx, part in self.body_parts.items() if part.score > _THRESHOLD_PART_CONFIDENCE]
        part_coords = [(img_w * part.x, img_h * part.y) for part in parts if
                       part.part_idx in [0, 1, 4, 7, 10, 13]]

        if len(part_coords) < 4:
            return None

        # 検出されたボディパーツをちょうど囲むbboxで初期化
        x = min([part[0] for part in part_coords])
        y = min([part[1] for part in part_coords])
        x2 = max([part[0] for part in part_coords])
        y2 = max([part[1] for part in part_coords])

        # # ------ ヒューリスティックに基づくbboxの調整 +
        # 頭の位置でbboxの上端を調整

        is_head, part_head = _include_part(parts, _HEAD)
        is_neck, part_neck = _include_part(parts, _NECK)
        torso_height = 0
        if is_head and is_neck:
            y -= (part_neck.y * img_h - y) * 0.8
            torso_height = max(0, (part_neck.y - part_head.y) * img_h * 2.5)

        # 肩の座標で幅を調整
        is_rshoulder, part_rshoulder = _include_part(parts, _RSHO)
        is_lshoulder, part_lshoulder = _include_part(parts, _LSHO)
        if is_rshoulder and is_lshoulder:
            half_w = x2 - x
            dx = half_w * 0.15
            x -= dx
            x2 += dx
        elif is_neck:
            if is_lshoulder and not is_rshoulder:
                half_w = abs(part_lshoulder.x - part_neck.x) * img_w * 1.15
                x = min(part_neck.x * img_w - half_w, x)
                x2 = max(part_neck.x * img_w + half_w, x2)
            elif not is_lshoulder and is_rshoulder:
                half_w = abs(part_rshoulder.x - part_neck.x) * img_w * 1.15
                x = min(part_neck.x * img_w - half_w, x)
                x2 = max(part_neck.x * img_w + half_w, x2)

        # ------ ヒューリスティックに基づく調整 -

        # 画像への座標調整
        x = max(0, x)
        y = max(0, y)
        x2 = min(img_w - x, x2 - x) + x
        y2 = min(img_h - y, y2 - y) + y

        if _round(x2 - x) == 0.0 or _round(y2 - y) == 0.0:
            return None
        return {"x": _round((x + x2) / 2),
                "y": _round((y + y2) / 2),
                "w": _round(x2 - x),
                "h": _round(y2 - y)}

    def __str__(self):
        return ' '.join([str(x) for x in self.body_parts.values()])

    def __repr__(self):
        return self.__str__()

def draw_humans(npimg, humans, imgcopy=False):
    if imgcopy:
        npimg = np.copy(npimg)
    image_h, image_w = npimg.shape[:2]
    centers = {}
    for human in humans:
        # 点を描画
        for i in range(CustomPart.Background.value):
            if i not in human.body_parts.keys():
                continue

            body_part = human.body_parts[i]
            center = (int(body_part.x * image_w + 0.5), int(body_part.y * image_h + 0.5))
            centers[i] = center
            cv2.circle(npimg, center, 3, CustomColors[i], thickness=3, lineType=8, shift=0)

        # 線を描画
        for pair_order, pair in enumerate(CustomPairsRender):
            if pair[0] not in human.body_parts.keys() or pair[1] not in human.body_parts.keys():
                continue

            # npimg = cv2.line(npimg, centers[pair[0]], centers[pair[1]], common.CustomColors[pair_order], 3)
            cv2.line(npimg, centers[pair[0]], centers[pair[1]], CustomColors[pair_order], 3)

    return npimg
    
class BodyPart:
    """
    part_idx : キーポイントのインデックス（eg. 頭は0）
    x, y: キーポイントの座標
    score : 信頼度スコア
    """
    __slots__ = ('uidx', 'part_idx', 'x', 'y', 'score')

    def __init__(self, uidx, part_idx, x, y, score):
        self.uidx = uidx
        self.part_idx = part_idx
        self.x, self.y = x, y
        self.score = score

    def get_part_name(self):
        return CustomPart(self.part_idx)

    def __str__(self):
        return 'BodyPart:%d-(%.2f, %.2f) score=%.2f' % (self.part_idx, self.x, self.y, self.score)

    def __repr__(self):
        return self.__str__()
