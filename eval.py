import os
import time

import cv2
import numpy as np
import argparse
import json
from pycocotools.coco import COCO
from lib.evaluate.cocoeval import COCOeval
from tqdm import tqdm

from lib.config import cfg
from lib.utils.common import draw_humans
from lib.utils.paf_to_pose import paf_to_pose_cpp

from lib.evaluate.estimator import load_ckpt, get_outputs, get_using_device

# モデルのインポート
from lib.network.networks import get_model

'''
(0-'head' 1-'neck' 2-'left_shoulder' 3-'left_elbow'  4-'left_wrist'  5-'right_shoulder'
6-'right_elbow'  7-'right_wrist'  8-'left_hip'  9-'left_knee'  10-'left_ankle'
11-'right_hip'  12-'right_knee'  13-'right_ankle')
'''

ORDER_CUSTOM = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
DATA_DIR = './data/'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default='vgg2016', help='使用するモデル名')
    parser.add_argument('-c', '--ckpt', type=str, help='pthファイルのパス')
    parser.add_argument('-d', '--data_dir', type=str, help='データセットディレクトリ名')
    parser.add_argument('--preprocess', type=str, default='vgg', choices=['vgg', 'rtpose'], help='前処理の種類')
    args = parser.parse_args()

    device = get_using_device()

    # データのパス設定
    args.test_image_dir = os.path.join(DATA_DIR, args.data_dir, 'images/val') 
    args.test_annotation_file = os.path.join(DATA_DIR, args.data_dir, 'annotations_val.json')


    # モデルの選択
    model = get_model(model_name=args.model)

    # チェックポイントのロード
    model = load_ckpt(model, args.ckpt, device)

    # 評価
    run_eval(image_dir=args.test_image_dir, anno_file=args.test_annotation_file, vis_dir='results/', model=model, preprocess='vgg', device=device)


def eval_coco(outputs, annFile, imgIds):
    """coco形式で画像を評価"""
    with open('results.json', 'w') as f:
        json.dump(outputs, f)  
    cocoGt = COCO(annFile)  # load annotations
    cocoDt = cocoGt.loadRes('results.json')  # load model outputs

    # running evaluation
    cocoEval = COCOeval(cocoGt, cocoDt, 'keypoints')
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    # os.remove('results.json')

    # return Average Precision
    return cocoEval.stats[0]


def append_result(image_id, humans, upsample_keypoints, outputs):
    """ 評価用の形式に変換して結果を追加する"""

    for human in humans:
        one_result = {
            "image_id": 0,
            "category_id": 1,
            "keypoints": [],
            "score": 0
        }
        one_result["image_id"] = image_id
        keypoints = np.zeros((14, 3))       # 14 キーポイント
        
        all_scores = []
        for i in range(14):     # 14 キーポイント
            if i not in human.body_parts.keys():
                keypoints[i, 0] = 0
                keypoints[i, 1] = 0
                keypoints[i, 2] = 0
            else:
                body_part = human.body_parts[i]
                center = (body_part.x * upsample_keypoints[1] + 0.5, body_part.y * upsample_keypoints[0] + 0.5)           
                keypoints[i, 0] = center[0]
                keypoints[i, 1] = center[1]
                keypoints[i, 2] = 1     
                score = human.body_parts[i].score 
                all_scores.append(score)

        keypoints = keypoints[ORDER_CUSTOM,:]
        one_result["score"] = 1.
        one_result["keypoints"] = list(keypoints.reshape(39))   # 13 * 3

        outputs.append(one_result)


        
def run_eval(image_dir, anno_file, vis_dir, model, preprocess, device):
    """評価"""
    
    coco = COCO(anno_file)
    cat_ids = coco.getCatIds(catNms=['person'])    
    img_ids = coco.getImgIds(catIds=cat_ids)
    print(f"INFO: Test Data: {len(img_ids)}")

    # iterate all val images
    outputs = []
    print("\nvvvvvvvvvvv Start Test vvvvvvvvvvv\n")

    for i in tqdm(range(len(img_ids))):
        
        img = coco.loadImgs(img_ids[i])[0]
        file_name = img['file_name']
        file_path = os.path.join(image_dir, file_name)

        oriImg = cv2.imread(file_path)
        # 画像の最も短い辺を取得
        shape_dst = np.min(oriImg.shape[0:2])

        # モデルの出力を取得
        paf, heatmap, scale_img = get_outputs(oriImg, model,  preprocess, device)

        # pafprocess
        humans = paf_to_pose_cpp(heatmap, paf, cfg)
        
        # 骨格を描画
        out = draw_humans(oriImg, humans)
            
        vis_path = os.path.join(vis_dir, file_name)
        cv2.imwrite(vis_path, out)
        # subsetはこの画像に写っている人物の人数を示す
        upsample_keypoints = (heatmap.shape[0]*cfg.MODEL.DOWNSAMPLE/scale_img, heatmap.shape[1]*cfg.MODEL.DOWNSAMPLE/scale_img)
        append_result(img_ids[i], humans, upsample_keypoints, outputs)

    # 評価を出力
    return eval_coco(outputs=outputs, annFile=anno_file, imgIds=img_ids)


if __name__ == "__main__":
    main()
