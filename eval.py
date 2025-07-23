import os

import cv2
import numpy as np
import argparse
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

from lib.config import cfg
from lib.utils.common import draw_humans
from lib.utils.paf_to_pose import paf_to_pose_cpp

from lib.evaluate.estimator import load_ckpt, get_outputs, get_using_device

# モデルのインポート
from lib.network.networks import get_model

'''
MS COCO annotation order:
0: nose	   		1: l eye		2: r eye	3: l ear	4: r ear
5: l shoulder	6: r shoulder	7: l elbow	8: r elbow
9: l wrist		10: r wrist		11: l hip	12: r hip	13: l knee
14: r knee		15: l ankle		16: r ankle

The order in this work:
(0-'nose'	1-'neck' 2-'right_shoulder' 3-'right_elbow' 4-'right_wrist'
5-'left_shoulder' 6-'left_elbow'	    7-'left_wrist'  8-'right_hip'
9-'right_knee'	 10-'right_ankle'	11-'left_hip'   12-'left_knee'
13-'left_ankle'	 14-'right_eye'	    15-'left_eye'   16-'right_ear'
17-'left_ear' )
'''

ORDER_COCO = [0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10]
DATA_DIR = './data/'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default='vgg2016', help='使用するモデル名')
    parser.add_argument('-c', '--ckpt', type=str, help='pthファイルのパス')
    parser.add_argument('-d', '--datasets', type=str, help='データセットディレクトリ名')
    parser.add_argument('--mode', type=str, default='val', help='評価に使うディレクトリ名')
    parser.add_argument('--save', type=int, default=1, help='可視化結果を保存間隔(0で保存しない)')
    parser.add_argument('--json', action='store_true', help='jsonデータを保存するかどうか')
    parser.add_argument('--preprocess', type=str, default='vgg', choices=['vgg', 'rtpose'], help='前処理の種類')
    args = parser.parse_args()

    device = get_using_device()

    # データのパス設定
    args.test_image_dir = os.path.join(DATA_DIR, args.datasets, 'images', args.mode) 
    args.test_annotation_file = os.path.join(DATA_DIR, args.datasets, f'annotations_{args.mode}.json')

    # モデルの選択
    model = get_model(model_name=args.model)

    # チェックポイントのロード
    model = load_ckpt(model, args.ckpt, device)

    save_flag = args.save

    # 評価
    run_eval(
        image_dir=args.test_image_dir,
        anno_file=args.test_annotation_file,
        vis_dir='results/',
        model=model,
        preprocess='vgg',
        device=device,
        args=args)


def eval_coco(outputs, annFile, imgIds, args):
    """coco形式で画像を評価"""
    with open('./results/results.json', 'w') as f:
        json.dump(outputs, f)
    cocoGt = COCO(annFile)  # load annotations
    cocoDt = cocoGt.loadRes('./results/results.json')  # load model outputs

    # running evaluation
    cocoEval = COCOeval(cocoGt, cocoDt, 'keypoints')
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    if args.json is False:
        os.remove('./results/results.json')

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
        keypoints = np.zeros((18, 3))       # 18 キーポイント

        all_scores = []
        for i in range(18):     # 18 キーポイント
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

        keypoints = keypoints[ORDER_COCO,:]
        one_result["score"] = 1.
        one_result["keypoints"] = list(keypoints.reshape(51))   # 17 * 3

        outputs.append(one_result)


        
def run_eval(image_dir, anno_file, vis_dir, model, preprocess, device, args):
    """評価"""
    
    coco = COCO(anno_file)
    cat_ids = coco.getCatIds(catNms=['person'])    
    img_ids = coco.getImgIds(catIds=cat_ids)
    print(f"INFO: Test Data: {len(img_ids)}")

    # iterate all val images
    outputs = []
    print("\nvvvvvvvvvvv Start Test vvvvvvvvvvv\n")

    for i in tqdm(range(1000)):
        
        img = coco.loadImgs(img_ids[i])[0]
        file_name = img['file_name']
        file_path = os.path.join(image_dir, file_name)

        oriImg = cv2.imread(file_path)

        # モデルの出力を取得
        paf, heatmap, scale_img = get_outputs(oriImg, model,  preprocess, device)

        # pafprocess
        humans = paf_to_pose_cpp(heatmap, paf, cfg)
        
        # 骨格を描画
        if args.save != 0:
            if i % args.save == 0:
                out = draw_humans(oriImg, humans)
                vis_path = os.path.join(vis_dir, file_name)
                cv2.imwrite(vis_path, out)

        # subsetはこの画像に写っている人物の人数を示す
        upsample_keypoints = (heatmap.shape[0]*cfg.MODEL.DOWNSAMPLE/scale_img, heatmap.shape[1]*cfg.MODEL.DOWNSAMPLE/scale_img)
        append_result(img_ids[i], humans, upsample_keypoints, outputs)

    # 評価を出力
    return eval_coco(outputs=outputs, annFile=anno_file, imgIds=img_ids, args=args)


if __name__ == "__main__":
    main()
