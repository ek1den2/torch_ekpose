import os
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils

def main():    
    # データセットのパス設定
    train_json = 'coco2017/annotations_train.json'
    val_json = 'coco2017/annotations_val.json'

    train_output_dir = 'coco2017/masks/train/'
    val_output_dir = 'coco2017/masks/val/'
    if not os.path.exists(train_output_dir):
        os.makedirs(train_output_dir)
    if not os.path.exists(val_output_dir):
        os.makedirs(val_output_dir)

    # 訓練データの処理
    if os.path.exists(train_json):
        print("Processing training data...")
        process_coco_masks(train_json, train_output_dir)

    # 検証データの処理
    if os.path.exists(val_json):
        print("Processing validation data...")
        process_coco_masks(val_json, val_output_dir)

def process_coco_masks(json_file, output_dir):
    """COCOデータセットからマスクを生成（例外処理を追加した最終版）"""
    
    coco = COCO(json_file)
    os.makedirs(output_dir, exist_ok=True)
    img_ids = coco.getImgIds()
    processed_data = []
    
    for i, img_id in enumerate(img_ids):
        print(f"Processing image {i+1}/{len(img_ids)} (ID: {img_id})")
        
        img_info = coco.loadImgs(img_id)[0]
        height, width = img_info['height'], img_info['width']
        
        mask_miss_name = f"mask_{img_id:012d}.png"

        mask_miss_path = os.path.join(output_dir, mask_miss_name)
        
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        
        ignore_mask = np.zeros((height, width), dtype=bool)
        
        for ann in anns:
            segmentation = ann.get('segmentation')
            if not segmentation:
                continue

            # RLE形式
            if isinstance(segmentation, dict):
                try:
                    mask = maskUtils.decode(segmentation)
                    ignore_mask = np.logical_or(ignore_mask, mask.astype(bool))
                except TypeError as e:
                    print(f"  [Warning] Skipping malformed RLE annotation (ID: {ann['id']}). Error: {e}")
                    continue

            # Polygon形式
            elif isinstance(segmentation, list):
                rles = maskUtils.frPyObjects(segmentation, height, width)
                rle = maskUtils.merge(rles)
                current_ann_mask = maskUtils.decode(rle).astype(bool)
                
                if ann.get('num_keypoints', 0) <= 0 or ann.get('iscrowd', 0) == 1:
                    ignore_mask = np.logical_or(ignore_mask, current_ann_mask)

        Image.fromarray((~ignore_mask * 255).astype(np.uint8)).save(mask_miss_path)


if __name__ == "__main__":
    main()
