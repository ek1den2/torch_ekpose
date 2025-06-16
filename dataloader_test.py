from lib.dataloader import pose_datasets, transforms
from torch.utils.data import ConcatDataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import numpy as np
from PIL import Image


def data_loader(preprocess, target_transform):
    """データローダーを作成"""
    
    val_image_dir = 'data/coco2014/images/val'
    val_annotation_file = 'data/coco2014/annotations_val.json'
    # 検証データ
    print("Loading val dataset...")
    val_data = pose_datasets.OriginalKeyPoints(
        root=val_image_dir,
        annFile=val_annotation_file,
        preprocess=preprocess,
        image_transform=transforms.image_transform_train,
        target_transform=target_transform,
        n_images=None,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        drop_last=False
    )

    return val_loader, val_data

preprocess = transforms.Compose([
    transforms.Normalize(),
    transforms.RandomApply(transforms.HFlip(), probability=0.5),
    transforms.RescaleRelative(),
    transforms.Crop(512),
    transforms.CenterCrop(512),
])

val_loader, val_data = data_loader(preprocess, target_transform=None)


for i, (img, heat, paf) in enumerate(val_loader):
    print(i)
    # img, heat, paf を (H, W, C) の形式に変換
    img = img.squeeze(0).permute(1, 2, 0).numpy()
    heat = heat.squeeze(0).permute(1, 2, 0).numpy()
    paf = paf.squeeze(0).permute(1, 2, 0).numpy()

    # heat = cv2.resize(heat[:, :, -1], (img.shape[1], img.shape[0]))

    # heat_colormap = cm.jet(heat)  # jet カラーマップを適用
    # heat_colormap = (heat_colormap[:, :, :3] * 255).astype(np.float32)
    # heat_colormap = np.asarray(heat_colormap)

    # # heatmapの最後のチャンネルを使ってヒートマップに変換
    # blended = cv2.addWeighted(img, 0.5, heat_colormap, 0.5, 0)


    paf = cv2.resize(paf[:, :, 1], (img.shape[1], img.shape[0]))

    paf_colormap = cm.jet(paf)  # jet カラーマップを適用
    paf_colormap = (paf_colormap[:, :, :3] * 255).astype(np.float32)
    paf_colormap = np.asarray(paf_colormap)

    # heatmapの最後のチャンネルを使ってヒートマップに変換
    blended = cv2.addWeighted(img, 0.5, paf_colormap, 0.5, 0)


    # 結果を表示
    plt.imshow(blended)
    plt.show()


