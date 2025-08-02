from torch.utils.data import ConcatDataset, DataLoader
from lib.datasets import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np


def data_loader(image_dir, mask_dir, ann_file, preprocess, target_transforms):
    """データローダーを作成"""

    # 検証データ
    print("Loading val dataset...")
    val_data = datasets.CocoKeypoints(
        root=image_dir,
        mask_dir=mask_dir,
        annFile=ann_file,
        preprocess=preprocess,
        image_transform=transforms.image_transform_train,
        target_transforms=target_transforms,
        n_images=None,
        input_x=368,
        input_y=368,
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


if __name__ == "__main__":
        # 訓練データの読み込み
    preprocess = transforms.Compose([
        transforms.Normalize(),
        transforms.RandomApply(transforms.HFlip(), probability=0.5),
        transforms.RandomApply(transforms.RandomRotate(), probability=0.5),
        transforms.RescaleRelative(),
        transforms.Crop(368),
        transforms.CenterPad(368),
    ])

    image_dir = "data/coco2017/images/val"
    mask_dir = "data/coco2017/masks/val"
    ann_file = "data/coco2017/annotations_val.json"

    val_loader, val_data = data_loader(image_dir, mask_dir, ann_file, preprocess, target_transforms=None)

    for img, heatmap, paf, mask in val_loader:
        img = img.numpy().transpose(0, 2, 3, 1)[0]  # (H, W, C)
        mask = mask.numpy().transpose(0, 2, 3, 1)[0]  # (H, W, 1)
        mask_to_show = mask.squeeze() # (H, W) に

        # heatmap と paf はチャンネル数が多いので、例として最初のチャンネルのみ表示
        heatmap = heatmap.numpy()
        last_channel_index = heatmap.shape[1] - 1  # 最後のチャネル番号
        heatmap = heatmap[0, last_channel_index] 
        paf = paf.numpy()[0, 0]  # (H, W)

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 4, 1)
        plt.imshow(img)
        plt.title("Image")
        plt.axis('off')

        plt.subplot(1, 4, 2)
        plt.imshow(mask_to_show, cmap='gray', vmin=0, vmax=1)
        plt.title("Mask")
        plt.axis('off')

        plt.subplot(1, 4, 3)
        plt.imshow(heatmap, cmap='hot')
        plt.title("Heatmap (ch 0)")
        plt.axis('off')

        plt.subplot(1, 4, 4)
        plt.imshow(paf, cmap='bwr')
        plt.title("PAF (ch 0)")
        plt.axis('off')

        plt.tight_layout()
        plt.show()