# torch_ekpose

PyTorchによるtf-pose-estimation実装

# インストール


# 学習
```s
$ python train.py -m vgg2016 -d coco -b 128 -e 300 --gpus 0,1,2,3 --training_curve
```


# 評価
```s
$ python eval.py --model vgg2016 --ckpt ./checkpoints/vgg2016/best_epoch.pth --datasets coco
```

# 実行
### ・画像　`run_image.py`
```s
$ python run_image.py --model vgg2016 --ckpt ./checkpoints/vgg2016/best_epoch.pth --image demo.jpeg
```
--imageを指定しない場合、./data/のすべての画像データに対して推定を行う。


### ・動画　`run_video.py`
```s
$ python run_video.py --model vgg2016 --ckpt ./checkpoints/vgg2016/best_epoch.pth --video demo.mp4
```

### ・ウェブカメラ　`run_webcam.py`
```s
$ python run_webcam.py --model vgg2016 --ckpt ./checkpoints/vgg2016/best_epoch.pth
``` 
