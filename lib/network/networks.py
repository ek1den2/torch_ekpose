import torch


from . import vgg2016
from . import mobilenet
from . import mobilenetV2
from . import shufflenetV2


def get_model(
        model_name='vgg2016',
        pretrained_path=None,
        imagenet_pretrained=False):

    if model_name == 'vgg2016':
        return vgg2016.load_model(
            pretrained_path=pretrained_path,
            imagenet_pretrained=imagenet_pretrained
        )
    
    elif model_name == 'mobilenet':
        return mobilenet.load_model(
            pretrained_path=pretrained_path,
            conv_width=1.0,
            conv_width2=1.0
        )
    
    elif model_name == 'mobilenet_thin':
        return mobilenet.load_model(
            pretrained_path=pretrained_path,
            conv_width=0.75,
            conv_width2=0.50
        )
    
    elif model_name == 'mobilenetV2':
        return mobilenetV2.load_model(
            pretrained_path=pretrained_path,
            conv_width=1.0,
            conv_width2=1.0
        )
    
    elif model_name == 'mobilenetV2_large':
        return mobilenetV2.load_model(
            pretrained_path=pretrained_path,
            conv_width=1.4,
            conv_width2=1.0
        )
    
    elif model_name == 'mobilenetV2_small':
        return mobilenetV2.load_model(
            pretrained_path=pretrained_path,
            conv_width=0.50,
            conv_width2=0.50
        )
    
    elif model_name == 'shufflenetV2_1.0x':
        return shufflenetV2.load_model(
            pretrained_path=pretrained_path,
            conv_width=1.0,
            conv_width2=1.0
        )
    
    elif model_name == 'shufflenetV2_0.5x':
        return shufflenetV2.load_model(
            pretrained_path=pretrained_path,
            conv_width=0.5,
            conv_width2=0.5
        )

if __name__ == "__main__":
    import os
    import thop
    from torchinfo import summary
    from torchviz import make_dot
    from torch.utils.tensorboard import SummaryWriter

    model_list = [
        'vgg2016',
        'mobilenet',
        'mobilenet_thin',
        'mobilenetV2',
        'mobilenetV2_large',
        'mobilenetV2_small',
        'shufflenetV2',
        'shufflenetV2_1.0x',
        'shufflenetV2_0.5x'
    ]


    print("使用可能モデル:")
    for model in model_list:
        print(f"- {model}")    

    model_name = input("モデル名を入力（デフォルトはvgg2016）") or 'vgg2016'
    conv_width = float(input("Convの幅を入力（デフォルトは1.0）") or 1.0)
    conv_width2 = float(input("Conv2の幅を入力（デフォルトは1.0）") or 1.0)


    model = get_model(
        model_name=model_name,
        pretrained_path=None,
        imagenet_pretrained=False,
        conv_width=conv_width,
        conv_width2=conv_width2
    )

    print(model)
    
    # テスト用のダミー入力
    dummy_input = torch.randn(1, 3, 160, 160)

    # モデルのパラメータを確認
    flops, params = thop.profile(model, inputs=(dummy_input, ))
    gflops = flops / 1e9
    print(f"FLOPs: {flops} ({gflops:.2f} GFLOPs)")
    print(f"Parameters: {params}")

    # 計算グラフを出力
    MODELNAME = model_name + "_conv" + str(conv_width) + "_conv2" + str(conv_width2)

    _, saved_for_loss = model(dummy_input)
    out_dir = f"experiments/{MODELNAME}"
    os.makedirs(out_dir, exist_ok=True)

    g = make_dot(saved_for_loss[-2], params=dict(model.named_parameters()))
    g.render("experiments/" + MODELNAME + "/pafs_model", format="png")
    g = make_dot(saved_for_loss[-1], params=dict(model.named_parameters()))
    g.render("experiments/" + MODELNAME + "/cmap_output", format="png")

    writer = SummaryWriter("experiments/" + MODELNAME + "/tbX/")
    writer.add_graph(model, (dummy_input, ))
    writer.close()

    # tensorboard --logdir= experiments/mobilenet/tbX/  でネットワークを可視化

    summary(model, input_size=(1, 3, 368, 368), depth=4)
