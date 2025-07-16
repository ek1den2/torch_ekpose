import torch


from . import vgg2016
from . import mobilenet
from . import mobilenetV2
from . import shufflenetV2


def get_model(
        model_name='vgg2016',
        pretrained_path=None,
        imagenet_pretrained=False,
        conv_width=1.0,
        conv_width2=1.0):

    if model_name == 'vgg2016':
        return vgg2016.load_model(
            pretrained_path=pretrained_path,
            imagenet_pretrained=imagenet_pretrained
        )
    
    elif model_name == 'mobilenet':
        return mobilenet.load_model(
            pretrained_path=pretrained_path,
            conv_width=conv_width,
            conv_width2=conv_width2
        )
    
    elif model_name == 'mobilenetV2':
        return mobilenetV2.load_model(
            pretrained_path=pretrained_path,
            conv_width=conv_width,
            conv_width2=conv_width2
        )
    
    elif model_name == 'shufflenetV2':
        return shufflenetV2.load_model(
            pretrained_path=pretrained_path,
            conv_width=conv_width,
            conv_width2=conv_width2
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
        'mobilenetV2',
        'shufflenetV2'
    ]


    print("使用可能モデル:")
    for model in model_list:
        print(f"- {model}")    

    model_name = input("モデル名を入力（デフォルトはVGG2016）") or 'vgg2016'
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
    dummy_input = torch.randn(1, 3, 368, 368)

    # モデルのパラメータを確認
    flops, params = thop.profile(model, inputs=(dummy_input, ))
    gflops = flops / 1e9
    print(f"FLOPs: {flops} ({gflops:.2f} GFLOPs)")
    print(f"Parameters: {params}")

    # 計算グラフを出力
    MODELNAME = model_name + "_conv" + str(conv_width) + "_conv2" + str(conv_width2)

    _, saved_for_loss = model(dummy_input)
    out_dir = f"../../experiments/img_network/{MODELNAME}"
    os.makedirs(out_dir, exist_ok=True)
    g = make_dot(saved_for_loss[-2], params=dict(model.named_parameters()))
    g.render("../../experiments/img_network/" + MODELNAME + "/pafs_model", format="png")
    g = make_dot(saved_for_loss[-1], params=dict(model.named_parameters()))
    g.render("../../experiments/img_network/" + MODELNAME + "/cmap_output", format="png")

    writer = SummaryWriter("../../experiments/img_network/" + MODELNAME + "/tbX/")
    writer.add_graph(model, (dummy_input, ))
    writer.close()

    # tensorboard --logdir=../../experiments/img_network/mobilenet/tbX/  でネットワークを可視化

    summary(model, input_size=(1, 3, 368, 368), depth=4)
