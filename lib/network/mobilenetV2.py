import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
import math


# Depthwise Separable Convolution
class DSConv(nn.Module):
    def __init__(self, nin, nout, kernel_size=3, stride=1, padding=1, bias=False, relu=True):
        super(DSConv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=kernel_size, stride=stride, padding=padding, groups=nin, bias=bias)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1, bias=bias)
        self.bn = nn.BatchNorm2d(nout)
        self.activation = nn.ReLU(inplace=True) if relu else nn.Identity()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

def ConvBN(nin, nout, kernel_size=3, stride=1, padding=1, bias=False):
    return nn.Sequential(
        nn.Conv2d(nin, nout, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
        nn.BatchNorm2d(nout),
        nn.ReLU6(inplace=True)
    )

def Conv1x1BN(nin, nout, bias=False):
    return nn.Sequential(
        nn.Conv2d(nin, nout, kernel_size=1, stride=1, padding=0, bias=bias),
        nn.BatchNorm2d(nout),
        nn.ReLU6(inplace=True)
    )


class IRB(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(IRB, self).__init__()
        self.stride = stride
        assert stride in [1, 2]  # ストライドは1 or 2のみ 

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # Depthwise convolution
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # Pointwise-linear
                nn.Conv2d(hidden_dim, oup, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(oup)
            )
        else:
            self.conv = nn.Sequential(
                # Pointwise convolution
                nn.Conv2d(inp, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # Depthwise convolution
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # Pointwise-linear
                nn.Conv2d(hidden_dim, oup, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(oup)
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)



class MobileNetV2(nn.Module):
    """MobileNetV2 モデル"""

    def __init__(self, conv_width=1.0):
        super(MobileNetV2, self).__init__()
        print("Building MobileNetV2")

        block = IRB
        in_channels = 32
        last_channels = 1280
        Inverted_Residual_Setting = [
            # t, c, n, s
            [1, 16, 1, 1],  # 1
            [6, 24, 2, 2],  # 2-3
            [6, 32, 3, 2],  # 4-6
            [6, 64, 4, 2],  # 7-10
        ]

        # 1層目
        input_channel = int(in_channels * conv_width)
        self.last_channel = int(last_channels * conv_width) if conv_width > 1.0 else last_channels
        self.features = [ConvBN(3, input_channel, 2)]
        
        # 反転残差ブロックの構築
        for t, c, n, s in Inverted_Residual_Setting:
            output_channel = int(c * conv_width)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        
        # 2層目以降
        self.features.append(Conv1x1BN(input_channel, self.last_channel))
        # nn.Sequentialを作成
        self.features = nn.Sequential(*self.features)


    def forward(self, x):
        layer_4_outputs = None
        layer_11_outputs = None

        for i, layer in enumerate(self.features):
            x = layer(x)
            if i == 3:
                layer_4_outputs = x
            elif i == 10:
                layer_11_outputs = x
        
        # 4層目と11層目の出力を保存
        layer_11_upsampled = nn.functional.interpolate(
            layer_11_outputs,
            size=layer_4_outputs.shape[2:],
            mode='bilinear',
            align_corners=False
        )

        # チャネル次元で結合
        concat_features = torch.cat([layer_4_outputs, layer_11_upsampled], dim=1)

        return concat_features



class OpenPose(nn.Module):
    """OpenPose モデル"""

    def __init__(self, conv_width=1.0):
        super(OpenPose, self).__init__()
        
        # VGG19バックボーン（前処理ステージ）
        self.model0 = MobileNetV2()

        min_depth = 8
        depth = lambda d: max(int(d * conv_width), min_depth)
        

        print("Building OpenPose2016")
        # Stage 1 - L1ブランチ（Part Affinity Fields）
        self.model1_1 = nn.Sequential(
            DSConv(depth(88), depth(128), 3, 1, 1),
            DSConv(depth(128), depth(128), 3, 1, 1),
            DSConv(depth(128), depth(128), 3, 1, 1),
            DSConv(depth(128), depth(512), 1, 1, 0),
            DSConv(depth(512), 30, 1, 1, 0, relu=False)
        )
        
        # Stage 1 - L2ブランチ（Part Confidence Maps）
        self.model1_2 = nn.Sequential(
            DSConv(depth(88), depth(128), 3, 1, 1),
            DSConv(depth(128), depth(128), 3, 1, 1),
            DSConv(depth(128), depth(128), 3, 1, 1),
            DSConv(depth(128), depth(512), 1, 1, 0),
            DSConv(depth(512), 15, 1, 1, 0, bias=False)
        )
        
        # Stage 2 - 6
        for stage in range(2, 7):
            # L1ブランチ（出力30チャンネル）
            setattr(self, f'model{stage}_1', nn.Sequential(
                DSConv(depth(88)+30+15, depth(128), 3, 1, 1),
                DSConv(depth(128), depth(128), 3, 1, 1),
                DSConv(depth(128), depth(128), 3, 1, 1),
                DSConv(depth(128), depth(128), 1, 1, 0),
                DSConv(depth(128), 30, 1, 1, 0, relu=False)
            ))
            
            # L2ブランチ（出力15チャンネル）
            setattr(self, f'model{stage}_2', nn.Sequential(
                DSConv(depth(88)+30+15, depth(128), 3, 1, 1),
                DSConv(depth(128), depth(128), 3, 1, 1),
                DSConv(depth(128), depth(128), 3, 1, 1),
                DSConv(depth(128), depth(128), 1, 1, 0),
                DSConv(depth(128), 15, 1, 1, 0, relu=False)
            ))

        self._initialize_weights_norm()


    def forward(self, x):
        """順伝播"""
        saved_for_loss = []
        
        # mobilenetバックボーン
        features = self.model0(x)

        # Stage 1
        out1_1 = self.model1_1(features)
        out1_2 = self.model1_2(features)
        saved_for_loss.extend([out1_1, out1_2])

        first_stage_outputs = torch.cat([out1_1, out1_2, features], 1)

        # Stage 2 - 6
        for stage in range(2, 7):
            model_l1 = getattr(self, f'model{stage}_1')
            model_l2 = getattr(self, f'model{stage}_2')

            out_l1 = model_l1(first_stage_outputs)
            out_l2 = model_l2(first_stage_outputs)
            saved_for_loss.extend([out_l1, out_l2])

            if stage < 6:
                first_stage_outputs = torch.cat([out_l1, out_l2, features], 1)

        return (saved_for_loss[-2], saved_for_loss[-1]), saved_for_loss
    


    def _initialize_weights_norm(self):
        """重みの初期化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0.0)

        # 各ブロック最後の層の重みを正規分布で初期化
        final_layers = [
            self.model1_1[-1], self.model1_2[-1],
            self.model2_1[-1], self.model2_2[-1],
            self.model3_1[-1], self.model3_2[-1],
            self.model4_1[-1], self.model4_2[-1],
            self.model5_1[-1], self.model5_2[-1],
            self.model6_1[-1], self.model6_2[-1]
        ]
        # 1x1Convだけ
        for layer in final_layers:
            init.normal_(layer.pointwise.weight, std=0.01)



def get_model(pretrained_path=None, imagenet_pretrained=False):
    """ 呼び出される処理 """
    model = OpenPose()

    if pretrained_path:
        print(f'>>> Loading pretrained model from "{pretrained_path}" <<<')
        model.load_state_dict(torch.load(pretrained_path))

    elif imagenet_pretrained:
        print('>>> Loading imagenet pretrained model <<<')
        pretrained = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V2)
        backbone_state_dict = model.backbone.state_dict()
        pretrained_dict = {k: v for k, v in pretrained.state_dict().items() if k in backbone_state_dict}
        backbone_state_dict.update(pretrained_dict)
        model.backbone.load_state_dict(backbone_state_dict)
    
    return model


if __name__ == "__main__":
    import os
    import thop
    from torchinfo import summary
    from torchviz import make_dot
    from torch.utils.tensorboard import SummaryWriter

    model = get_model()
    print(model)
    
    # テスト用のダミー入力
    dummy_input = torch.randn(1, 3, 224, 224)

    # モデルのパラメータを確認
    flops, params = thop.profile(model, inputs=(dummy_input, ))
    gflops = flops / 1e9
    print(f"FLOPs: {flops} ({gflops:.2f} GFLOPs)")
    print(f"Parameters: {params}")

    # 計算グラフを出力
    MODELNAME = "mobilenet_v2"
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

    # tensorboard --logdir=../../experiments/img_network/mobilenet_v2/tbX/  でネットワークを可視化

    summary(model, input_size=(1, 3, 224, 224), depth=4)