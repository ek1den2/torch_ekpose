import torch
import torch.nn as nn
from torch.nn import init

# conv + bn + relu
class ConvBN(nn.Module):
    def __init__(self, nin, nout, kernel_size=3, stride=1, padding=1, bias=False, relu=True):
        super(ConvBN, self).__init__()
        self.conv = nn.Conv2d(nin, nout, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(nout)
        self.activation = nn.ReLU(inplace=True) if relu else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

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


class MobileNet(nn.Module):
    """MobileNet モデル"""
    
    def __init__(self, conv_width=1.0):
        super(MobileNet, self).__init__()
        print("Building MobileNet")

        self.conv_width = conv_width

        min_depth = 8
        depth = lambda d: max(int(d * self.conv_width), min_depth)

        # MobileNetバックボーン（前処理ステージ）
        self.model0 = nn.ModuleList([
            ConvBN(3, depth(32), 3, 2, 1),        # index 0
            DSConv(depth(32), depth(64), 3, 1, 1),       # index 1
            DSConv(depth(64), depth(128), 3, 2, 1),      # index 2
            DSConv(depth(128), depth(128), 3, 1, 1),     # index 3
            DSConv(depth(128), depth(256), 3, 2, 1),     # index 4
            DSConv(depth(256), depth(256), 3, 1, 1),     # index 5
            DSConv(depth(256), depth(512), 3, 1, 1),     # index 6
            DSConv(depth(512), depth(512), 3, 1, 1),     # index 7
            DSConv(depth(512), depth(512), 3, 1, 1),     # index 8
            DSConv(depth(512), depth(512), 3, 1, 1),     # index 9
            DSConv(depth(512), depth(512), 3, 1, 1),     # index 10
            DSConv(depth(512), depth(512), 3, 1, 1)      # index 11
        ])

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)


    def forward(self, x):
        hidden_outputs = {}
        for i, layer in enumerate(self.model0):
            x = layer(x)
            if i == 3:
                hidden_outputs['out0_3'] = x
            elif i == 7:
                hidden_outputs['out0_7'] = x
            elif i == 11:
                hidden_outputs['out0_11'] = x
            
        out_pool = self.maxpool(hidden_outputs["out0_3"])
        out1 = torch.cat([out_pool, hidden_outputs["out0_7"], hidden_outputs["out0_11"]], 1)

        return out1



class OpenPose(nn.Module):
    """OpenPose モデル"""
    
    def __init__(self, conv_width=1.0):
        super(OpenPose, self).__init__()
        
        # Mobilenetバックボーン
        self.model0 = MobileNet(conv_width=conv_width)

        min_depth = 8
        depth = lambda d: max(int(d * conv_width), min_depth)
        
        # Stage 1 - L1ブランチ（Part Affinity Fields）
        self.model1_1 = nn.Sequential(
            DSConv(depth(1152), depth(128), 3, 1, 1),
            DSConv(depth(128), depth(128), 3, 1, 1),
            DSConv(depth(128), depth(128), 3, 1, 1),
            DSConv(depth(128), depth(512), 1, 1, 0),
            DSConv(depth(512), 30, 1, 1, 0, relu=False)
        )
        
        # Stage 1 - L2ブランチ（Part Confidence Maps）
        self.model1_2 = nn.Sequential(
            DSConv(depth(1152), depth(128), 3, 1, 1),
            DSConv(depth(128), depth(128), 3, 1, 1),
            DSConv(depth(128), depth(128), 3, 1, 1),
            DSConv(depth(128), depth(512), 1, 1, 0),
            DSConv(depth(512), 15, 1, 1, 0, bias=False)
        )
        
        # Stage 2 - 6
        for stage in range(2, 7):
            # L1ブランチ（出力30チャンネル）
            setattr(self, f'model{stage}_1', nn.Sequential(
                DSConv(depth(1152)+30+15, depth(128), 3, 1, 1),
                DSConv(depth(128), depth(128), 3, 1, 1),
                DSConv(depth(128), depth(128), 3, 1, 1),
                DSConv(depth(128), depth(128), 1, 1, 0),
                DSConv(depth(128), 30, 1, 1, 0, relu=False)
            ))
            
            # L2ブランチ（出力15チャンネル）
            setattr(self, f'model{stage}_2', nn.Sequential(
                DSConv(depth(1152)+30+15, depth(128), 3, 1, 1),
                DSConv(depth(128), depth(128), 3, 1, 1),
                DSConv(depth(128), depth(128), 3, 1, 1),
                DSConv(depth(128), depth(128), 1, 1, 0),
                DSConv(depth(128), 15, 1, 1, 0, relu=False)
            ))
        
        self._initialize_weights_norm()


    def forward(self, x):
        """順伝播"""
        saved_for_loss = []
        
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



def get_model(pretrained=False, pretrained_path=None):
    """ 呼び出される処理 """
    model = OpenPose()

    if pretrained:
        print(f'>>> Loading pretrained model from "{pretrained_path}" <<<')
        model.load_state_dict(torch.load(pretrained_path))

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
    MODELNAME = "mobilenet"
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

    summary(model, input_size=(1, 3, 224, 224), depth=4)