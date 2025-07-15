import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models

# チャネルシャッフル
def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.size()
    channels_per_groupe = num_channels // groups

    # reshape
    x = x.view(batchsize, groups, channels_per_groupe, height, width)
    
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, num_channels, height, width)

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
    def __init__(self, inp, oup, stride):
        super(IRB, self).__init__()
      
        self.stride = stride
        branch_features = oup // 2

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                # Depthwise convolution
                nn.Conv2d(inp, inp, kernel_size=3, stride=self.stride, padding=1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                
                # Pointwise-linear
                nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True)
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            # Pointwise-linear
            nn.Conv2d(
                inp if (self.stride > 1) else branch_features,
                branch_features, kernel_size=1, stride=1, padding=0, bias=False
            ),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),

            # Depthwise convolution
            nn.Conv2d(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1, groups=branch_features, bias=False),
            nn.BatchNorm2d(branch_features),
            
            # Pointwise-linear
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        
        out = channel_shuffle(out, 2)

        return out


class ShuffleNetV2(nn.Module):
    """ShuffleNetV2 モデル"""

    def __init__(self, conv_width=1.0):
        super(ShuffleNetV2, self).__init__()
        print("Building ShuffleNetV2")

        # モデルの設定
        settings = {
            0.5: [24, 48, 96, 192, 1024], 
            1.0: [24, 116, 232, 464, 1024],
            1.5: [24, 176, 352, 704, 1024],
            2.0: [24, 244, 488, 976, 2048]
        }

        
        in_channels = 3
        out_channels = settings[conv_width][0]
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        in_channels = out_channels

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        stage_names = [f"stage{i}" for i in [2, 3, 4]]
        stage_repeats = [4, 8, 4]  # 各ステージの繰り返し回数

        for name, repeats, out_channels in zip(stage_names, stage_repeats, settings[conv_width][1:]):
            seq = [IRB(in_channels, out_channels, stride=2)]
            for i in range(repeats - 1):
                seq.append(IRB(out_channels, out_channels, stride=1))
            setattr(self, name, nn.Sequential(*seq))
            in_channels = out_channels
        
        out_channels = settings[conv_width][-1]
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


    def forward(self, x):
        out1 = self.conv1(x)
        out1p = self.maxpool(out1)
        out2 = self.stage2(out1p)
        out3 = self.stage3(out2)
        # out4 = self.stage4(out3)
        # out5 = self.conv5(out4)

        # out5_upsample = nn.functional.interpolate(out5, size=out2.shape[2:], mode='bilinear', align_corners=False)
        out3_upsample = nn.functional.interpolate(out3, size=out2.shape[2:], mode='bilinear', align_corners=False)
        outputs = torch.cat([out2, out3_upsample], dim=1)


        return outputs


class OpenPose(nn.Module):
    """OpenPose モデル"""

    def __init__(self, conv_width=1.0, conv_width2=1.0):
        super(OpenPose, self).__init__()
        
        # VGG19バックボーン（前処理ステージ）
        self.model0 = ShuffleNetV2(conv_width=conv_width)

        min_depth = 8
        depth = lambda d: max(round(d * conv_width), min_depth)
        depth2 = lambda d: max(round(d * conv_width2), min_depth)


        print("Building OpenPose2016")
        # Stage 1 - L1ブランチ（Part Affinity Fields）
        self.model1_1 = nn.Sequential(
            DSConv(depth(348), depth2(128), 3, 1, 1),
            DSConv(depth2(128), depth2(128), 3, 1, 1),
            DSConv(depth2(128), depth2(128), 3, 1, 1),
            DSConv(depth2(128), depth2(512), 1, 1, 0),
            DSConv(depth2(512), 30, 1, 1, 0, relu=False)
        )
        
        # Stage 1 - L2ブランチ（Part Confidence Maps）
        self.model1_2 = nn.Sequential(
            DSConv(depth(348), depth2(128), 3, 1, 1),
            DSConv(depth2(128), depth2(128), 3, 1, 1),
            DSConv(depth2(128), depth2(128), 3, 1, 1),
            DSConv(depth2(128), depth2(512), 1, 1, 0),
            DSConv(depth2(512), 15, 1, 1, 0, relu=False)
        )
        
        # Stage 2 - 6
        for stage in range(2, 7):
            # L1ブランチ（出力30チャンネル）
            setattr(self, f'model{stage}_1', nn.Sequential(
                DSConv(depth(348)+30+15, depth2(128), 3, 1, 1),
                DSConv(depth2(128), depth2(128), 3, 1, 1),
                DSConv(depth2(128), depth2(128), 3, 1, 1),
                DSConv(depth2(128), depth2(128), 1, 1, 0),
                DSConv(depth2(128), 30, 1, 1, 0, relu=False)
            ))
            
            # L2ブランチ（出力15チャンネル）
            setattr(self, f'model{stage}_2', nn.Sequential(
                DSConv(depth(348)+30+15, depth2(128), 3, 1, 1),
                DSConv(depth2(128), depth2(128), 3, 1, 1),
                DSConv(depth2(128), depth2(128), 3, 1, 1),
                DSConv(depth2(128), depth2(128), 1, 1, 0),
                DSConv(depth2(128), 15, 1, 1, 0, relu=False)
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
    dummy_input = torch.randn(1, 3, 368, 368)

    # モデルのパラメータを確認
    flops, params = thop.profile(model, inputs=(dummy_input, ))
    gflops = flops / 1e9
    print(f"FLOPs: {flops} ({gflops:.2f} GFLOPs)")
    print(f"Parameters: {params}")

    # 計算グラフを出力
    MODELNAME = "shufflenet_v2_1_0"
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

    summary(model, input_size=(1, 3, 368, 368), depth=4)
