import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models


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

        min_depth = 8
        depth = lambda d: max(round(d * conv_width), min_depth)
        
        # 1層
        self.features = ConvBN(3, depth(32), stride=2, padding=1, bias=False) # 0

        # 2-18層
        self.irblock1 = IRB(depth(32), depth(16), stride=1, expand_ratio=1)    # 1  n = 1
        self.irblock2 = IRB(depth(16), depth(24), stride=2, expand_ratio=6)    # 2  n = 2
        self.irblock3 = IRB(depth(24), depth(24), stride=1, expand_ratio=6)    # 3
        self.irblock4 = IRB(depth(24), depth(32), stride=2, expand_ratio=6)    # 4  n = 3
        self.irblock5 = IRB(depth(32), depth(32), stride=1, expand_ratio=6)    # 5
        self.irblock6 = IRB(depth(32), depth(32), stride=1, expand_ratio=6)    # 6
        self.irblock7 = IRB(depth(32), depth(64), stride=2, expand_ratio=6)    # 7  n = 4
        self.irblock8 = IRB(depth(64), depth(64), stride=1, expand_ratio=6)    # 8
        self.irblock9 = IRB(depth(64), depth(64), stride=1, expand_ratio=6)    # 9
        self.irblock10 = IRB(depth(64), depth(64), stride=1, expand_ratio=6)   # 10
        self.irblock11 = IRB(depth(64), depth(96), stride=1, expand_ratio=6)   # 11  n = 3
        self.irblock12 = IRB(depth(96), depth(96), stride=1, expand_ratio=6)   # 12
        self.irblock13 = IRB(depth(96), depth(96), stride=1, expand_ratio=6)   # 13
        self.irblock14 = IRB(depth(96), depth(160), stride=2, expand_ratio=6)  # 14  n = 3
        self.irblock15 = IRB(depth(160), depth(160), stride=1, expand_ratio=6) # 15
        self.irblock16 = IRB(depth(160), depth(160), stride=1, expand_ratio=6) # 16
        self.irblock17 = IRB(depth(160), depth(320), stride=1, expand_ratio=6) # 17  n = 1

        self.avgpool = nn.AdaptiveAvgPool2d(7)  # 7x7の適応平均プーリング
        # 最終層
        self.last_layer = (Conv1x1BN(depth(320), 1280, bias=False))            # 18



    def forward(self, x):
        out0 = self.features(x)  # 1層目
        out1 = self.irblock1(out0)  # 2層目
        out2 = self.irblock2(out1)  # 3層目
        out3 = self.irblock3(out2)  # 4層目
        out4 = self.irblock4(out3)  # 5層目
        out5 = self.irblock5(out4)  # 6層目
        out6 = self.irblock6(out5)  # 7層目
        out7 = self.irblock7(out6)  # 8層目
        out8 = self.irblock8(out7)  # 9層目
        out9 = self.irblock9(out8)  # 10層目
        out10 = self.irblock10(out9)  # 11層目
        out11 = self.irblock11(out10)  # 12層目
        out12 = self.irblock12(out11)  # 13層目
        out13 = self.irblock13(out12)  # 14層目

        # 7層目とアップサンプリングした14層目を結合
        out13_upsample = nn.functional.interpolate(out13, size=out6.shape[2:], mode='bilinear', align_corners=False)
        outputs = torch.cat([out6, out13_upsample], dim=1)

        return outputs

class OpenPose(nn.Module):
    """OpenPose モデル"""

    def __init__(self, conv_width=1.4, conv_width2=1.0):
        super(OpenPose, self).__init__()
        
        # VGG19バックボーン（前処理ステージ）
        self.model0 = MobileNetV2(conv_width=conv_width)

        min_depth = 8
        depth = lambda d: max(round(d * conv_width), min_depth)
        depth2 = lambda d: max(round(d * conv_width2), min_depth)


        print("Building OpenPose2016")
        # Stage 1 - L1ブランチ（Part Affinity Fields）
        self.model1_1 = nn.Sequential(
            DSConv(depth(128), depth2(128), 3, 1, 1),
            DSConv(depth2(128), depth2(128), 3, 1, 1),
            DSConv(depth2(128), depth2(128), 3, 1, 1),
            DSConv(depth2(128), depth2(512), 1, 1, 0),
            DSConv(depth2(512), 30, 1, 1, 0, relu=False)
        )
        
        # Stage 1 - L2ブランチ（Part Confidence Maps）
        self.model1_2 = nn.Sequential(
            DSConv(depth(128), depth2(128), 3, 1, 1),
            DSConv(depth2(128), depth2(128), 3, 1, 1),
            DSConv(depth2(128), depth2(128), 3, 1, 1),
            DSConv(depth2(128), depth2(512), 1, 1, 0),
            DSConv(depth2(512), 15, 1, 1, 0, relu=False)
        )
        
        # Stage 2 - 6
        for stage in range(2, 7):
            # L1ブランチ（出力30チャンネル）
            setattr(self, f'model{stage}_1', nn.Sequential(
                DSConv(depth(128)+30+15, depth2(128), 3, 1, 1),
                DSConv(depth2(128), depth2(128), 3, 1, 1),
                DSConv(depth2(128), depth2(128), 3, 1, 1),
                DSConv(depth2(128), depth2(128), 1, 1, 0),
                DSConv(depth2(128), 30, 1, 1, 0, relu=False)
            ))
            
            # L2ブランチ（出力15チャンネル）
            setattr(self, f'model{stage}_2', nn.Sequential(
                DSConv(depth(128)+30+15, depth2(128), 3, 1, 1),
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



def load_model(pretrained_path=None, conv_width=1.0, conv_width2=1.0):
    """ 呼び出される処理 """
    model = OpenPose(conv_width=conv_width, conv_width2=conv_width2)

    if pretrained_path:
        print(f'>>> Loading pretrained model from "{pretrained_path}" <<<')
        model.load_state_dict(torch.load(pretrained_path))
    
    return model