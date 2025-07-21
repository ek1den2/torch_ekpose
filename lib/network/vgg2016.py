import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models

class VGG19(nn.Module):
    """VGG19"""
    
    def __init__(self):
        super(VGG19, self).__init__()
        print("Building VGG19")
        
        # VGG19バックボーン（前処理ステージ）
        self.backbone = nn.Sequential(
            *list(models.vgg19().features.children())[:23],
            nn.Conv2d(512, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.ReLU(inplace=True)
            )   # 0-22まで

    def forward(self, x):
        x = self.backbone(x)
        return x


class OpenPose(nn.Module):
    """OpenPose2016"""
    
    def __init__(self):
        super(OpenPose, self).__init__()
        # VGG19バックボーン
        self.model0 = VGG19()

        print("Building OpenPose2016")
        # Stage 1 - L1ブランチ（Part Affinity Fields）
        self.model1_1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 512, 1, 1, 0), nn.ReLU(inplace=True),
            nn.Conv2d(512, 38, 1, 1, 0)
        )
        
        # Stage 1 - L2ブランチ（Part Confidence Maps）
        self.model1_2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 512, 1, 1, 0), nn.ReLU(inplace=True),
            nn.Conv2d(512, 19, 1, 1, 0)
        )

        # Stage 2 - 6
        for stage in range(2, 7):
            # L1ブランチ（出力38チャンネル）
            setattr(self, f'model{stage}_1', nn.Sequential(
                nn.Conv2d(128+38+19, 128, 7, 1, 3), nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 7, 1, 3), nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 7, 1, 3), nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 7, 1, 3), nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 7, 1, 3), nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 1, 1, 0), nn.ReLU(inplace=True),
                nn.Conv2d(128, 38, 1, 1, 0)
            ))
            
            # L2ブランチ（出力19チャンネル）
            setattr(self, f'model{stage}_2', nn.Sequential(
                nn.Conv2d(128+38+19, 128, 7, 1, 3), nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 7, 1, 3), nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 7, 1, 3), nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 7, 1, 3), nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 7, 1, 3), nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 1, 1, 0), nn.ReLU(inplace=True),
                nn.Conv2d(128, 19, 1, 1, 0)
            ))


    def forward(self, x):
        """順伝播"""
        saved_for_loss = []
        
        # VGG19バックボーン
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
        
        for layer in final_layers:
            init.normal_(layer.weight, std=0.01)


def load_model(pretrained_path=None, imagenet_pretrained=False):
    """ 呼び出される処理 """
    model = OpenPose()

    if pretrained_path:
        print(f'>>> Loading pretrained model from "{pretrained_path}" <<<')
        model.load_state_dict(torch.load(pretrained_path))

    elif imagenet_pretrained:
        print('>>> Loading imagenet pretrained model <<<')
        pretrained = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        backbone_state_dict = model.model0.backbone.state_dict()
        pretrained_dict = {k: v for k, v in pretrained.state_dict().items() if k in backbone_state_dict}
        backbone_state_dict.update(pretrained_dict)
        model.model0.backbone.load_state_dict(backbone_state_dict)

    return model