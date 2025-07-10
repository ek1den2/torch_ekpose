import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.nn import init

class VGG19(nn.Module):
    """VGG19モデル"""
    
    def __init__(self, pretrained=True, pretrained_path=None):
        super(VGG19, self).__init__()
        
        # VGG19バックボーン（前処理ステージ）
        self.model0 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2, 0),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2, 0),
            nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2, 0),
            nn.Conv2d(256, 512, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, 1, 1), nn.ReLU(inplace=True)
        )
        
        # VGG19事前学習済み重みの読み込み
        if pretrained:
            if pretrained_path is not None:
                print(f'>>> load pretrained model from {pretrained_path} <<<')
                self.load_state_dict(torch.load(pretrained_path))
            else:
                print('>>> load imagenet pretrained model <<<')
                self._load_vgg_pretrained()
    
    def forward(self, x):
        out1 = self.model0(x)
        return out1

class OpenPose(nn.Module):
    """CPM (Convolutional Pose Machine) モデル"""
    
    def __init__(self, pretrained=True, pretrained_path=None):
        super(OpenPose, self).__init__()
        
        # VGG19バックボーン
        print("Building VGG19")
        self.model0 = VGG19(pretrained=pretrained, pretrained_path=pretrained_path)
        # Stage 1 - L1ブランチ（Part Affinity Fields）
        self.model1_1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 512, 1, 1, 0), nn.ReLU(inplace=True),
            nn.Conv2d(512, 30, 1, 1, 0)
        )
        
        # Stage 1 - L2ブランチ（Part Confidence Maps）
        self.model1_2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 512, 1, 1, 0), nn.ReLU(inplace=True),
            nn.Conv2d(512, 15, 1, 1, 0)
        )

        # Stage 2 - 6
        for stage in range(2, 7):
            # L1ブランチ（出力30チャンネル）
            setattr(self, f'model{stage}_1', nn.Sequential(
                nn.Conv2d(173, 128, 7, 1, 3), nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 7, 1, 3), nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 7, 1, 3), nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 7, 1, 3), nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 7, 1, 3), nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 1, 1, 0), nn.ReLU(inplace=True),
                nn.Conv2d(128, 30, 1, 1, 0)
            ))
            
            # L2ブランチ（出力15チャンネル）
            setattr(self, f'model{stage}_2', nn.Sequential(
                nn.Conv2d(173, 128, 7, 1, 3), nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 7, 1, 3), nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 7, 1, 3), nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 7, 1, 3), nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 7, 1, 3), nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 1, 1, 0), nn.ReLU(inplace=True),
                nn.Conv2d(128, 15, 1, 1, 0)
            ))


        self._initialize_weights_norm()
        
        # VGG19事前学習済み重みの読み込み
        if pretrained:
            if pretrained_path is not None:
                print(f'>>> load pretrained model from {pretrained_path} <<<')
                self.load_state_dict(torch.load(pretrained_path))
            else:
                print('>>> load imagenet pretrained model <<<')
                self._load_vgg_pretrained()

    def forward(self, x):
        """順伝播"""
        saved_for_loss = []
        
        # VGG19バックボーン
        out1 = self.model0(x)

        # Stage 1
        out1_1 = self.model1_1(out1)
        out1_2 = self.model1_2(out1)
        out2 = torch.cat([out1_1, out1_2, out1], 1)
        saved_for_loss.extend([out1_1, out1_2])

        # Stage 2
        out2_1 = self.model2_1(out2)
        out2_2 = self.model2_2(out2)
        out3 = torch.cat([out2_1, out2_2, out1], 1)
        saved_for_loss.extend([out2_1, out2_2])

        # Stage 3
        out3_1 = self.model3_1(out3)
        out3_2 = self.model3_2(out3)
        out4 = torch.cat([out3_1, out3_2, out1], 1)
        saved_for_loss.extend([out3_1, out3_2])

        # Stage 4
        out4_1 = self.model4_1(out4)
        out4_2 = self.model4_2(out4)
        out5 = torch.cat([out4_1, out4_2, out1], 1)
        saved_for_loss.extend([out4_1, out4_2])

        # Stage 5
        out5_1 = self.model5_1(out5)
        out5_2 = self.model5_2(out5)
        out6 = torch.cat([out5_1, out5_2, out1], 1)
        saved_for_loss.extend([out5_1, out5_2])

        # Stage 6 (最終出力)
        out6_1 = self.model6_1(out6)
        out6_2 = self.model6_2(out6)
        saved_for_loss.extend([out6_1, out6_2])

        return (out6_1, out6_2), saved_for_loss

    def _initialize_weights_norm(self):
        """重みの初期化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0.0)

        # 各ブロック最後の層の重みを正規分布で初期化
        final_layers = [
            self.model1_1[8], self.model1_2[8],
            self.model2_1[12], self.model2_2[12],
            self.model3_1[12], self.model3_2[12],
            self.model4_1[12], self.model4_2[12],
            self.model5_1[12], self.model5_2[12],
            self.model6_1[12], self.model6_2[12]
        ]
        
        for layer in final_layers:
            init.normal_(layer.weight, std=0.01)

    def _load_vgg_pretrained(self):
        """VGG19の事前学習済み重みを読み込み"""
        url = 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'
        vgg_state_dict = model_zoo.load_url(url)
        vgg_keys = list(vgg_state_dict.keys())

        # VGGの重みをmodel0（VGG19バックボーン）に適用
        weights_load = {}
        model_keys = list(self.state_dict().keys())
        
        # 最初の20個の重み（VGG19部分）を対応させる
        for i in range(min(20, len(vgg_keys))):
            if i < len(model_keys):
                weights_load[model_keys[i]] = vgg_state_dict[vgg_keys[i]]

        # 現在のモデル状態を更新
        current_state = self.state_dict()
        current_state.update(weights_load)
        self.load_state_dict(current_state)


def get_model(pretrained=False, pretrained_path=None):
    """CPMモデルを構築して返す"""
    return OpenPose(pretrained=pretrained)


if __name__ == "__main__":
    import os
    import thop
    from torchviz import make_dot
    from torch.utils.tensorboard import SummaryWriter

    model = get_model(pretrained=False)
    print(model)
    
    # テスト用のダミー入力
    dummy_input = torch.randn(1, 3, 224, 224)

    # モデルのパラメータを確認
    flops, params = thop.profile(model, inputs=(dummy_input, ))
    gflops = flops / 1e9
    print(f"FLOPs: {flops} ({gflops:.2f} GFLOPs)")
    print(f"Parameters: {params}")

    # 計算グラフを出力
    MODELNAME = "VGG19"
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

    # tensorboard --logdir=../../experiments/img_network/VGG19/tbX/  でネットワークを可視化