import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.nn import init


def make_stages(cfg_dict):
    """辞書からCPMステージを構築"""
    layers = []
    # 最後の要素を除外してループ
    for i in range(len(cfg_dict) - 1):
        one_ = cfg_dict[i]
        for k, v in one_.items():
            if 'pool' in k:
                layers += [nn.MaxPool2d(kernel_size=v[0], stride=v[1],
                                        padding=v[2])]
            else:
                conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1],
                                   kernel_size=v[2], stride=v[3],
                                   padding=v[4])
                layers += [conv2d, nn.ReLU(inplace=True)]
    
    # 最後のレイヤーはReLUを適用しない
    one_ = list(cfg_dict[-1].keys())
    k = one_[0]
    v = cfg_dict[-1][k]
    conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1],
                       kernel_size=v[2], stride=v[3], padding=v[4])
    layers += [conv2d]
    return nn.Sequential(*layers)


def make_vgg19_block(block):
    """辞書からVGG19ブロックを構築"""
    layers = []
    for i in range(len(block)):
        one_ = block[i]
        for k, v in one_.items():
            if 'pool' in k:
                layers += [nn.MaxPool2d(kernel_size=v[0], stride=v[1],
                                        padding=v[2])]
            else:
                conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1],
                                   kernel_size=v[2], stride=v[3],
                                   padding=v[4])
                layers += [conv2d, nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)


def _get_block_configs():
    """ブロック設定を返すヘルパー関数"""
    blocks = {}
    
    # block0 is the preprocessing stage
    block0 = [{'conv1_1': [3, 64, 3, 1, 1]},
              {'conv1_2': [64, 64, 3, 1, 1]},
              {'pool1_stage1': [2, 2, 0]},
              {'conv2_1': [64, 128, 3, 1, 1]},
              {'conv2_2': [128, 128, 3, 1, 1]},
              {'pool2_stage1': [2, 2, 0]},
              {'conv3_1': [128, 256, 3, 1, 1]},
              {'conv3_2': [256, 256, 3, 1, 1]},
              {'conv3_3': [256, 256, 3, 1, 1]},
              {'conv3_4': [256, 256, 3, 1, 1]},
              {'pool3_stage1': [2, 2, 0]},
              {'conv4_1': [256, 512, 3, 1, 1]},
              {'conv4_2': [512, 512, 3, 1, 1]},
              {'conv4_3_CPM': [512, 256, 3, 1, 1]},
              {'conv4_4_CPM': [256, 128, 3, 1, 1]}]

    # Stage 1
    blocks['block1_1'] = [{'conv5_1_CPM_L1': [128, 128, 3, 1, 1]},
                          {'conv5_2_CPM_L1': [128, 128, 3, 1, 1]},
                          {'conv5_3_CPM_L1': [128, 128, 3, 1, 1]},
                          {'conv5_4_CPM_L1': [128, 512, 1, 1, 0]},
                          {'conv5_5_CPM_L1': [512, 26, 1, 1, 0]}]

    blocks['block1_2'] = [{'conv5_1_CPM_L2': [128, 128, 3, 1, 1]},
                          {'conv5_2_CPM_L2': [128, 128, 3, 1, 1]},
                          {'conv5_3_CPM_L2': [128, 128, 3, 1, 1]},
                          {'conv5_4_CPM_L2': [128, 512, 1, 1, 0]},
                          {'conv5_5_CPM_L2': [512, 15, 1, 1, 0]}]

    # Stages 2 - 6
    for i in range(2, 7):
        blocks['block%d_1' % i] = [
            {'Mconv1_stage%d_L1' % i: [169, 128, 7, 1, 3]},
            {'Mconv2_stage%d_L1' % i: [128, 128, 7, 1, 3]},
            {'Mconv3_stage%d_L1' % i: [128, 128, 7, 1, 3]},
            {'Mconv4_stage%d_L1' % i: [128, 128, 7, 1, 3]},
            {'Mconv5_stage%d_L1' % i: [128, 128, 7, 1, 3]},
            {'Mconv6_stage%d_L1' % i: [128, 128, 1, 1, 0]},
            {'Mconv7_stage%d_L1' % i: [128, 26, 1, 1, 0]}
        ]

        blocks['block%d_2' % i] = [
            {'Mconv1_stage%d_L2' % i: [169, 128, 7, 1, 3]},
            {'Mconv2_stage%d_L2' % i: [128, 128, 7, 1, 3]},
            {'Mconv3_stage%d_L2' % i: [128, 128, 7, 1, 3]},
            {'Mconv4_stage%d_L2' % i: [128, 128, 7, 1, 3]},
            {'Mconv5_stage%d_L2' % i: [128, 128, 7, 1, 3]},
            {'Mconv6_stage%d_L2' % i: [128, 128, 1, 1, 0]},
            {'Mconv7_stage%d_L2' % i: [128, 15, 1, 1, 0]}
        ]

    return block0, blocks


class OpenPose(nn.Module):
    """CPM (Convolutional Pose Machine) モデル"""
    
    def __init__(self, pretrained=True, pretrained_path=None):
        super(OpenPose, self).__init__()
        
        # ブロック設定を取得
        block0, blocks = _get_block_configs()
        
        # VGG19バックボーン
        print("Building VGG19")
        self.model0 = make_vgg19_block(block0)
        
        # CPMステージ（L1ブランチ: Part Affinity Fields）
        self.model1_1 = make_stages(list(blocks['block1_1']))
        self.model2_1 = make_stages(list(blocks['block2_1']))
        self.model3_1 = make_stages(list(blocks['block3_1']))
        self.model4_1 = make_stages(list(blocks['block4_1']))
        self.model5_1 = make_stages(list(blocks['block5_1']))
        self.model6_1 = make_stages(list(blocks['block6_1']))
        
        # CPMステージ（L2ブランチ: Part Confidence Maps）
        self.model1_2 = make_stages(list(blocks['block1_2']))
        self.model2_2 = make_stages(list(blocks['block2_2']))
        self.model3_2 = make_stages(list(blocks['block3_2']))
        self.model4_2 = make_stages(list(blocks['block4_2']))
        self.model5_2 = make_stages(list(blocks['block5_2']))
        self.model6_2 = make_stages(list(blocks['block6_2']))
        
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
    print(f"FLOPs: {flops}")
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