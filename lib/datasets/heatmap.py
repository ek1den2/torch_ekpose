import numpy as np

"""Confidence Maps（heatmap）の実装
    center: 元画像上の関節の座標（x, y）
    accumulate_cofidence_map: 累積された信頼度マップ（heatmap's'と呼ぶべきか？）
    sigma: ガウス分布の標準偏差（広がりの大きさ）
    grid_x, grid_y: 特徴マップのサイズ（x, y）
    stride: 特徴マップのダウンサンプリング率
"""

def putGaussianMaps(center, accumulate_confid_map, sigma, grid_y, grid_x, stride):
    # 特徴マップ座標の中心点を計算
    start = stride / 2.0 - 0.5     

    # ヒートマップ用のグリッドを生成
    y_range = [i for i in range(int(grid_y))]
    x_range = [i for i in range(int(grid_x))]
    xx, yy = np.meshgrid(x_range, y_range)
    
	# 特徴マップを元画像の座標に変換
    xx = xx * stride + start
    yy = yy * stride + start
    
	# cmapの計算
    d2 = (xx - center[0]) ** 2 + (yy - center[1]) ** 2	# ある点pからキーポイントxとの距離の2乗
    exponent = d2 / 2.0 / sigma / sigma		# 論文式(7)expの中身
    mask = exponent <= 4.6052	# e^(-4.6052)≈0.01=1%以下の場所を無視
    cofid_map = np.exp(-exponent)	# 論文式(7)に基づくheatmapの計算
    cofid_map = np.multiply(mask, cofid_map)
    accumulate_confid_map += cofid_map		# 累積信頼度マップに加算
    accumulate_confid_map[accumulate_confid_map > 1.0] = 1.0	# 1.0を超える値は1.0に制限
    
    return accumulate_confid_map