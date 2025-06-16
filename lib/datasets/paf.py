import numpy as np

"""Part Affinity Fields (vecmaps) の実装
centerA, centerB: それぞれ元画像上の２つの関節の座標
accumulate_vec_map: 累積されたベクトル場（vecmap's'と呼ぶべきか？）
count: 各ピクセルに何本のvecmapベクトルが加算されているか（平均用）
grid_y, grid_x: 特徴マップのサイズ
stride: 特徴マップのダウンサンプリング率
"""

def putVecMaps(centerA, centerB, accumulate_vec_map, count, grid_y, grid_x, stride):
    centerA = centerA.astype(float)
    centerB = centerB.astype(float)

    thre = 1   # vecmapを描画する幅のしきい値（threshold）

    # キーポイントの座標を特徴マップサイズに変換（stride=ダウンサンプリング回数）
    centerB = centerB / stride
    centerA = centerA / stride

    # 四肢X2からX1のユークリッド距離を計算
    limb_vec = centerB - centerA    # A -> Bへのベクトル
    norm = np.linalg.norm(limb_vec) # ベクトルの大きさ
    if (norm == 0.0):
        return accumulate_vec_map, count
    limb_vec_unit = limb_vec / norm     # 単位ベクトルを算出

    # x, y座標の範囲を制限（0~gridの範囲）（論文式(9)のpがlimb上にあるかの判別式の左側に相当）
    min_x = max(int(round(min(centerA[0], centerB[0]) - thre)), 0)
    max_x = min(int(round(max(centerA[0], centerB[0]) + thre)), grid_x)
    min_y = max(int(round(min(centerA[1], centerB[1]) - thre)), 0)
    max_y = min(int(round(max(centerA[1], centerB[1]) + thre)), grid_y)

    # ベクトル場用のグリッドを生成
    range_x = list(range(int(min_x), int(max_x), 1))
    range_y = list(range(int(min_y), int(max_y), 1))
    xx, yy = np.meshgrid(range_x, range_y)

    # 論文式(9)のpがlimb上にあるかの判別式の右側に相当
    ba_x = xx - centerA[0]  # 各pxの x - centerA の x
    ba_y = yy - centerA[1]  # 各pxの y - centerA の y
    limb_width = np.abs(ba_x * limb_vec_unit[1] - ba_y * limb_vec_unit[0])
    mask = limb_width < thre  # 四肢の幅

    # PAFを計算
    vec_map = np.copy(accumulate_vec_map) * 0.0     # ベクトル場の初期化
    vec_map[yy, xx] = np.repeat(mask[:, :, np.newaxis], 2, axis=2)  # pafのベクトルをmaskで制限
    vec_map[yy, xx] *= limb_vec_unit[np.newaxis, np.newaxis, :]     # 単位ベクトルを掛ける

    # x, yのどちらかが0でない場所をマスク
    mask = np.logical_or.reduce((np.abs(vec_map[:, :, 0]) > 0, np.abs(vec_map[:, :, 1]) > 0))

    accumulate_vec_map = np.multiply(accumulate_vec_map, count[:, :, np.newaxis])
    accumulate_vec_map += vec_map   # vecmapを累積ベクトル場に加算
    count[mask == True] += 1        # 各ピクセルにベクトルが加算された回数をカウント

    # ベクトル場を平均化
    mask = count == 0   # countが0のときmaskをTrueにする
    count[mask == True] = 1 # countが0のとき0で割るエラーを防ぐために1にしておく
    accumulate_vec_map = np.divide(accumulate_vec_map, count[:, :, np.newaxis]) # 累積ベクトル場をcountで割る
    count[mask == True] = 0 # もとに戻す

    return accumulate_vec_map, count
