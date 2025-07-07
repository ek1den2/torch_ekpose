import cv2
import numpy as np
import time
from scipy.ndimage.filters import gaussian_filter, maximum_filter

from scipy.ndimage.morphology import generate_binary_structure
from lib.pafprocess import pafprocess

from lib.utils.common import Human, BodyPart, CustomPart, CustomColors, CustomPairsRender

# 各四肢（関節の接続）を見つけるためのヒートマップのインデックス。
# 例: limb_type=1 は 首 -> 左肩 なので、
# joint_to_limb_heatmap_relationship[1] は関節を探すための
# ヒートマップのインデックス（首=1, 左肩=5）を表す。

joint_to_limb_heatmap_relationship = [[1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 0]]

# 特定の四肢に対するPAFのx座標とy座標を含むPAFのインデックス。
# 例: limb_type=1 は 首 -> 左肩 なので、
# PAFneckLShoulder_x = paf_xy_coords_per_limb[1][0] および
# PAFneckLShoulder_y = paf_xy_coords_per_limb[1][1] となる。
paf_xy_coords_per_limb = np.arange(14).reshape(7, 2)
NUM_LIMBS = len(joint_to_limb_heatmap_relationship)


def find_peaks(param, img):
    """
    グレースケール画像が与えられたとき、指定された閾値（param['thre1']）
    より大きい値を持つ局所的最大値を見つける。
    :param img: ピークを見つけたい入力画像（2D配列）
    :return: 画像内で見つかった各ピークの[x, y]座標を含む2D np.array
    """

    peaks_binary = (maximum_filter(img, footprint=generate_binary_structure(
        2, 1)) == img) * (img > param)
    # 逆順（[::-1]）に注意：[[y x], [y x]...]の代わりに[[x y], [x y]...]を返す。
    return np.array(np.nonzero(peaks_binary)[::-1]).T


def compute_resized_coords(coords, resizeFactor):
    """
    ある入力配列（例：画像）内のセルのインデックス/座標が与えられたとき、
    その配列が resizeFactor 倍にリサイズされた場合の新しい座標を提供する。
    例：サイズ3x3の画像が6x6にリサイズされた場合（resizeFactor=2）、
    セル[1,2]の新しい座標を知りたい -> この関数は[2.5, 4.5]を返す。
    :param coords: ある入力配列内のセルの座標（インデックス）
    :param resizeFactor: リサイズ係数 = shape_dest / shape_source。
    例：resizeFactor=2は、出力先の配列が元の配列の2倍の大きさであることを意味する。
    :return: shape_dest = resizeFactor * shape_source のサイズの配列における座標。
    これは、shape_sourceのサイズの画像がshape_destにリサイズされた場合に、
    'coords'に最も近い点の配列インデックスを表す。
    """

    # 1) coordsに0.5を足してピクセルの中心座標を得る（例：インデックス[0,0]は位置[0.5,0.5]のピクセルを表す）。
    # 2) resizeFactorを掛けることで、それらの座標をshape_destに変換する。
    # 3) その数値は新しい配列におけるピクセル中心の位置を表すので、0.5を引いて
    #    配列のインデックスの座標を得る（ステップ1を元に戻す）。
    return (np.array(coords, dtype=float) + 0.5) * resizeFactor - 0.5


def NMS(heatmaps, upsampFactor=1., bool_refine_center=True, bool_gaussian_filt=False, config=None):
    """
    Non-Maxima Suppression（非極大値抑制）：一連のグレースケール画像からピーク（局所的最大値）を見つける。
    :param heatmaps: 局所的最大値を見つける対象のグレースケール画像のセット（3D np.array、
    次元は image_height x image_width x num_heatmaps）。
    :param upsampFactor: CPMのヒートマップ出力と入力画像サイズの比率。
    例：元の画像が480x640でヒートマップが30x40xNの場合、upsampFactor=16。
    :param bool_refine_center: 以下を示すフラグ：
     - False: 単純に低解像度で見つかったピークをupsampFactorでスケールアップして返す（グリッドにスナップされる影響を受ける）。
     - True: （推奨、非常に正確）各低解像度ピークの周りの小さなパッチをアップサンプルし、
     元の入力画像の解像度でピークの位置を微調整する。
    :param bool_gaussian_filt: 各ピークの位置を微調整する前に、アップサンプルされた各パッチに
    1Dガウシアンフィルタ（平滑化）を適用するかどうかを示すフラグ。
    :return: NUM_JOINTS x 4 の np.array。各行は関節タイプ（0=鼻, 1=首...）を表し、
    列は{x, y}位置、スコア（確率）、およびユニークID（カウンター）を示す。
    """
    # CARLOSによる変更：ヒートマップをheatmap_avgにアップサンプリングしてから
    # NMSを実行してピークを見つける代わりに、このステップは以下によって約25〜50倍高速化できる：
    # （RoG上で、ガウシアンフィルタありで9-10ms、なしで5-6ms vs 250-280ms）
    # 1. （低解像度の）CPMの出力解像度でNMSを実行する。
    # 1.1. scipy.ndimage.filters.maximum_filter を使用してピークを見つける。
    # 2. ピークが見つかったら、ピークを中心とする5x5のパッチを取得し、それをアップサンプルして、
    #    実際の最大値の位置を微調整する。
    #  '-> これはheatmap_avg上でピークを見つけたことと等価だが、
    #      完全な（例：480x640）画像ではなく5x5のパッチのみをアップサンプルしてスキャンするため、はるかに高速。

    joint_list_per_joint_type = []
    cnt_total_joints = 0

    # 見つかったすべてのピークに対して、win_sizeはピークから各方向に何ピクセル取るかを指定し、
    # アップサンプルされるパッチを取得する。例：win_size=1 -> パッチは3x3; win_size=2 -> 5x5
    # （BICUBIC補間を正確に行うためには、win_sizeは2以上である必要がある！）
    win_size = 2

    for joint in range(config.MODEL.NUM_KEYPOINTS):
        map_orig = heatmaps[:, :, joint]
        peak_coords = find_peaks(config.TEST.THRESH_HEATMAP, map_orig)
        peaks = np.zeros((len(peak_coords), 4))
        for i, peak in enumerate(peak_coords):
            if bool_refine_center:
                x_min, y_min = np.maximum(0, peak - win_size)
                x_max, y_max = np.minimum(
                    np.array(map_orig.T.shape) - 1, peak + win_size)

                # 各ピークの周りの小さなパッチを取得し、その小さな領域のみをアップサンプルする。
                patch = map_orig[y_min:y_max + 1, x_min:x_max + 1]
                map_upsamp = cv2.resize(
                    patch, None, fx=upsampFactor, fy=upsampFactor, interpolation=cv2.INTER_CUBIC)

                # ガウシアンフィルタリングはピークあたり平均0.8msかかる（そして関節ごとに
                # 複数のピークが存在する可能性がある）-> 現時点ではスキップする（十分に正確なため）。
                map_upsamp = gaussian_filter(
                    map_upsamp, sigma=3) if bool_gaussian_filt else map_upsamp

                # パッチ内の最大値の座標を取得する。
                location_of_max = np.unravel_index(
                    map_upsamp.argmax(), map_upsamp.shape)
                # peaksは[x,y]を示していることを思い出す -> [y,x]にするために逆にする必要がある。
                location_of_patch_center = compute_resized_coords(
                    peak[::-1] - [y_min, x_min], upsampFactor)
                # 実際の最大値があるパッチ中心からのオフセットを計算する。
                refined_center = (location_of_max - location_of_patch_center)
                peak_score = map_upsamp[location_of_max]
            else:
                refined_center = [0, 0]
                # ピーク座標は[y,x]ではなく[x,y]なので反転させる。
                peak_score = map_orig[tuple(peak[::-1])]
            peaks[i, :] = tuple(
                x for x in compute_resized_coords(peak_coords[i], upsampFactor) + refined_center[::-1]) + (
                              peak_score, cnt_total_joints)
            cnt_total_joints += 1
        joint_list_per_joint_type.append(peaks)

    return joint_list_per_joint_type


def find_connected_joints(paf_upsamp, joint_list_per_joint_type, num_intermed_pts=10, config=None):
    """
    四肢のタイプごと（例：前腕、すねなど）に、可能性のあるすべての関節ペア
    （例：すべての手首と肘の組み合わせ）を探し、PAFを評価して、
    どのペアが実際に体の四肢であるかを判断する。
    :param paf_upsamp: 元の入力画像解像度にアップサンプルされたPAF。
    :param joint_list_per_joint_type: NMS()の'return'ドキュメントを参照。
    :param num_intermed_pts: joint_srcとjoint_dstの間で取得する中間点の数を示す整数。
    これらの点でPAFが評価される。
    :return: NUM_LIMBS行のリスト。各limb_type（行）には、そのタイプで見つかった
    すべての四肢のリストが格納される（例：すべての右前腕）。
    各四肢（connected_limbs[limb_type]の各アイテム）には、5つのセルが格納される：
    # {joint_src_id, joint_dst_id}: 各関節に関連付けられたユニークな番号。
    # limb_score_penalizing_long_dist: 関節の接続の良さを示すスコア。
    四肢の長さが長すぎる場合はペナルティが課される。
    # {joint_src_index, joint_dst_index}: そのタイプの見つかったすべての関節の中での
    関節のインデックス（例：見つかった3番目の右肘）。
    """
    connected_limbs = []

    # paf_upsampに素早くアクセスするための補助配列
    limb_intermed_coords = np.empty((4, num_intermed_pts), dtype=np.intp)
    for limb_type in range(NUM_LIMBS):
        # limb_typeによって指定されるタイプAの見つかったすべての関節のリスト
        # （例：右前腕は右肘から始まる）
        joints_src = joint_list_per_joint_type[joint_to_limb_heatmap_relationship[limb_type][0]]
        # limb_typeによって指定されるタイプBの見つかったすべての関節のリスト
        # （例：右前腕は右手首で終わる）
        joints_dst = joint_list_per_joint_type[joint_to_limb_heatmap_relationship[limb_type][1]]
        # print(joint_to_limb_heatmap_relationship[limb_type][0])
        # print(joint_to_limb_heatmap_relationship[limb_type][1])
        # print(paf_xy_coords_per_limb[limb_type][0])
        # print(paf_xy_coords_per_limb[limb_type][1])
        if len(joints_src) == 0 or len(joints_dst) == 0:
            # このタイプの四肢は見つからなかった（例：右手首または右肘が見つからなかったため、
            # 右前腕は見つからなかった）。
            connected_limbs.append([])
        else:
            connection_candidates = []
            # この四肢のpafのx座標を含むpafインデックスを指定する。
            limb_intermed_coords[2, :] = paf_xy_coords_per_limb[limb_type][0]
            # そしてy座標のpafインデックス
            limb_intermed_coords[3, :] = paf_xy_coords_per_limb[limb_type][1]
            for i, joint_src in enumerate(joints_src):
                # すべての可能なjoints_src[i]-joints_dst[j]ペアを試して、
                # それが実現可能な四肢であるかを確認する。
                for j, joint_dst in enumerate(joints_dst):
                    # 両方の関節の位置を引いて、可能性のある四肢の方向を取得する。
                    limb_dir = joint_dst[:2] - joint_src[:2]
                    # 可能性のある四肢の距離/長さを計算する（limb_dirのノルム）。
                    limb_dist = np.sqrt(np.sum(limb_dir ** 2)) + 1e-8
                    limb_dir = limb_dir / limb_dist  # Normalize limb_dir to be a unit vector

                    # joint_srcのx座標からjoint_dstのx座標まで、num_intermed_pts個の点を線形に配置する。
                    limb_intermed_coords[1, :] = np.round(np.linspace(
                        joint_src[0], joint_dst[0], num=num_intermed_pts))
                    limb_intermed_coords[0, :] = np.round(np.linspace(
                        joint_src[1], joint_dst[1], num=num_intermed_pts))  # y座標も同様
                    intermed_paf = paf_upsamp[limb_intermed_coords[0, :],
                                              limb_intermed_coords[1, :], limb_intermed_coords[2:4, :]].T

                    score_intermed_pts = intermed_paf.dot(limb_dir)
                    score_penalizing_long_dist = score_intermed_pts.mean(
                    ) + min(0.5 * paf_upsamp.shape[0] / limb_dist - 1, 0)
                    # 条件1：中間点の少なくとも80%がthre2より高いスコアを持つ。
                    criterion1 = (np.count_nonzero(
                        score_intermed_pts > config.TEST.THRESH_PAF) > 0.8 * num_intermed_pts)
                    # 条件2：大きな四肢の距離（画像の高さの半分以上）に対してペナルティを課した
                    # 平均スコアが正である。
                    criterion2 = (score_penalizing_long_dist > 0)
                    if criterion1 and criterion2:
                        # 最後の値は、両関節のpaf(+limb_dist)スコアとヒートマップスコアを合わせたもの。
                        connection_candidates.append(
                            [i, j, score_penalizing_long_dist,
                             score_penalizing_long_dist + joint_src[2] + joint_dst[2]])

            # 接続候補をscore_penalizing_long_distに基づいてソートする。
            connection_candidates = sorted(
                connection_candidates, key=lambda x: x[2], reverse=True)
            connections = np.empty((0, 5))
            # 四肢の数は、ソースまたはデスティネーション関節の少ない方の数しか存在できない
            # （例：手首が5つあっても肘が2つしかない場合、前腕は2つしか存在できない）。
            max_connections = min(len(joints_src), len(joints_dst))
            # すべての潜在的な関節接続を（スコア順に）走査する。
            for potential_connection in connection_candidates:
                i, j, s = potential_connection[0:3]
                # joints_src[i]やjoints_dst[j]が他のjoints_dstやjoints_srcに
                # まだ接続されていないことを確認する。
                if i not in connections[:, 3] and j not in connections[:, 4]:
                    # [joint_src_id, joint_dst_id, limb_score_penalizing_long_dist, joint_src_index, joint_dst_index]
                    connections = np.vstack(
                        [connections, [joints_src[i][3], joints_dst[j][3], s, i, j]])
                    # すでにmax_connectionsの接続を確立している場合は終了する
                    # （各関節は複数の関節に接続できない）。
                    if len(connections) >= max_connections:
                        break
            connected_limbs.append(connections)

    return connected_limbs


def group_limbs_of_same_person(connected_limbs, joint_list, config):
    """
    同じ人物に属する四肢をまとめる。
    :param connected_limbs: find_connected_joints()の'return'ドキュメントを参照。
    :param joint_list: joint_list_per_joint の展開版 [NMS()の'return'ドキュメントを参照]。
    :return: サイズ num_people x (NUM_JOINTS+2) の2D np.array。見つかった各人物について：
    # 最初のNUM_JOINTS列には、その人物に関連付けられた関節のインデックス（joint_list内）
    が含まれる（i番目の関節が見つからなかった場合は-1）。
    # 最後から2番目の列：この人物に属する関節+四肢の総合スコア。
    # 最後の列：この人物について見つかった関節の総数。
    """
    person_to_joint_assoc = []

    for limb_type in range(NUM_LIMBS):
        joint_src_type, joint_dst_type = joint_to_limb_heatmap_relationship[limb_type]

        for limb_info in connected_limbs[limb_type]:
            person_assoc_idx = []
            for person, person_limbs in enumerate(person_to_joint_assoc):
                if person_limbs[joint_src_type] == limb_info[0] or person_limbs[joint_dst_type] == limb_info[1]:
                    person_assoc_idx.append(person)

            # 関節の1つが人物に関連付けられており、もう一方の関節が
            # 同じ人物に関連付けられているか、まだ誰にも関連付けられていない場合：
            if len(person_assoc_idx) == 1:
                person_limbs = person_to_joint_assoc[person_assoc_idx[0]]
                # もう一方の関節がまだ誰にも関連付けられていない場合、
                if person_limbs[joint_dst_type] != limb_info[1]:
                    # 現在の人物に関連付ける。
                    person_limbs[joint_dst_type] = limb_info[1]
                    # この人物に関連付けられた四肢の数を増やす。
                    person_limbs[-1] += 1
                    # そして合計スコアを更新する（+= joint_dstのヒートマップスコア
                    # + joint_srcとjoint_dstを接続するスコア）。
                    person_limbs[-2] += joint_list[limb_info[1]
                                                       .astype(int), 2] + limb_info[2]
            elif len(person_assoc_idx) == 2:  # 2つ見つかり、それらがばらばらの場合、マージする。
                person1_limbs = person_to_joint_assoc[person_assoc_idx[0]]
                person2_limbs = person_to_joint_assoc[person_assoc_idx[1]]
                membership = ((person1_limbs >= 0) & (person2_limbs >= 0))[:-2]
                if not membership.any():  # 両方の人物に同じ接続関節がない場合、1人の人物にマージする。
                    # どの関節が接続されているかを更新する。
                    person1_limbs[:-2] += (person2_limbs[:-2] + 1)
                    # 総合スコアと接続された関節の総数を、
                    # それらのカウンターを合計して更新する。
                    person1_limbs[-2:] += person2_limbs[-2:]
                    # 現在の関節接続のスコアを総合スコアに加算する。
                    person1_limbs[-2] += limb_info[2]
                    person_to_joint_assoc.pop(person_assoc_idx[1])
                else:  # 上記の len(person_assoc_idx)==1 と同じケース
                    person1_limbs[joint_dst_type] = limb_info[1]
                    person1_limbs[-1] += 1
                    person1_limbs[-2] += joint_list[limb_info[1]
                                                        .astype(int), 2] + limb_info[2]
            else:  # どの人物もこれらの関節を要求していない場合、新しい人物を作成する。
                # 人物情報をすべて-1（関節の関連付けなし）に初期化する。
                row = -1 * np.ones(config.MODEL.NUM_KEYPOINTS + 2)
                # 新しい接続の関節情報を保存する。
                row[joint_src_type] = limb_info[0]
                row[joint_dst_type] = limb_info[1]
                # この人物の接続された関節の総数：2
                row[-1] = 2
                # 総合スコアを計算：スコア joint_src + スコア joint_dst + スコア 接続
                # {joint_src,joint_dst}
                row[-2] = sum(joint_list[limb_info[:2].astype(int), 2]
                              ) + limb_info[2]
                person_to_joint_assoc.append(row)

    # 接続されている部位が非常に少ない人物を削除する。
    people_to_delete = []
    for person_id, person_info in enumerate(person_to_joint_assoc):
        if person_info[-1] < 3 or person_info[-2] / person_info[-1] < 0.2:
            people_to_delete.append(person_id)
    # リストを逆順に走査し、最後のインデックスから削除していく
    # （そうしないと、例えば0番目のアイテムを削除すると、
    # 削除されるべき残りの人物のインデックスが変更されてしまうため！）
    for index in people_to_delete[::-1]:
        person_to_joint_assoc.pop(index)

    # np.arrayにアイテムを追加するのはコストがかかる場合がある（新しいメモリの割り当て、配列のコピー、そして新しい行の追加）
    # 代わりに、人物のセットをリストとして扱い（アイテムの追加が速い）、
    # 最後にのみnp.arrayに変換する。
    return np.array(person_to_joint_assoc)


def paf_to_pose(heatmaps, pafs, config):
    # ボトムアップアプローチ：
    # ステップ1：画像内のすべての関節を見つける（関節タイプ別：[0]=鼻, [1]=首...）
    joint_list_per_joint_type = NMS(heatmaps, upsampFactor=config.MODEL.DOWNSAMPLE, config=config)
    # joint_listはjoint_list_per_jointの展開版で、
    # 5番目の列に関節タイプ（0=鼻, 1=首...）を示す列を追加する。
    joint_list = np.array([tuple(peak) + (joint_type,) for joint_type,
                                                           joint_peaks in enumerate(joint_list_per_joint_type) for peak in joint_peaks])

    # import ipdb
    # ipdb.set_trace()
    # ステップ2：どの関節がどの肘と組み合わさって四肢を形成するかを見つける。
    paf_upsamp = cv2.resize(
        pafs, None, fx=config.MODEL.DOWNSAMPLE, fy=config.MODEL.DOWNSAMPLE, interpolation=cv2.INTER_CUBIC)
    connected_limbs = find_connected_joints(paf_upsamp, joint_list_per_joint_type,
                                            config.TEST.NUM_INTERMED_PTS_BETWEEN_KEYPOINTS, config)

    # ステップ3：同じ人物に属する四肢を関連付ける。
    person_to_joint_assoc = group_limbs_of_same_person(
        connected_limbs, joint_list, config)

    return joint_list, person_to_joint_assoc


def paf_to_pose_cpp(heatmaps, pafs, config):
    humans = []
    joint_list_per_joint_type = NMS(heatmaps, upsampFactor=config.MODEL.DOWNSAMPLE, config=config)

    joint_list = np.array(
        [tuple(peak) + (joint_type,) for joint_type, joint_peaks in enumerate(joint_list_per_joint_type) for peak in
         joint_peaks]).astype(np.float32)

    if joint_list.shape[0] > 0:
        joint_list = np.expand_dims(joint_list, 0)
        paf_upsamp = cv2.resize(
            pafs, None, fx=config.MODEL.DOWNSAMPLE, fy=config.MODEL.DOWNSAMPLE, interpolation=cv2.INTER_NEAREST)
        heatmap_upsamp = cv2.resize(
            heatmaps, None, fx=config.MODEL.DOWNSAMPLE, fy=config.MODEL.DOWNSAMPLE, interpolation=cv2.INTER_NEAREST)
        pafprocess.process_paf(joint_list, heatmap_upsamp, paf_upsamp)
        for human_id in range(pafprocess.get_num_humans()):
            human = Human([])
            is_added = False
            for part_idx in range(config.MODEL.NUM_KEYPOINTS):
                c_idx = int(pafprocess.get_part_cid(human_id, part_idx))
                if c_idx < 0:
                    continue
                is_added = True
                human.body_parts[part_idx] = BodyPart(
                    '%d-%d' % (human_id, part_idx), part_idx,
                    float(pafprocess.get_part_x(c_idx)) / heatmap_upsamp.shape[1],
                    float(pafprocess.get_part_y(c_idx)) / heatmap_upsamp.shape[0],
                    pafprocess.get_part_score(c_idx)
                )
            if is_added:
                score = pafprocess.get_score(human_id)
                human.score = score
                humans.append(human)

    return humans
