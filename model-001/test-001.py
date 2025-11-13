import pandas as pd
import numpy as np
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import datetime
import xgboost
import xgboost as xgb
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from sklearn.preprocessing import LabelEncoder

kaggle = False
innter_test = True
enable_save_train_data = True

# 
INTERVAL = 0.25

row_threshold = 10000

# Variables for environment (環境) -------------------------
if kaggle == True:
    INPUT_TRAIN_FILE = "train.csv"
    INPUT_TEST_FILE = "test.csv"
    TRAIN_ANNOTATION_DIR   = "train_annotation/"
    TRAIN_TRACKING_DIR     = "train_tracking/"
    TEST_TRACKING_DIR      = "test_tracking/"
    OUTPUT_FILE            = "submission.csv"
else:
    if innter_test == True:
        INPUT_TRAIN_FILE = "train.csv"
    else:
        INPUT_TRAIN_FILE = "../train.csv"
    INPUT_TEST_FILE = "../test.csv"
    TRAIN_ANNOTATION_DIR   = "../train_annotation/"
    TRAIN_TRACKING_DIR     = "../train_tracking/"
    TEST_TRACKING_DIR      = "../test_tracking/"
    OUTPUT_FILE            = "submission.csv"

# Hyper Parameter ------------------------------
xgb_param_space={
    'n_estimators': Integer(100, 1000), #森の中の木の数
    'learning_rate': Real(0.001, 0.5),  #学習率
    'max_depth': Integer(1, 15),        #各決定木の最大深さ
    'min_child_weight': Integer(1,20),  #子ノードに必要な最小サンプル重みの合
    'gamma':    Real(0.0, 1.0),         #ノードの分割に必要な最小損失減少
    'subsample': Real(0.5, 1.0),        #各決定木を構築するために使用されるサンプルの割合
    'colsample_bytree': Real(0.0001,1.0),   #各決定木を構築するために使用される特徴量の割合
    'colsample_bylevel': Real(0.0001,1.0),   #各レベルの決定木を構築するために使用される特徴量の割合
    'colsample_bynode': Real(0.0001, 1.0), #各ノードの分割に使用される特徴量の割合
    'reg_alpha': Real(0.01, 1.0),         #L1正則化項の重み
    'reg_lambda': Real(0.01, 1.0),           #L2正則化項の重み
#    'scale_pos_weight': Real(0.1, 4.0),      #正例と負例の不均衡を補正するための重み
#    'base_score': Real(0.1, 1.0),           #すべての観測値に対する初期予測確率
}

# todo: action_remapを自動的に生成するようにする (2025-10-14)
action_remap = {
    0: 'approach',
    1: 'attack',
    2: 'avoid',
    3: 'chase',
    4: 'chaseattack',
    5: 'rear',
    6: 'submit'
}

# todo: solo_actionを自動的に生成するようにする (2025-10-14)
solo_action = [
    'rear'
]


# return tracking_method in test
# testのtracking_methodを返す
def get_test_tracking_method():
    test_data = pd.read_csv(INPUT_TEST_FILE)
    test = test_data.iloc[0]
    return test['tracking_method']

# resuce row data
# trackingデータを間引く
# src_data : original data
# start_time : start time for reducing
# end_time   : end time for reducing
# interval   : inval time for reducting (sec)
def reduce_rows(src_data, start_time, end_time, interval):
    target_times = np.arange(start_time, end_time + 1e-8, interval)  # end を含めたい場合

    # necessary for sort by key if using merge_asof
    # merge_asof を使う場合はキーでソートが必要
    sorted = src_data.sort_values('video_time').reset_index(drop=True)
    target_df = pd.DataFrame({'video_time': target_times})

    # merge 'nearest' line by direction='nearest
    # direction='nearest' で「最も近い」行を結合
    result = pd.merge_asof(target_df, sorted, on='video_time', direction='nearest')
    return result

# initialize DataFrame for training data
def init_bodypart_dataframe():
    retval = pd.DataFrame({
        'lab_id'      : [],
        'video_id'    : [],
        'video_time'  : [],
        'video_frame' : [],
        'mouse_id'    : [],
        'bodypart'    : [],
        'x'           : [],
        'y'           : [],
        'x_cm'        : [],
        'y_cm'        : [],
        'vx_cm'       : [],
        'vy_cm'       : [],
    })
    return retval

# returns bodypart and action in src_data
# src_dataに記述されているbodypartとactionを返す
def find_bodypart_action(src_data):
    bodyparts = []
    actions   = []
    for i in range(len(src_data)):
        train = src_data.iloc[i]
        body_parts_tracked = json.loads(train['body_parts_tracked'].replace('""', '"'))
        for j in range(len(body_parts_tracked)):
            body_parts = body_parts_tracked[j]
            bodyparts.append(body_parts)
        if pd.isna(train['behaviors_labeled']) == False:
            behaviors_labeled  = json.loads(train['behaviors_labeled'].replace('""', '"'))
            for i in range(len(behaviors_labeled)):
                act = behaviors_labeled[i].split(',')
                act[2] = act[2].replace("'", "")
                actions.append(act[2])
    return bodyparts, actions

def wrap_up_body_center(src_data, anno_data, action, agent):
    tmp_center = src_data[src_data['bodypart']=='body_center']
    tmp_nose   = src_data[src_data['bodypart']=='nose']
    if (len(tmp_center) == 0) | (len(tmp_nose) == 0):
        return pd.DataFrame()
    if len(anno_data) == 0:
        # for test
        retval = pd.DataFrame({
            'lab_id'       : [tmp_center['lab_id'].iloc[0]]      * 1,
            'video_id'     : [tmp_center['video_id'].iloc[0]]    * 1,
            'video_frame'  : [tmp_center['video_frame'].iloc[0]] * 1,
            'video_time'   : [tmp_center['video_time'].iloc[0]]  * 1,
            'mouse_id'     : [tmp_center['mouse_id'].iloc[0]]    * 1,
            'agent_target' :  [np.nan]  * 1,
            'action'       :  [np.nan]  * 1,
            'center_x_cm'  : [tmp_center['x_cm'].iloc[0]]  * 1,
            'center_y_cm'  : [tmp_center['y_cm'].iloc[0]]  * 1,
            'center_vx_cm' : [tmp_center['vx_cm'].iloc[0]] * 1,
            'center_vy_cm' : [tmp_center['vy_cm'].iloc[0]] * 1,
            'nose_x_cm'    : [tmp_nose['x_cm'].iloc[0]]    * 1,
            'nose_y_cm'    : [tmp_nose['y_cm'].iloc[0]]    * 1,
            'nose_vx_cm'   : [tmp_nose['vx_cm'].iloc[0]]   * 1,
            'nose_vy_cm'   : [tmp_nose['vy_cm'].iloc[0]]   * 1,
        })
    else:
        # for train
        retval = pd.DataFrame({
            'lab_id'       : [tmp_center['lab_id'].iloc[0]]      * 1,
            'video_id'     : [tmp_center['video_id'].iloc[0]]    * 1,
            'video_frame'  : [tmp_center['video_frame'].iloc[0]] * 1,
            'video_time'   : [tmp_center['video_time'].iloc[0]]  * 1,
            'mouse_id'     : [tmp_center['mouse_id'].iloc[0]]    * 1,
            'agent_target' :  [agent]  * 1,
            'action'       :  [action] * 1,
            'center_x_cm'  : [tmp_center['x_cm'].iloc[0]]  * 1,
            'center_y_cm'  : [tmp_center['y_cm'].iloc[0]]  * 1,
            'center_vx_cm' : [tmp_center['vx_cm'].iloc[0]] * 1,
            'center_vy_cm' : [tmp_center['vy_cm'].iloc[0]] * 1,
            'nose_x_cm'    : [tmp_nose['x_cm'].iloc[0]]    * 1,
            'nose_y_cm'    : [tmp_nose['y_cm'].iloc[0]]    * 1,
            'nose_vx_cm'   : [tmp_nose['vx_cm'].iloc[0]]   * 1,
            'nose_vy_cm'   : [tmp_nose['vy_cm'].iloc[0]]   * 1,
        })
    if (len(tmp_center) > 0) & (len(tmp_nose) > 0):
        retval['rotate'] = np.arctan2(retval['center_y_cm']-retval['nose_y_cm'], retval['center_x_cm']-retval['nose_x_cm'])
    return retval

def init_bodypart_dataframeWithCenterNose(src_data, anno_data):
    frames = src_data['video_frame'].unique()
    mice   = src_data['mouse_id'].unique()
    retval = pd.DataFrame()

    for i in range(len(anno_data)):
        anno = anno_data.iloc[i]
        start_frame = anno['start_frame']
        stop_frame = anno['stop_frame']
        for mouse in mice:
            mouse_data = src_data[src_data['mouse_id']==mouse]
            for frm in frames:
                if (frm < start_frame) |(frm > stop_frame):
                    continue
                tmp = mouse_data[mouse_data['video_frame'] == frm]
                if len(tmp) > 0:
                    if (anno['agent_id'] != mouse) & (anno['target_id'] != mouse):
                        continue
                    if anno['agent_id'] == mouse:
                        agent = True
                    if anno['target_id'] == mouse:
                        agent = False
                    action = anno['action']
                    tmp2 = wrap_up_body_center(tmp, anno_data, action, agent)
                    if len(tmp2)>0:
                        retval = pd.concat([retval, tmp2], ignore_index=True)
    return retval

def init_bodypart_dataframeWithCenterNoseNonAnnotation(src_data):
    frames = src_data['video_frame'].unique()
    mice   = src_data['mouse_id'].unique()
    retval = pd.DataFrame()

    for mouse in mice:
        mouse_data = src_data[src_data['mouse_id']==mouse]
        for frm in frames:
            tmp = mouse_data[mouse_data['video_frame'] == frm]
            if len(tmp) > 0:
                tmp2 = wrap_up_body_center(tmp, pd.DataFrame(), "", True)
                if len(tmp2)>0:
                    retval = pd.concat([retval, tmp2], ignore_index=True)
    return retval

# トラッキングデータを取得する
def read_tracking(lab_id, video_id, tracking_dir, unnecessary_bodypart, unnecessary_action, fps, pps, train):
    tracking_file = f"{tracking_dir}{lab_id}/{video_id}.parquet"
    if not os.path.exists(tracking_file):
        return None

    tracking_data = pd.read_parquet(tracking_file)

    annotation_file = f"{TRAIN_ANNOTATION_DIR}{lab_id}/{video_id}.parquet"
    if os.path.exists(annotation_file):
        annotation_data = pd.read_parquet(annotation_file)
    else:
        if train == True:
            return None

        annotation_data = pd.DataFrame()
        if len(unnecessary_bodypart)>0:
            return None
    # 不要な行を削除する
    if len(unnecessary_bodypart) > 0:
        for bp in unnecessary_bodypart:
            tracking_data = tracking_data[tracking_data['bodypart']!=bp]
    if len(unnecessary_action) > 0:
        for act in unnecessary_action:
            annotation_data = annotation_data[annotation_data['action']!=act]
        if len(annotation_data) == 0:
            return None
    # 時間、座標を正規化する
    # frame⇨sec変換する
    tracking_data['video_time'] = tracking_data['video_frame'] / fps

    # px→cm変換する
    tracking_data['x_cm'] = tracking_data['x'] / pps
    tracking_data['y_cm'] = tracking_data['y'] / pps

    # trackingを間引く
    start_frame = tracking_data['video_frame'].min()
    end_frame   = tracking_data['video_frame'].max()
    bodyparts   = tracking_data['bodypart'].unique()
    mice        = tracking_data['mouse_id'].unique()

    tmp_all = pd.DataFrame()

    for mouse in mice:
        for bp in bodyparts:
            tmp = tracking_data[(tracking_data['mouse_id']==mouse) &
                                (tracking_data['bodypart']==bp)
                                ]
            result = reduce_rows(tmp, start_frame, end_frame, INTERVAL)
            # vx/vyを求める
            result['vx_cm'] = result['x_cm'].diff()
            result['vy_cm'] = result['y_cm'].diff()
            tmp = pd.DataFrame({
                'lab_id'      : [lab_id]   * len(result),
                'video_id'    : [video_id] * len(result),
                'video_time'  : result['video_time'],
                'video_frame' : result['video_frame'],
                'mouse_id'    : result['mouse_id'],
                'bodypart'    : result['bodypart'],
                'x'           : result['x'],
                'y'           : result['y'],
                'x_cm'        : result['x_cm'],
                'y_cm'        : result['y_cm'],
                'vx_cm'       : result['vx_cm'],
                'vy_cm'       : result['vy_cm']
            })
            tmp = tmp.dropna(subset=['video_frame'])
            tmp_all = pd.concat([tmp_all, tmp], ignore_index=True)

    if len(annotation_data)>0:
        output = init_bodypart_dataframeWithCenterNose(tmp_all, annotation_data)
    else:
        output = init_bodypart_dataframeWithCenterNoseNonAnnotation(tmp_all)

    if len(unnecessary_action)==0:
        return output

    # todo: srotateを追加する
    return output

# 
def get_tracking_data(org_data, tracking_dir, unnecessary_bodypart, unnecessary_action, train):
    global tracking_method

    tasks = []
    retval = pd.DataFrame()

    # 並列処理対象をリスト化
    args_list = []
    for i in range(len(org_data)):
        row = org_data.iloc[i]
        lab_id   = str(row['lab_id'])
        video_id = str(row['video_id'])
        trk_mthd = row['tracking_method']
        fps      = row['frames_per_second']
        pps      = row['pix_per_cm_approx']
        if tracking_method != trk_mthd:
            continue
        args_list.append((lab_id, video_id, tracking_dir, unnecessary_bodypart, unnecessary_action, fps, pps, train))

    # 並列実行
    max_workers = max(1, multiprocessing.cpu_count() - 1)  # CPUコア数-1
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(read_tracking, *args): args for args in args_list}

        for future in as_completed(futures):
            try:
                tracking = future.result()
                if tracking is not None and len(tracking) > 0:
                    retval = pd.concat([retval, tracking], ignore_index=True)
            except Exception as e:
                lab_id, video_id, *_ = futures[future]
                print(f"Error processing {lab_id}/{video_id}: {e}")

    return retval

def add_rotate(src_data, with_action):
    if with_action == True:
        src_data.sort_values(by=['video_id','mouse_id','action'],ascending=[True,True,True], inplace=True)
    else:
        src_data.sort_values(by=['video_id','mouse_id'],ascending=[True,True], inplace=True)
    src_data['delta_rotate'] = src_data['rotate'].diff()
    src_data['delta_rotate'] = (src_data['delta_rotate'] + np.pi) % (2 * np.pi) - np.pi
    return src_data

def add_agent_target_features(df):
    """
    agent_target（行動主体/対象）判定用に、マウス間の相対特徴量を作成する。
    各 video_id, video_frame ごとにマウスをペアリングして、相対的な特徴量を生成。
    """
    features = []

    # video_id, video_frame ごとにペアを作成
    for (vid, frame), group in df.groupby(['video_id', 'video_frame']):
        group = group.reset_index(drop=True)
        n = len(group)
        if n < 2:
            continue

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue

                m1 = group.iloc[i]  # 対象マウス1
                m2 = group.iloc[j]  # 比較相手マウス2

                # === 距離 ===
                dist = np.hypot(m1.center_x_cm - m2.center_x_cm,
                                m1.center_y_cm - m2.center_y_cm)

                # === 速度（中心） ===
                speed1 = np.hypot(m1.center_vx_cm, m1.center_vy_cm)
                speed2 = np.hypot(m2.center_vx_cm, m2.center_vy_cm)
                rel_speed = speed1 - speed2  # +ならm1のほうが速い

                # === 向き（nose方向の差） ===
                nose_vec1 = np.array([m1.nose_x_cm - m1.center_x_cm,
                                      m1.nose_y_cm - m1.center_y_cm])
                nose_vec2 = np.array([m2.nose_x_cm - m2.center_x_cm,
                                      m2.nose_y_cm - m2.center_y_cm])
                # コサイン類似度で方向差（1=同方向, -1=反対）
                cos_sim = np.dot(nose_vec1, nose_vec2) / (
                    np.linalg.norm(nose_vec1) * np.linalg.norm(nose_vec2) + 1e-8
                )

                # === 回転変化 ===
                rel_rotate = m1.delta_rotate - m2.delta_rotate

                # === noseの向きが相手方向を向いているか ===
                vec_to_other = np.array([m2.center_x_cm - m1.center_x_cm,
                                         m2.center_y_cm - m1.center_y_cm])
                cos_to_other = np.dot(nose_vec1, vec_to_other) / (
                    np.linalg.norm(nose_vec1) * np.linalg.norm(vec_to_other) + 1e-8
                )

                # === 特徴量としてまとめる ===
                features.append({
                    'lab_id': m1.lab_id,
                    'video_id': vid,
                    'video_frame': frame,
                    'mouse_id': m1.mouse_id,
                    'pair_id': m2.mouse_id,
                    'distance': dist,
                    'rel_speed': rel_speed,
                    'rel_rotate': rel_rotate,
                    'nose_dir_sim': cos_sim,
                    'nose_to_other': cos_to_other,
                    'action': m1.action if 'action' in group.columns else np.nan,
                    'agent_target': m1.agent_target if 'action' in group.columns else np.nan,
                })

    return pd.DataFrame(features)

def shape_data_for_submit(all_data, actions, mice):
    tmp = pd.DataFrame({
        'video_id'   : [],
        'agent_id'   : [],
        'target_id'  : [],
        'action'     : [],
        'start_frame': [],
        'stop_frame' : [],
    })

    target_group = pd.DataFrame()

    # agent mouse
    for action in actions:
        for mouse in mice:
            row_data = all_data[(all_data['action']       == action) &
                                (all_data['mouse_id']     == mouse)  &
                                (all_data['agent_target'] == True)
                                ]
            if len(row_data) > 0:
                row_df = pd.DataFrame()
                if len(row_data) >= 1:
                    row_df = pd.DataFrame({
                        'video_id'   : row_data['video_id'].iloc[0],
                        'agent_id'   : row_data['mouse_id'].iloc[0],
                        'target_id'  : np.nan,
                        'action'     : [action] * 1,
                        'start_frame': row_data['video_frame'].iloc[0],
                        'stop_frame' : row_data['video_frame'].iloc[len(row_data)-1],
                    })

                tmp = pd.concat([tmp, row_df], ignore_index=True)

            row_data = all_data[(all_data['action']       == action) &
                                (all_data['mouse_id']     == mouse)  &
                                (all_data['agent_target'] == False)
                                ]
            if len(row_data) == 0:
                continue
            if len(row_data) >= 1:
                row_df = pd.DataFrame({
                    'video_id'   : row_data['video_id'].iloc[0],
                    'agent_id'   : np.nan,
                    'target_id'  : row_data['mouse_id'].iloc[0],
                    'action'     : [action] * 1,
                    'start_frame': row_data['video_frame'].iloc[0],
                    'stop_frame' : row_data['video_frame'].iloc[len(row_data)-1],
                })
            tmp = pd.concat([tmp, row_df], ignore_index=True)

    tmp['action'] = tmp['action'].map(action_remap)
    # solo_actionのagent_idとtarget_idを一致させる
    for i in range(len(tmp)):
        row = tmp.iloc[i]
        for act in solo_action:
            if row['action'] == act:
                if np.isnan(row['agent_id']):
                    tmp.loc[i, 'agent_id'] = row['target_id']
                if np.isnan(row['target_id']):
                    tmp.loc[i, 'target_id'] = row['agent_id']

    # start_frame順に並び替える
    tmp = tmp.sort_values(by=['start_frame','action'])

    # 1. action + start_time ごとにグループ化
    paired_rows = []
    for (act, start), g in tmp.groupby(['action', 'start_frame']):
        # agent_id のみある行
        agents = g[g['agent_id'].notna() & g['target_id'].isna()].copy()
        # target_id のみある行
        targets = g[g['target_id'].notna() & g['agent_id'].isna()].copy()
        
        used_targets = set()
        for i, agent_row in agents.iterrows():
            # 未使用の target を探す
            available_targets = targets[~targets.index.isin(used_targets)]
            if available_targets.empty:
                continue
            target_row = available_targets.iloc[0]  # 最初のものを対応付け
            used_targets.add(target_row.name)

            # stop_frame は小さい方を採用
            stop_frame = min(agent_row['stop_frame'], target_row['stop_frame'])

            # ペアリング情報を統合
            new_row = agent_row.copy()
            new_row['target_id'] = target_row['target_id']
            new_row['stop_frame'] = stop_frame
            paired_rows.append(new_row)

    # ペアリング結果を DataFrame に
    paired_df = pd.DataFrame(paired_rows)

    # 2. 既存データとマージして、ペアがあるものを更新
    merged = pd.merge(tmp, paired_df, on=['action', 'start_frame', 'agent_id'], how='left', suffixes=('', '_new'))

    # 既にtarget_idが空欄なら新しいものを採用
    merged['target_id'] = merged['target_id'].combine_first(merged['target_id_new'])
    merged['stop_frame'] = merged['stop_frame_new'].combine_first(merged['stop_frame'])

    # 不要列を削除
    merged = merged[tmp.columns]

    adjusted_stop = []

    for i, row in merged.iterrows():
        vid = row["video_id"]

        # 同じ video 内で agent_id と target_id の次の出現を探す
        next_agent = merged[(merged["video_id"] == vid) &
                        ((merged["agent_id"] == row["agent_id"]) |
                        (merged["agent_id"] == row["target_id"]) 
                        ) &
                        (merged["start_frame"] > row["start_frame"])]
        
        next_target = merged[(merged["video_id"] == vid) &
                        ((merged["target_id"] == row["target_id"]) |
                        (merged["target_id"] == row["agent_id"])
                        ) &
                        (merged["start_frame"] > row["start_frame"])]

        # 次の出現フレーム候補を集める
        next_frames = []
        if not next_agent.empty:
            next_frames.append(next_agent["start_frame"].min())
        if not next_target.empty:
            next_frames.append(next_target["start_frame"].min())

        # stop_frame調整判定
        if next_frames:
            next_min = min(next_frames)
            if row["stop_frame"] > next_min:
                stop_frame = next_min - 1
            else:
                stop_frame = row["stop_frame"]
        else:
            stop_frame = row["stop_frame"]

        adjusted_stop.append(stop_frame)

    merged["stop_frame"] = adjusted_stop

    merged.to_csv('tmp.csv')

    return merged


# === メイン ===
if __name__ == "__main__":
    start_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('  start time:', start_time)

    global tracking_method
    tracking_method = get_test_tracking_method()
    train_data = pd.read_csv(INPUT_TRAIN_FILE)
    test_data  = pd.read_csv(INPUT_TEST_FILE)
    train_bodypart,train_action = find_bodypart_action(train_data)
    test_bodypart, test_action = find_bodypart_action(test_data)
    unnecessary_bodypart = list(set(train_bodypart) - set(test_bodypart))
    unnecessary_action   = list(set(train_action)   - set(test_action))
    train_tracking = get_tracking_data(train_data, TRAIN_TRACKING_DIR, unnecessary_bodypart, unnecessary_action, True)
    test_tracking  = get_tracking_data(test_data,  TEST_TRACKING_DIR,  [], [], False)

    train_tracking = add_rotate(train_tracking, True)
    test_tracking  = add_rotate(test_tracking,  False)

#    print('train_tracking:')
#    train_tracking.info()
#    print('test_tracking')
#    test_tracking.info()

    if enable_save_train_data == True:
        train_tracking.to_csv('train_tracking.csv')
        test_tracking.to_csv('test_tracking.csv')

    train_tracking = add_agent_target_features(train_tracking)
    test_tracking  = add_agent_target_features(test_tracking)

    train_tracking = train_tracking.drop('lab_id', axis=1)
    y_agent_target = train_tracking.pop('agent_target')
    test_tracking  = test_tracking.drop('lab_id', axis=1)
    test_tracking  = test_tracking.drop('agent_target', axis=1)

    y = train_tracking.pop('action')
    X = train_tracking

    test_tracking = test_tracking.drop('action', axis=1)

    if enable_save_train_data == True:
        X.to_csv("X.csv", index=False)
        y.to_csv("y.csv", index=False)

    le = LabelEncoder()
    y = le.fit_transform(y)

    X = X.fillna(0)
    test_tracking = test_tracking.fillna(0)

    X['video_id'] = pd.to_numeric(X['video_id'], errors='coerce')
    test_tracking['video_id'] = pd.to_numeric(test_tracking['video_id'], errors='coerce')

    model = xgb.XGBClassifier()
    # ベイズ最適化の設定
    opt = BayesSearchCV(
        estimator=model,
        search_spaces=xgb_param_space,
        n_iter=50,
        cv=3,
        n_jobs=-1,
        random_state=42
    )
    opt.fit(X, y)
    pred_y = opt.best_estimator_.predict(test_tracking)

    # todo: pred_yから提出用形式のDataFrameを作る
    pred_y_trans = le.inverse_transform(pred_y)
    pred_y_trans = pd.DataFrame(pred_y_trans)
    if enable_save_train_data == True:
        pred_y_trans.to_csv('pred_y.csv')

    test_tracking['action'] = pred_y_trans
    if enable_save_train_data == True:
        test_tracking.to_csv('test_tracking_action.csv')

    # agent_targetを求める
    ## actionを追加する
    X['action']             = y
    test_tracking['action'] = pred_y

    # True/Falseを1/0に変換する
    y_agent_target = y_agent_target.fillna(0)
    y_agent_target_df = pd.DataFrame(y_agent_target)
    if enable_save_train_data == True:
        y_agent_target_df.to_csv('y_agent_target_df_1.csv')
    y_agent_target = y_agent_target.map({True:1,False:0})
    y_agent_target_df = pd.DataFrame(y_agent_target)
    if enable_save_train_data == True:
        y_agent_target_df.to_csv('y_agent_target_df_2.csv')

    # クラス不均衡対策
    pos_weight = (len(y_agent_target) - y_agent_target.sum()) / (y_agent_target.sum() + 1e-5)
    model2 = xgb.XGBClassifier(scale_pos_weight=pos_weight)

#    model2 = xgb.XGBClassifier()
    # ベイズ最適化の設定
    opt2 = BayesSearchCV(
        estimator=model2,
        search_spaces=xgb_param_space,
        n_iter=50,
        cv=3,
        n_jobs=-1,
        random_state=42
    )
    y_agent_target_df = pd.DataFrame(y_agent_target)
    if enable_save_train_data == True:
        y_agent_target_df.to_csv('y_agent_target_df_3.csv')
    opt2.fit(X, y_agent_target)
    pred_y = opt2.best_estimator_.predict(test_tracking)
    pred_y_df = pd.DataFrame(pred_y)
    if enable_save_train_data == True:
        pred_y_df.to_csv('pred_y_df.csv')
    agent_target_y = [bool(x) for x in pred_y]
    test_tracking['agent_target'] = agent_target_y
    if enable_save_train_data == True:
        test_tracking.to_csv('test_tracking_action_agent_target.csv')

    end_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    label_mapping = dict(zip(le.classes_, range(len(le.classes_))))

    actions = test_tracking['action'].unique()
    mice    = test_tracking['mouse_id'].unique()

    output_data = shape_data_for_submit(test_tracking, actions, mice)
    output_data = output_data.astype({'video_id': int, 'start_frame': int, 'end_frame': int})
    output_data.to_csv(OUTPUT_FILE)

    print('  end time:', end_time)
    print('complete of script')

