import os
import datetime
import json
import pandas as pd
import numpy as np
import xgboost
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

kaggle = False
innter_test = True
enable_save_train_data = True

# 
INTERVAL = 0.25

row_threshold = 10000

# Variables for environment (環境) -------------------------
if kaggle == True:
    DATA_DIR = Path("/kaggle/input/MABe-mouse-behavior-detection")
    SUBMISSION_DIR = Path("/kaggle/working")
    INPUT_TRAIN_FILE = DATA_DIR / "train.csv"
    INPUT_TEST_FILE = DATA_DIR / "test.csv"
    TRAIN_ANNOTATION_DIR   = DATA_DIR / "train_annotation"
    TRAIN_TRACKING_DIR     = DATA_DIR / "train_tracking"
    TEST_TRACKING_DIR      = DATA_DIR / "test_tracking"
    OUTPUT_FILE            = SUBMISSION_DIR / "submission.csv"
else:
    if innter_test == True:
        INPUT_TRAIN_FILE = "train.csv"
    else:
        INPUT_TRAIN_FILE = "../train.csv"
    INPUT_TEST_FILE = "../test.csv"
    TRAIN_ANNOTATION_DIR   = "../train_annotation"
    TRAIN_TRACKING_DIR     = "../train_tracking"
    TEST_TRACKING_DIR      = "../test_tracking"
    OUTPUT_FILE            = "submission.csv"

# -----------------------------
# 行動ラベルと単独行動
# -----------------------------
action_remap = {
    0: 'approach',
    1: 'attack',
    2: 'avoid',
    3: 'chase',
    4: 'chaseattack',
    5: 'rear',
    6: 'submit'
}
solo_action = ['rear']

pair_actions = [
    'approach', 'attack', 'avoid', 'chase', 'chaseattack', 'submit'
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
#    print(start_time,'-',end_time)
#    print(src_data['bodypart'].unique())
    target_times = np.arange(start_time, end_time + 1e-8, interval)  # end を含めたい場合

    # necessary for sort by key if using merge_asof
    # merge_asof を使う場合はキーでソートが必要
    sorted = src_data.sort_values('video_time').reset_index(drop=True)
    target_df = pd.DataFrame({'video_time': target_times})

    # merge 'nearest' line by direction='nearest
    # direction='nearest' で「最も近い」行を結合
    result = pd.merge_asof(target_df, sorted, on='video_time', direction='nearest')
    return result

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

def wrap_up_body_center(src_data,lab_id,video_id):
    mice = src_data['mouse_id'].unique()
    frms = src_data['video_frame'].unique()
#    print("len(mice):",len(mice))
#    print("len(frms):",len(frms))
    retval = pd.DataFrame()
    for mouse in mice:
        for frm in frms:
            tmp = src_data[(src_data['mouse_id']==mouse) &
                           (src_data['video_frame']==frm)]
            tmp_center = tmp[tmp['bodypart']=='body_center']
            tmp_nose   = tmp[tmp['bodypart']=='nose']
            if (len(tmp_center) == 0) | (len(tmp_nose) == 0):
                continue
            tmp2 = pd.DataFrame({
                    'lab_id'       : [lab_id]                            * 1,
                    'video_id'     : [video_id]                          * 1,
                    'video_frame'  : [tmp_center['video_frame'].iloc[0]] * 1,
                    'video_time'   : [tmp_center['video_time'].iloc[0]]  * 1,
                    'mouse_id'     : [tmp_center['mouse_id'].iloc[0]]    * 1,
                    'center_x_cm'  : [tmp_center['x_cm'].iloc[0]]        * 1,
                    'center_y_cm'  : [tmp_center['y_cm'].iloc[0]]        * 1,
                    'center_vx_cm' : [tmp_center['vx_cm'].iloc[0]]       * 1,
                    'center_vy_cm' : [tmp_center['vy_cm'].iloc[0]]       * 1,
                    'nose_x_cm'    : [tmp_nose['x_cm'].iloc[0]]          * 1,
                    'nose_y_cm'    : [tmp_nose['y_cm'].iloc[0]]          * 1,
                    'nose_vx_cm'   : [tmp_nose['vx_cm'].iloc[0]]         * 1,
                    'nose_vy_cm'   : [tmp_nose['vy_cm'].iloc[0]]         * 1,
            })
            tmp2['rotate'] = np.arctan2(tmp2['center_y_cm']-tmp2['nose_y_cm'], tmp2['center_x_cm']-tmp2['nose_x_cm'])
            retval = pd.concat([retval,tmp2],ignore_index=True)
    return retval

def add_speed(src_df):
    retval = pd.DataFrame()
    grouped = src_df.groupby(['mouse_id','bodypart'])
    for (mouse,bodypart),group  in grouped:
        tmp = group.sort_values(by='video_frame')
        tmp['vx_cm'] = tmp['x_cm'].diff()
        tmp['vy_cm'] = tmp['y_cm'].diff()
        retval = pd.concat([retval,tmp], ignore_index=True)

    return retval

# --- 各ビデオ単位で特徴量を生成する関数（並列実行用） ---
def process_video_feature(args):
    train, tracking_dir, unnecessary_bodypart, unnecessary_action = args
    video_id = str(train['video_id'])
    lab_id = train['lab_id']
    tracking_file_name = lab_id + '/' + video_id + '.parquet'
    tracking_full_path = os.path.join(tracking_dir, tracking_file_name)
    if os.path.exists(tracking_full_path) == False:
        return pd.DataFrame()

    fps  = train['frames_per_second']
    ppcm = train['pix_per_cm_approx']
    df = pd.read_parquet(tracking_full_path)

    # 不要部位除去
    if len(unnecessary_bodypart) > 0:
        for bp in unnecessary_bodypart:
            df = df[df['bodypart'] != bp]

    df['video_time'] = df['video_frame'] / fps
    df['x_cm'] = df['x'] / ppcm
    df['y_cm'] = df['y'] / ppcm
    start_time = df['video_time'].min()
    end_time   = df['video_time'].max()

    tmp = pd.DataFrame()
    for mouse in df['mouse_id'].unique():
        for bp in df['bodypart'].unique():
            tmp_bp_mouse = df[(df['bodypart'] == bp) & (df['mouse_id'] == mouse)]
            tmp_reduced_time = reduce_rows(tmp_bp_mouse, start_time, end_time, INTERVAL)
            tmp_bp_mouse_spd = add_speed(tmp_reduced_time)
            tmp = pd.concat([tmp, tmp_bp_mouse_spd], ignore_index=True)

    df2 = wrap_up_body_center(tmp, lab_id, video_id)
    if len(df2) == 0:
        return pd.DataFrame()

    # フレームごとにペア特徴量生成
    all_features = []
    grouped = df2.groupby("video_frame")
    for frame, group in grouped:
        mice = group.to_dict("records")
        n = len(mice)
        if n == 1:
            m1 = mice[0]
            all_features.append({
                "lab_id": lab_id,
                "video_id": video_id,
                "video_frame": frame,
                "mouse_i": m1["mouse_id"],
                "mouse_j": m1["mouse_id"],
                "distance": 0,
                "rel_speed": 0,
                "rel_rotate": 0,
                "nose_dir_sim": 1,
                "nose_to_other": 0,
                "action_pair": "rear"
            })
            continue
        for i, m1 in enumerate(mice):
            for j, m2 in enumerate(mice):
                if i == j:
                    continue
                dx = m2["center_x_cm"] - m1["center_x_cm"]
                dy = m2["center_y_cm"] - m1["center_y_cm"]
                dist = np.sqrt(dx**2 + dy**2)
                dvx = m2["center_vx_cm"] - m1["center_vx_cm"]
                dvy = m2["center_vy_cm"] - m1["center_vy_cm"]
                rel_speed = np.sqrt(dvx**2 + dvy**2)
                rel_rotate = m2["rotate"] - m1["rotate"]
                v1 = np.array([m1["nose_vx_cm"], m1["nose_vy_cm"]])
                v2 = np.array([m2["nose_vx_cm"], m2["nose_vy_cm"]])
                cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
                to_other = np.array([dx, dy])
                cos_to_other = np.dot(v1, to_other) / (np.linalg.norm(v1) * np.linalg.norm(to_other) + 1e-6)
                all_features.append({
                    "lab_id": lab_id,
                    "video_id": video_id,
                    "video_frame": frame,
                    "mouse_i": m1["mouse_id"],
                    "mouse_j": m2["mouse_id"],
                    "distance": dist,
                    "rel_speed": rel_speed,
                    "rel_rotate": rel_rotate,
                    "nose_dir_sim": cos_sim,
                    "nose_to_other": cos_to_other,
                    "action_pair": None
                })
    return pd.DataFrame(all_features)


# -----------------------------
# ペア特徴量生成
#   @param  train_test_data : 
#   @param  tracking_dir    : トラッキングファイル名(フルパス)
#   @param  unnecessary_bodypart : 不要部位
#   @param  unnecessary_action   : 不要行動
#   @return マウスをペア化したデータフレーム
#       lab_id
#       video_id
#       video_frame:   frame,
#       mouse_i:       m1["mouse_id"],
#       mouse_j:       m2["mouse_id"],
#       distance:      dist,
#       rel_speed:     rel_speed,
#       rel_rotate:    rel_rotate,
#       nose_dir_sim:  cos_sim,
#       nose_to_other: cos_to_other,
#       action_pair:   None
# -----------------------------
def create_pair_features_parallel(train_test_data, tracking_dir, unnecessary_bodypart, unnecessary_action, max_workers=6):
    tasks = []
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for i in range(len(train_test_data)):
            train = train_test_data.iloc[i]
            args = (train, tracking_dir, unnecessary_bodypart, unnecessary_action)
            tasks.append(executor.submit(process_video_feature, args))
        for future in as_completed(tasks):
            df = future.result()
            if not df.empty:
                results.append(df)
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()

def create_pair_features(train_test_data,tracking_dir):
    retval = pd.DataFrame()
    for i in range(len(train_test_data)):
        train_test = train_test_data.iloc[i]
        video_id = str(train_test['video_id'])
        lab_id = train_test['lab_id']
        tracking_file_name = lab_id + '/' + video_id + '.parquet'
        tracking_full_path = os.path.join(tracking_dir, tracking_file_name)
        if os.path.exists(tracking_full_path) == False:
            continue

        fps  = train_test_data['frames_per_second']
        ppcm = train_test_data['pix_per_cm_approx']
        df = pd.read_parquet(tracking_full_path)

        df['video_time'] = df['video_frame'] / fps
        df['x_cm'] = df['x'] / ppcm
        df['y_cm'] = df['y'] / ppcm
        start_time = df['video_time'].min()
        end_time   = df['video_time'].max()

        tmp = pd.DataFrame()
        for mouse in df['mouse_id'].unique():
            for bp in df['bodypart'].unique():
                tmp_bp_mouse = df[(df['bodypart'] == bp) & (df['mouse_id'] == mouse)]
                tmp_reduced_time = reduce_rows(tmp_bp_mouse, start_time, end_time, INTERVAL)
                tmp_bp_mouse_spd = add_speed(tmp_reduced_time)
                tmp = pd.concat([tmp, tmp_bp_mouse_spd], ignore_index=True)

        df2 = wrap_up_body_center(tmp, lab_id, video_id)
        if len(df2) == 0:
            continue

        # フレームごとにペア特徴量生成
        all_features = []
        grouped = df2.groupby("video_frame")
        for frame, group in grouped:
            mice = group.to_dict("records")
            n = len(mice)
            if n == 1:
                m1 = mice[0]
                all_features.append({
                    "lab_id": lab_id,
                    "video_id": video_id,
                    "video_frame": frame,
                    "mouse_i": m1["mouse_id"],
                    "mouse_j": m1["mouse_id"],
                    "distance": 0,
                    "rel_speed": 0,
                    "rel_rotate": 0,
                    "nose_dir_sim": 1,
                    "nose_to_other": 0,
                    "action_pair": "rear"
                })
                continue
            for i, m1 in enumerate(mice):
                for j, m2 in enumerate(mice):
                    if i == j:
                        continue
                    dx = m2["center_x_cm"] - m1["center_x_cm"]
                    dy = m2["center_y_cm"] - m1["center_y_cm"]
                    dist = np.sqrt(dx**2 + dy**2)
                    dvx = m2["center_vx_cm"] - m1["center_vx_cm"]
                    dvy = m2["center_vy_cm"] - m1["center_vy_cm"]
                    rel_speed = np.sqrt(dvx**2 + dvy**2)
                    rel_rotate = m2["rotate"] - m1["rotate"]
                    v1 = np.array([m1["nose_vx_cm"], m1["nose_vy_cm"]])
                    v2 = np.array([m2["nose_vx_cm"], m2["nose_vy_cm"]])
                    cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
                    to_other = np.array([dx, dy])
                    cos_to_other = np.dot(v1, to_other) / (np.linalg.norm(v1) * np.linalg.norm(to_other) + 1e-6)
                    all_features.append({
                        "lab_id": lab_id,
                        "video_id": video_id,
                        "video_frame": frame,
                        "mouse_i": m1["mouse_id"],
                        "mouse_j": m2["mouse_id"],
                        "distance": dist,
                        "rel_speed": rel_speed,
                        "rel_rotate": rel_rotate,
                        "nose_dir_sim": cos_sim,
                        "nose_to_other": cos_to_other,
                        "action_pair": None
                    })
        retval = pd.concat([retval,pd.DataFrame(all_features)], ignore_index=True)
    return retval


# -----------------------------
# 教師データ結合
#   @param  pair_df        : ペア化されたデータフレーム
#   @param  annotation_dir : annotationディレクトリ
#   @return 
# -----------------------------
def merge_with_labels(train_data, pair_df, annotation_dir, unnecessary_action):
    label_list = []
#    for file in sorted(os.listdir(annotation_dir)):
#        if not file.endswith(".parquet"):
#            continue
#        df = pd.read_parquet(os.path.join(annotation_dir, file))
#        df["lab_id"] = file.replace(".parquet","")
#        label_list.append(df)

    for i in range(len(train_data)):
        train = train_data.iloc[i]
        lab_id = str(train['lab_id'])
        video_id = str(train['video_id'])
        train_file_name = lab_id + '/' + video_id + '.parquet'
        train_file = os.path.join(annotation_dir, train_file_name)
        if os.path.exists(train_file) == False:
            continue
        df = pd.read_parquet(train_file)
        df['lab_id'] = lab_id
        df['video_id'] = video_id

        # 不要行動除去
        if len(unnecessary_action) > 0:
            for act in unnecessary_action:
                df = df[df['action'] != act]

        label_list.append(df)

    label_df = pd.concat(label_list, ignore_index=True)
    print("pair_df:")
    pair_df.info()
    print("label_df:")
    label_df.info()

    merged = pd.merge(pair_df, label_df, on=["lab_id","video_id","video_frame","mouse_i","mouse_j"], how="left")
    merged["action_pair"] = merged["action_pair"].fillna(merged.get("action", None))
    return merged


def merge_with_labels_fast(train_data, pair_df, annotation_dir, unnecessary_action):
    label_df = pd.DataFrame()
    for i in range(len(train_data)):
        train = train_data.iloc[i]
        lab_id = str(train['lab_id'])
        video_id = str(train['video_id'])
        train_file_name = lab_id + '/' + video_id + '.parquet'
        train_file = os.path.join(annotation_dir, train_file_name)
        if os.path.exists(train_file) == False:
            continue
        df = pd.read_parquet(train_file)
        df['lab_id'] = lab_id
        df['video_id'] = video_id

        # 不要行動除去
        if len(unnecessary_action) > 0:
            for act in unnecessary_action:
                df = df[df['action'] != act]

        label_df = pd.concat([label_df, df])

    # 左右のキー列をNumPy配列で取り出す
    pair_lab = pair_df["lab_id"].values
    pair_vid = pair_df["video_id"]
    pair_frame = pair_df["video_frame"].values
    pair_i = pair_df["mouse_i"].values
    pair_j = pair_df["mouse_j"].values

    # ラベル側の配列
    lab_lab = label_df["lab_id"].values
    lab_vid = label_df["video_id"]
    lab_agent = label_df["agent_id"].values
    lab_target = label_df["target_id"].values
    lab_start = label_df["start_frame"].values
    lab_stop = label_df["stop_frame"].values
    lab_action = label_df["action"].values

    # 出力用の空配列
    matched_action = np.full(len(pair_df), np.nan, dtype=object)

    # 辞書化で lab_id ごとにラベルをまとめる
    from collections import defaultdict
    lab_dict = defaultdict(list)
    for i in range(len(label_df)):
        lab_dict[lab_lab[i]].append(i)

    # 各 lab_id ごとにマッチング
    for lab_id in lab_dict.keys():
        mask = pair_lab == lab_id
        p_idx = np.where(mask)[0]
        if len(p_idx) == 0:
            continue

        # ラベル候補
        l_idx = lab_dict[lab_id]
        for li in l_idx:
            # 範囲条件とマウス一致条件をベクトル化判定
            cond = (
                (pair_frame[p_idx] >= lab_start[li]) &
                (pair_frame[p_idx] <= lab_stop[li]) &
                (pair_i[p_idx] == lab_agent[li]) &
                (pair_j[p_idx] == lab_target[li])
            )
            cond |= (
                (pair_frame[p_idx] >= lab_start[li]) &
                (pair_frame[p_idx] <= lab_stop[li]) &
                (pair_i[p_idx] == lab_target[li]) &
                (pair_j[p_idx] == lab_agent[li])
            )
            matched_action[p_idx[cond]] = lab_action[li]

    # 結合結果を DataFrame に追加
    pair_df = pair_df.copy()
    pair_df["action"] = matched_action

    pair_df["action_pair"] = pair_df["action_pair"].fillna(pair_df.get("action", None))

    return pair_df



# -----------------------------
# XGBoost学習
#   @param  train_df : 訓練データフレーム
# -----------------------------
def train_xgb(train_df):
    print('train_xgb():')
    print('train_df:')
    train_df.info()
    df = train_df.dropna(subset=["action_pair"]).copy()
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["action_pair"])
    feature_cols = ["distance","rel_speed","rel_rotate","nose_dir_sim","nose_to_other"]
    X = df[feature_cols]
    y = df["label"]
    X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.2,random_state=42)

    print('X_train:')
    X_train.info()

    model = XGBClassifier(n_estimators=400,
                          max_depth=6,
                          learning_rate=0.05,
                          subsample=0.9,
                          colsample_bytree=0.9,
                          tree_method="hist",
                          random_state=42)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_val)
    print("=== Validation Report ===")
    print(classification_report(y_val, y_pred, target_names=le.classes_))
    return model, le, feature_cols

def remove_self_actions(df, actions_to_remove):
    """
    自分自身への行動を削除する。

    Parameters
    ----------
    df : pd.DataFrame
        agent_id, target_id, action を含む DataFrame
    actions_to_remove : list of str
        削除対象の行動名リスト（例：["attack", "chase"]）
    """
    cond = (df["agent_id"] == df["target_id"]) & (df["action"].isin(actions_to_remove))
    removed_count = cond.sum()
    df = df[~cond].reset_index(drop=True)
    print(f"削除行数: {removed_count}")
    return df


def build_submission_2(pred_df):
    retval = pd.DataFrame()
    tmp = pred_df.copy()
    tmp['used'] = [False] * len(pred_df)
    video_ids = pred_df['video_id'].unique()
    mice = pred_df['mouse_id'].unique()
    for video_id in video_ids:
        print('video_id:', video_id)
        for mouse in mice:
            tmp_df = pred_df[(pred_df['mouse_id']==mouse) & (pred_df['video_id']==video_id)]
#            frms = tmp_df['video_frame'].unique()
            actions = tmp_df['pred_action'].unique()
            for action in actions:
                tmp_act = tmp_df[tmp_df['pred_action']==action]
                if len(tmp_act)==0:
                    continue
                prev_frame = 0
                start_frame = tmp_act['video_frame'].iloc[0]
                frms = tmp_act['video_frame'].unique()
                for frm in frms:
                    tmp = tmp_act[tmp_act['video_frame']==frm]
                    if tmp['video_frame'].iloc[0] - prev_frame >= 60:
                        if tmp['agent_target'].iloc[0] == True:
                            agent = tmp['mouse_id'].iloc[0]
                            target = tmp['target_id'].iloc[0]
                            if prev_frame != 0:
                                row = pd.DataFrame({
                                        'video_id'   : [video_id]    * 1,
                                        'agent_id'   : [agent]       * 1,
                                        'target_id'  : [target]      * 1,
                                        'action'     : [action]      * 1,
                                        'start_frame': [start_frame] * 1,
                                        'stop_frame' : [prev_frame]  * 1
                                })
                                retval = pd.concat([retval, row], ignore_index=True)
                        start_frame = frm
                    prev_frame = tmp['video_frame'].iloc[0]
    return retval

def remove_pair_actions(df, actions_to_remove):
    """
    自分自身への行動を削除する。

    Parameters
    ----------
    df : pd.DataFrame
        agent_id, target_id, action を含む DataFrame
    actions_to_remove : list of str
        削除対象の行動名リスト（例：["attack", "chase"]）
    """
    cond = (df["agent_id"] != df["target_id"]) & (df["action"].isin(actions_to_remove))
    removed_count = cond.sum()
    df = df[~cond].reset_index(drop=True)
    print(f"削除行数: {removed_count}")
    return df

def remove_same_start_stop_frame(df):
    cond = (df['start_frame'] == df['stop_frame'])
    removed_count = cond.sum()
    df = df[~cond].reset_index(drop=True)
    print(f"削除行数: {removed_count}")
    return df

# 提出用フォーマットに整形し直す
def shape_submission_df(df):
    df = build_submission_2(df)
    df['agent_id']  = df['agent_id'].astype('Int64')
    df['target_id'] = df['target_id'].astype('Int64')

    df = df.sort_values(by=['start_frame'],ascending=[True])
    df = df.dropna(subset=['target_id'])
    df = df.dropna(subset=['agent_id'])
    df = remove_self_actions(df, pair_actions)
    df = remove_pair_actions(df, solo_action)
    df = remove_same_start_stop_frame(df)
    return df

def canonical_pair(row):
    """ペアをソートして正規化"""
    a, t = sorted([row['agent_id'], row['target_id']])
    return pd.Series({'p_agent': a, 'p_target': t})

# === 攻撃側を agent に統一 ===
def unify_attack_direction(sub):
    # 攻撃がない場合は小さい方をagent
    if not (sub['action'] == 'attack').any():
        a, t = sorted([sub['p_agent'].iloc[0], sub['p_target'].iloc[0]])
        return pd.Series({'agent_id': a, 'target_id': t})

    # 攻撃がある場合 → その行のscoreを使って判定
    attacks = sub[sub['action'] == 'attack']
    if len(attacks) == 1:
        row = attacks.iloc[0]
        return pd.Series({'agent_id': row['agent_id'], 'target_id': row['target_id']})
    else:
        # 複数攻撃あり → scoreが高い方をagent
        top = attacks.loc[attacks['score'].idxmax()]
        return pd.Series({'agent_id': top['agent_id'], 'target_id': top['target_id']})


def clear_contradiction(df):
    # スコアがなければ仮で 1.0
    if "score" not in df.columns:
        df["score"] = 1.0

    # 行動優先順位（上ほど優先度高い）
    priority = ['chase', 'approach', 'attack', 'avoid', 'rear', 'submit']
    priority_map = {a: i for i, a in enumerate(priority)}

    # === 1. agent_id と target_id の逆転重複を整理 ===
    df[['p_agent', 'p_target']] = df.apply(canonical_pair, axis=1)

    # === 2. 同一ペア・重複フレームを統合 ===
    merged = []
    for (vid, a, t), sub in df.groupby(['video_id', 'p_agent', 'p_target']):
        sub = sub.sort_values(['start_frame', 'stop_frame']).reset_index(drop=True)

        # 重複区間を統合（行動優先度を考慮）
        merged_rows = []
        current = sub.iloc[0].copy()
        for _, row in sub.iloc[1:].iterrows():
            overlap = not (row['start_frame'] > current['stop_frame'])
            if overlap:
                # 行動が違う場合：スコア or 優先度で選択
                if row['score'] > current['score']:
                    current['action'] = row['action']
                    current['score'] = row['score']
                elif row['score'] == current['score']:
                    if priority_map[row['action']] < priority_map[current['action']]:
                        current['action'] = row['action']
                # フレーム結合
                current['stop_frame'] = max(current['stop_frame'], row['stop_frame'])
            else:
                merged_rows.append(current)
                current = row.copy()
        merged_rows.append(current)

    clean_df = pd.DataFrame(merged_rows)
    merged_df = pd.DataFrame(merged_rows)

    # 各ペアごとにagent-target方向を決定
    agent_target_df = merged_df.groupby(['video_id', 'p_agent', 'p_target']).apply(unify_attack_direction).reset_index()

    # === 4. 結合して方向統一 ===
    final_df = merged_df.merge(agent_target_df, on=['video_id', 'p_agent', 'p_target'], how='left')

    final_df.info()
    for col in ['agent_id', 'target_id']:
        if f'{col}_y' in final_df.columns:
            final_df[col] = final_df[f'{col}_y']
        elif f'{col}_x' in final_df.columns:
            final_df[col] = final_df[f'{col}_x']

    # === 5. 統一方向に反転 ===
    # （p_agent/p_targetと決定されたagent/targetが逆ならswap）
    swap_mask = (final_df['p_agent'] == final_df['target_id']) & (final_df['p_target'] == final_df['agent_id'])
    final_df.loc[swap_mask, ['agent_id', 'target_id']] = final_df.loc[swap_mask, ['target_id', 'agent_id']].values

    # === 6. 出力整形 ===
    final_df = final_df[['video_id', 'agent_id', 'target_id', 'action', 'start_frame', 'stop_frame', 'score']]
    final_df = final_df.sort_values(['video_id', 'start_frame']).reset_index(drop=True)
    final_df.to_csv("submission_cleaned_scored.csv", index=False)

    final_df.index.name = 'row_index'
    print("✅ 出力:", final_df.shape)
    print(final_df.head(10))

    return final_df

def merge_overlapping_actions(df):
    """
    同一video_id, agent_id, target_id, actionの重複・隣接をマージ
    """
    merged = []
    for (vid, a, t, act), group in df.groupby(["video_id", "agent_id", "target_id", "action"]):
        group = group.sort_values("start_frame")
        cur_start, cur_stop = None, None
        for _, row in group.iterrows():
            if cur_start is None:
                cur_start, cur_stop = row.start_frame, row.stop_frame
            elif row.start_frame <= cur_stop + 1:
                cur_stop = max(cur_stop, row.stop_frame)
            else:
                merged.append([vid, a, t, act, cur_start, cur_stop])
                cur_start, cur_stop = row.start_frame, row.stop_frame
        merged.append([vid, a, t, act, cur_start, cur_stop])
    return pd.DataFrame(merged, columns=["video_id", "agent_id", "target_id", "action", "start_frame", "stop_frame"])

def resolve_reversed_actions(df):
    df = df.sort_values(["video_id", "action", "start_frame"])
    remove_idx = set()
    for (vid, act), group in df.groupby(["video_id", "action"]):
        for i, row_i in group.iterrows():
            for j, row_j in group.iterrows():
                if i >= j: continue
                if row_i.agent_id == row_j.target_id and row_i.target_id == row_j.agent_id:
                    # 時間区間が重なっていれば短い方を削除
                    overlap = min(row_i.stop_frame, row_j.stop_frame) - max(row_i.start_frame, row_j.start_frame)
                    if overlap > 0:
                        len_i = row_i.stop_frame - row_i.start_frame
                        len_j = row_j.stop_frame - row_j.start_frame
                        if len_i < len_j:
                            remove_idx.add(i)
                        else:
                            remove_idx.add(j)
    return df.drop(index=list(remove_idx)).reset_index(drop=True)

def remove_conflicting_actions(df):
    df = df.sort_values(["video_id", "start_frame"])
    clean = []
    for (vid, a, t), group in df.groupby(["video_id", "agent_id", "target_id"]):
        last_stop = -1
        for _, row in group.iterrows():
            if row.start_frame <= last_stop:
                # 重なっている -> スキップ（またはルールに応じて選択）
                continue
            clean.append(row)
            last_stop = row.stop_frame
    return pd.DataFrame(clean)


# -----------------------------
# 推論 + agent/target補正 + 提出CSV生成
#   @param  model        : XGBoostモデル
#   @param  le           : ラベルエンコーダー
#   @param  feature_cols : 
#   @param  pair_df      : ペア化されたデータフレーム
#   @param  out_csv      : 出力csvファイル
# -----------------------------
def predict_pipeline(model, le, feature_cols, pair_df, out_csv="submission.csv"):
    X = pair_df[feature_cols]
    pred_labels = model.predict(X)
    pred_actions = le.inverse_transform(pred_labels)
    pair_df = pair_df.copy()
    pair_df["pred_action"] = pred_actions

    # agent/target補正
    solo_action = ['rear']
    pair_agent_actions = ['approach','attack','chase','chaseattack']
    pair_target_actions = ['avoid','submit']

    pair_df['agent_target'] = False
    pair_df['target_id'] = None

    mask_solo = pair_df['pred_action'].isin(solo_action)
    pair_df.loc[mask_solo,'agent_target'] = True
    pair_df.loc[mask_solo,'target_id'] = None

    mask_agent = pair_df['pred_action'].isin(pair_agent_actions)
    pair_df.loc[mask_agent,'agent_target'] = True
    pair_df.loc[mask_agent,'target_id'] = pair_df.loc[mask_agent,'mouse_j']

    mask_target = pair_df['pred_action'].isin(pair_target_actions)
    pair_df.loc[mask_target,'agent_target'] = False
    pair_df.loc[mask_target,'target_id'] = pair_df.loc[mask_target,'mouse_j']

    # 優先度
    action_priority = {'attack':0,'chaseattack':1,'chase':2,'approach':3,'avoid':4,'submit':5,'rear':6}
    pair_df['priority'] = pair_df['pred_action'].map(action_priority)

    print("pair_df:")
    pair_df.info()

    # mouse_i & mouse_j を結合して重複除去
    mouse_i_df = pair_df.sort_values(['lab_id','video_id','video_frame','mouse_i','priority','agent_target'],ascending=[True]*6)
    mouse_i_df = mouse_i_df.drop_duplicates(subset=['lab_id','video_id','video_frame','mouse_i'], keep='first')
    mouse_j_df = pair_df.sort_values(['lab_id','video_id','video_frame','mouse_j','priority','agent_target'],ascending=[True]*6)
    mouse_j_df = mouse_j_df.drop_duplicates(subset=['lab_id','video_id','video_frame','mouse_j'], keep='first')
    mouse_j_df = mouse_j_df.rename(columns={'mouse_j':'mouse_id'})[['lab_id','video_id','video_frame','mouse_id','pred_action','agent_target','target_id','priority']]

    submission_df = pd.concat([
        mouse_i_df[['lab_id','video_id','video_frame','mouse_i','pred_action','agent_target','target_id','priority']].rename(columns={'mouse_i':'mouse_id'}),
        mouse_j_df
    ], ignore_index=True)

    # 最終整形：mouse_idごとに優先度最大の1行を残す
    submission_df.sort_values(['lab_id','video_id','video_frame','mouse_id','priority'], inplace=True)
    submission_df = submission_df.drop_duplicates(subset=['lab_id','video_id','video_frame','mouse_id'], keep='first')
    submission_df.reset_index(drop=True, inplace=True)
    submission_df.drop(columns=['priority'], inplace=True)

    submission_df.to_csv('submission-0.csv')
    submission_df = shape_submission_df(submission_df)

    submission_df = merge_overlapping_actions(submission_df)
    submission_df = resolve_reversed_actions(submission_df)
    submission_df = remove_conflicting_actions(submission_df)
    submission_df = submission_df.sort_values(by='start_frame')

    submission_df.to_csv(out_csv, index=False)
    print(f"✅ Submission CSV saved as {out_csv}")
    return submission_df

# -----------------------------
# ワンステップ実行関数
#   @param  train_tracking_dir      訓練用トラッキングディレクトリ
#   @param  train_annotation_dir    訓練用アノテーションディレクトリ
#   @param  test_tracking_dir       評価用トラッキングディレクトリ
#   @param  out_csv                 出力ファイル名
# -----------------------------
def run_full_pipeline(train_tracking_dir, train_annotation_dir, test_tracking_dir, out_csv="submission_final.csv"):
    global train_data
    global test_data

    train_data = pd.read_csv(INPUT_TRAIN_FILE)
    test_data  = pd.read_csv(INPUT_TEST_FILE)

    test_tracking_methods = test_data['tracking_method'].unique()
    # testで出てくるtracking_methodのみにする
    train_data = train_data[train_data['tracking_method'].isin(test_tracking_methods)]

    train_bodypart,train_action = find_bodypart_action(train_data)
    test_bodypart, test_action = find_bodypart_action(test_data)
    unnecessary_bodypart = list(set(train_bodypart) - set(test_bodypart))
    unnecessary_action   = list(set(train_action)   - set(test_action))

    # 特徴量生成
    print("🔹 Creating training features...")
    train_features = create_pair_features_parallel(train_data, train_tracking_dir, unnecessary_bodypart, unnecessary_action, max_workers=os.cpu_count())
    train_merged = merge_with_labels_fast(train_data, train_features, train_annotation_dir, unnecessary_action)

    #

    if enable_save_train_data == True:
        train_merged.to_csv('train_merged.csv')

    print('train_merged:')
    train_merged.info()

    train_merged = train_merged.dropna(subset=['action'])
    if enable_save_train_data == True:
        train_merged.to_csv('train_merged_drop_na.csv')

    # 学習
    print("🔹 Training XGBoost model...")
    model, le, feature_cols = train_xgb(train_merged)

    # テスト特徴量生成
    print("🔹 Creating test features...")
    test_features = create_pair_features_parallel(test_data,test_tracking_dir, [], [], max_workers=os.cpu_count())

    # 推論＋agent/target補正＋提出CSV生成
    print("🔹 Running prediction pipeline...")
    submission_df = predict_pipeline(model, le, feature_cols, test_features, out_csv=out_csv)
    return submission_df


if __name__ == "__main__":
    start_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('  start time:', start_time)

    submission_df = run_full_pipeline(
        train_tracking_dir=TRAIN_TRACKING_DIR,
        train_annotation_dir=TRAIN_ANNOTATION_DIR,
        test_tracking_dir=TEST_TRACKING_DIR,
        out_csv=OUTPUT_FILE
    )

    end_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('  end time:', end_time)
    print('complete of script')
