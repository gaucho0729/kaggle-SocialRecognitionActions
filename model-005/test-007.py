# test-006.pyにnose/bodycenter,hipのspeed、angleを追加して学習させるスクリプト

import pandas as pd
import numpy as np
import joblib
import os
import multiprocessing
import datetime
from tqdm import tqdm
from itertools import combinations, product
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from pathlib import Path

kaggle = False
innter_test = False
enable_save_train_data = True

# 
INTERVAL = 0.25

row_threshold = 10000

# Variables for environment (環境) -------------------------
if kaggle == True:
    DATA_DIR = Path("/kaggle/input/MABe-mouse-behavior-detection")
    SUBMISSION_DIR = Path("/kaggle/working")
    INPUT_TRAIN_FILE = DATA_DIR /"train.csv"
    INPUT_TEST_FILE = DATA_DIR /"test.csv"
    TRAIN_ANNOTATION_DIR   = DATA_DIR /"train_annotation/"
    TRAIN_TRACKING_DIR     = DATA_DIR /"train_tracking/"
    TEST_TRACKING_DIR      = DATA_DIR /"test_tracking/"
    OUTPUT_FILE            = SUBMISSION_DIR / "submission.csv"
else:
    if innter_test == True:
        INPUT_TRAIN_FILE = "train.csv"
    else:
        INPUT_TRAIN_FILE  = "../train.csv"
    INPUT_TEST_FILE       = "../test.csv"
    TRAIN_ANNOTATION_DIR  = "../train_annotation/"
    TRAIN_TRACKING_DIR    = "../train_tracking/"
    TEST_TRACKING_DIR     = "../test_tracking/"
    OUTPUT_FILE           = "submission.csv"

# === 設定 ===
TRACKING_FILE      = "tracking_features_simplified.csv"   # tracking_features.csv でも可
ANNOTATION_FILE    = "annotation.csv"
PAIR_DATA_SET_FILE = "pair_dataset_windowed.csv"
MODEL_FILE         = "xgb_mouse_behavior_model.pkl"

window_size = 5   # 前後5フレームで統計特徴を計算
neg_sampling_ratio = 1.0  # 正例1件あたり負例を何件残すか (Noneなら全件保持)

# カラム名の確認・設定
VID_COL = 'video_id'
FRAME_COL = 'video_frame'
MID_COL = 'mouse_id'

solo_action = [
    'rear'
]

def compute_angle(dx, dy):
    """差分から角度（度単位）を計算"""
    return np.degrees(np.arctan2(dy, dx))

def add_speed_and_angle(df):
    tmp_all = []
    for (mouse_id,bodypart),group in df.groupby(['mouse_id','bodypart']):
        speed=np.sqrt(group["x_cm"].diff()**2 + group["y_cm"].diff()**2)
        angle=compute_angle(group["x_cm"].diff(), group["y_cm"].diff())
        length = len(group)
        if length != 0:
            tmp = pd.DataFrame({
                'video_frame': group['video_frame'],
                'mouse_id'   : np.full(len(group), mouse_id, dtype='int8'),
                'bodypart'   : bodypart,
                'speed'      : speed,
                'angle'      : angle
            })
            tmp_all.append(tmp)
        else:
            print(f"[WARN]: result of groupby({mouse_id}) is 0")
    tmp_all = pd.concat(tmp_all, ignore_index=True)
    df = pd.merge(df, tmp_all, on=['video_frame', 'mouse_id','bodypart'], how='left')
    return df

# resuce row data
# trackingデータをinterval毎に間引く
# src_data : original data
# start_time : start time for reducing
# end_time   : end time for reducing
# interval   : inval time for reducting (sec)
def reduce_rows(src_data, start_time, end_time, interval):
    target_times = np.arange(start_time, end_time + 1e-8, interval)
    target_df = pd.DataFrame({'video_time': target_times})
    retval = []

    # 各 (mouse_id, bodypart) ごとに merge_asof し、結果を append
    for (mouse, bp), group in src_data.groupby(['mouse_id', 'bodypart'], sort=False):
        sorted_group = group.sort_values('video_time', ignore_index=True)
        merged = pd.merge_asof(target_df, sorted_group, on='video_time', direction='nearest')
        merged['mouse_id'] = mouse
        merged['bodypart'] = bp
        retval.append(merged)

    if len(retval) > 0:
        return pd.concat(retval, ignore_index=True)
    return pd.DataFrame()

def get_bodypart_df(df, bodypart):
    retval = pd.DataFrame()
    tmp = df[df['bodypart']==bodypart]
    if len(tmp) > 0:
        t = tmp.iloc[0]
        retval = pd.DataFrame({
            "x_cm": [t['x_cm']],
            "y_cm": [t['y_cm']]
        })
    else:
        retval = pd.DataFrame({
            "x_cm": [np.nan],
            "y_cm": [np.nan]
        })
    return retval

# trackingデータフレームからbodypartデータフレームに変換する
def reformat_tracking(df):
#    print("[DEBUG] reformat_tracking input columns:", df.columns.tolist())
    # bodypartを列に変換
    pivoted = df.pivot_table(
        index=['video_id', 'video_frame', 'mouse_id'],
        columns='bodypart',
        values=['x_cm', 'y_cm','speed','angle'],
        aggfunc='first'
    )
    pivoted.columns = [f'{a}_{b}' for a, b in pivoted.columns]
    pivoted = pivoted.reset_index()

    # hip の平均処理
    if 'hip_x_cm' in pivoted.columns:
        pivoted['hip_x_cm'] = pivoted['x_cm_hip_left', 'x_cm_hip_right'].mean(axis=1)
    else:
        pivoted['hip_x_cm'] = np.nan
    
    if 'hip_y_cm' in pivoted.columns:
        pivoted['hip_y_cm'] = pivoted['y_cm_hip_left', 'y_cm_hip_right'].mean(axis=1)
    else:
        pivoted['hip_y_cm'] = np.nan

    # 欠損を除外（nose, body_centerが欠けている行）
    if 'x_cm_nose' in pivoted.columns:
        pivoted = pivoted.dropna(subset=['x_cm_nose'])
    else:
        pivoted['x_cm_nose'] = np.nan
        pivoted['y_cm_nose'] = np.nan

    if 'x_cm_body_center' in pivoted.columns:
        pivoted = pivoted.dropna(subset=['x_cm_body_center'])
    else:
        pivoted['x_cm_body_center'] = np.nan
        pivoted['y_cm_body_center'] = np.nan

    if 'speed_hip' in pivoted.columns:
        pivoted = pivoted.dropna(subset=['speed_hip'])
    else:
        pivoted['speed_hip'] = np.nan

    if 'angle_hip' in pivoted.columns:
        pivoted = pivoted.dropna(subset=['angle_hip'])
    else:
        pivoted['angle_hip'] = np.nan

    if 'speed_nose' in pivoted.columns:
        pivoted = pivoted.dropna(subset=['speed_nose'])
    else:
        pivoted['speed_nose'] = np.nan

    if 'angle_nose' in pivoted.columns:
        pivoted = pivoted.dropna(subset=['angle_nose'])
    else:
        pivoted['angle_nose'] = np.nan

    if 'speed_body_center' in pivoted.columns:
        pivoted = pivoted.dropna(subset=['speed_body_center'])
    else:
        pivoted['speed_body_center'] = np.nan

    if 'angle_body_center' in pivoted.columns:
        pivoted = pivoted.dropna(subset=['angle_body_center'])
    else:
        pivoted['angle_body_center'] = np.nan

    # 列名を整理
    if 'x_cm_nose' in pivoted.columns:
        pivoted = pivoted.rename(columns={
        'x_cm_nose': 'nose_x_cm',
        'y_cm_nose': 'nose_y_cm',
    })
    if 'x_cm_body_center' in pivoted.columns:
        pivoted = pivoted.rename(columns={
            'x_cm_body_center': 'bodycenter_x_cm',
            'y_cm_body_center': 'bodycenter_y_cm',
        })
    if 'angle_body_center' in pivoted.columns:
        pivoted = pivoted.rename(columns={
        'angle_body_center': 'bodycenter_angle',
    })
    if 'speed_body_center' in pivoted.columns:
        pivoted = pivoted.rename(columns={
        'speed_body_center': 'bodycenter_speed',
    })

    if 'angle_nose' in pivoted.columns:
        pivoted = pivoted.rename(columns={
        'angle_nose': 'nose_angle',
    })
    if 'speed_nose' in pivoted.columns:
        pivoted = pivoted.rename(columns={
        'speed_nose': 'nose_speed',
    })

    if 'angle_hip' in pivoted.columns:
        pivoted = pivoted.rename(columns={
        'angle_hip': 'hip_angle',
    })
    if 'speed_hip' in pivoted.columns:
        pivoted = pivoted.rename(columns={
        'speed_hip': 'hip_speed',
    })

    return pivoted[[
        'video_id', 'video_frame', 'mouse_id',
        'nose_x_cm', 'nose_y_cm','nose_speed','nose_angle',
        'bodycenter_x_cm', 'bodycenter_y_cm','bodycenter_speed', 'bodycenter_angle',
        'hip_x_cm', 'hip_y_cm','hip_speed','hip_angle',
        
    ]]

def wrap_up_features(df):
#    print("[DEBUG] wrap_up_features input columns:", df.columns.tolist())
    results = []
    bodyparts = ['nose', 'bodycenter', 'hip']
    mouse_pairs = list(combinations([1, 2, 3, 4], 2))
    bp_pairs = list(product(bodyparts, bodyparts))
    mice = df['mouse_id'].unique()

    for (video_id, video_frame), group in df.groupby(['video_id', 'video_frame']):
        # bodypart ごとに安全に座標を抽出
        coord = {}
        row = {'video_id': video_id, 'video_frame': video_frame}
        for bp in bodyparts:
            xcol = f'{bp}_x_cm'
            ycol = f'{bp}_y_cm'
            if xcol in group.columns and ycol in group.columns:
                if not group.empty:
                    coord[bp] = group.set_index('mouse_id')[[xcol, ycol]]
                else:
                    coord[bp] = pd.DataFrame(columns=[xcol, ycol])
            else:
                coord[bp] = pd.DataFrame(columns=[xcol, ycol])

            for mouse in mice:
                tmp = group[group['mouse_id']==mouse]
                if len(tmp)==0:
                    continue
                row[f'{bp}{mouse}_speed'] = tmp[f'{bp}_speed'].iloc[0]
                row[f'{bp}{mouse}_angle'] = tmp[f'{bp}_angle'].iloc[0]

        # 各マウス・各部位ペア間の距離を計算
        for (m1, m2), (bp1, bp2) in product(mouse_pairs, bp_pairs):
            key = f'{bp1}{m1}_{bp2}{m2}'
            try:
                if bp1 not in coord or bp2 not in coord:
                    row[key] = np.nan
                    continue

                m1_data = coord[bp1].loc[m1] if m1 in coord[bp1].index else None
                m2_data = coord[bp2].loc[m2] if m2 in coord[bp2].index else None
                if m1_data is None or m2_data is None:
                    row[key] = np.nan
                    continue

                m1x, m1y = m1_data.values
                m2x, m2y = m2_data.values

                if pd.notna(m1x) and pd.notna(m2x) and pd.notna(m1y) and pd.notna(m2y):
                    row[key] = (m1x - m2x)**2 + (m1y - m2y)**2
                else:
                    row[key] = np.nan

            except Exception:
                row[key] = np.nan

        results.append(row)

    if len(results) == 0:
        print("results is empty")
    return pd.DataFrame(results)

def get_tracking_label(features, annos):
    action_list_1 = []
    action_list_2 = []
    agent_list_1  = []
    agent_list_2  = []
    target_list_1  = []
    target_list_2  = []
    for (video_frame), groups in features.groupby(['video_frame']):
        action_anno = annos[(annos['start_frame']<=video_frame) & (annos['stop_frame']>=video_frame)]
        if len(action_anno) == 0:
            action_list_1.append('none')
            action_list_2.append('none')
            agent_list_1.append(np.nan)
            target_list_1.append(np.nan)
            agent_list_2.append(np.nan)
            target_list_2.append(np.nan)
            continue
        tmp = action_anno.iloc[0]
        action1 = tmp['action']
        action_list_1.append(action1)
        agent_list_1.append(tmp['agent_id'])
        target_list_1.append(tmp['target_id'])
        if len(action_anno) >= 2:
            tmp = action_anno.iloc[1]
            action2 = tmp['action']
            action_list_2.append(action2)
            agent_list_2.append(tmp['agent_id'])
            target_list_2.append(tmp['target_id'])
        else:
            action_list_2.append('none')
            agent_list_2.append(np.nan)
            target_list_2.append(np.nan)
    return action_list_1, action_list_2,agent_list_1,agent_list_2,target_list_1,target_list_2


def process_train_video(train_row):
    vid  = str(train_row["video_id"])
    lid  = train_row["lab_id"]
    fps  = train_row["frames_per_second"]
    ppcm = train_row["pix_per_cm_approx"]

    # --- ファイルパス構築 ---
    tracking_path = os.path.join(TRAIN_TRACKING_DIR, f"{lid}/{vid}.parquet")
    annotation_path = os.path.join(TRAIN_ANNOTATION_DIR, f"{lid}/{vid}.parquet")
    if not os.path.exists(tracking_path) :
        print("not found tracking file.")
        print("  ", tracking_path)
        return pd.DataFrame()

    if not os.path.exists(annotation_path):
        print("not found annotation file.")
        print("  ", annotation_path)
        return pd.DataFrame()

    # --- データ読み込み ---
    trk = pd.read_parquet(tracking_path)
#    print("[DEBUG] trk columns after read:", trk.columns.tolist())
    ann = pd.read_parquet(annotation_path)
    trk["video_id"] = vid
    trk["video_time"] = trk["video_frame"] / fps
    trk["x_cm"] = trk["x"] / ppcm
    trk["y_cm"] = trk["y"] / ppcm
    ann["video_id"] = vid
    start_time = trk['video_time'].min()
    end_time = trk['video_time'].max()

    # bodypartでデータを間引く
    trk = trk[(trk['bodypart'] == 'body_center') |
              (trk['bodypart'] == 'nose') |
              (trk['bodypart'] == 'hip_left') |
              (trk['bodypart'] == 'hip_right')
              ]
    try:
        trk = reduce_rows(trk, start_time, end_time, INTERVAL)
    except Exception as e:
        raise ValueError("start_time=", start_time, " / end_time=", end_time)

    if len(trk) == 0:
        return pd.DataFrame()

    trk = add_speed_and_angle(trk)
    tracking_shrink = reformat_tracking(trk)
    tracking_features = wrap_up_features(tracking_shrink)
    action_list_1, action_list_2,agent_list_1,agent_list_2,target_list_1,target_list_2 = get_tracking_label(tracking_features, ann)
    tracking_features['action1']  = action_list_1
    tracking_features['agent_1']  = agent_list_1
    tracking_features['target_1'] = target_list_1
    tracking_features['action2']  = action_list_2
    tracking_features['agent_2']  = agent_list_2
    tracking_features['target_2'] = target_list_2
    return tracking_features

def process_test_video(test_row):
    vid  = str(test_row["video_id"])
    lid  = test_row["lab_id"]
    fps  = test_row["frames_per_second"]
    ppcm = test_row["pix_per_cm_approx"]

    # --- ファイルパス構築 ---
    tracking_path = os.path.join(TEST_TRACKING_DIR, f"{lid}/{vid}.parquet")
    if not os.path.exists(tracking_path) :
        print("not found tracking file.")
        print("  ", tracking_path)
        return pd.DataFrame()

    # --- データ読み込み ---
    trk = pd.read_parquet(tracking_path)
    trk["video_id"] = vid
    trk["video_time"] = trk["video_frame"] / fps
    trk["x_cm"] = trk["x"] / ppcm
    trk["y_cm"] = trk["y"] / ppcm

    start_time = trk['video_time'].min()
    end_time = trk['video_time'].max()

    # bodypartでデータを間引く
    trk = trk[(trk['bodypart'] == 'body_center') |
              (trk['bodypart'] == 'nose') |
              (trk['bodypart'] == 'hip_left') |
              (trk['bodypart'] == 'hip_right')
              ]
    try:
        trk = reduce_rows(trk, start_time, end_time, INTERVAL)
    except Exception as e:
        raise ValueError("start_time=", start_time, " / end_time=", end_time)

    if len(trk) == 0:
        return pd.DataFrame()

    print('add_speed_and_angle')
    trk = add_speed_and_angle(trk)
    print('reformat_tracking')
    tracking_shrink = reformat_tracking(trk)
    print('wrap_up_features')
    tracking_features = wrap_up_features(tracking_shrink)
    return tracking_features

def train_and_predict_action(train_org, test_org):
    train = train_org.copy()
    test = test_org.copy()
    label_y1 = train.pop('action1')
    label_y2 = train.pop('action2')
    label_agent1  = train.pop('agent_1')
    label_agent2  = train.pop('agent_2')
    label_target1 = train.pop('target_1')
    label_target2 = train.pop('target_2')

    # 欠損値処理
    X = train
    X = X.drop('video_id', axis=1)
    X = X.fillna(X.median())

    print("len(train)=",len(train))
    print("len(test)=",len(test))
    remove_cols = ['video_id', 'action1', 'action2', 'agent_1', 'agent_2', 'target_1', 'target_2']
    for col in remove_cols:
        test = test.drop(col, axis=1)
    test_X = test
    test_X.info()
    test_X = test_X.fillna(test_X.median())

    # === ラベルの分布確認 ===
    y = label_y1
    print("\nラベル分布:")
    print(y.value_counts())

    # === モデルとパラメータ探索設定 ===
    model = XGBClassifier(
        n_estimators  =  400,
        learning_rate = 0.05,
        max_depth     = 6,
        objective     = 'multi:softmax',
        num_class     = len(y.unique()),
        eval_metric   = 'logloss',
        tree_method   = 'hist',  # GPU利用可なら 'gpu_hist'
        subsample     = 0.9,
        colsample_bytree = 0.9,
        use_label_encoder=False,
        n_job = -1,
        random_state=42
    )

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    X_sample = X.sample(frac=0.2, random_state=42)
    y_sample = y.loc[X_sample.index]
    print("len(y_sample)=", len(y_sample))

    le = LabelEncoder()
    y_encoded = le.fit_transform(y_sample)
    y_encoded = pd.DataFrame(y_encoded)

    print("len(X)=", len(X))
    print("len(y)=", len(y))
    print("len(X_sample)=", len(X_sample))
    print("len(y_sample)=", len(y_sample))
    print("len(y_encoded)=", len(y_encoded))

    # === 学習 ===
    model.fit(X_sample, y_encoded)

    # === 最終学習済みモデルで予測と評価 ===
    y_pred_value = model.predict(X)
    y_pred = le.inverse_transform(y_pred_value)
    print("\n=== 訓練データでの性能評価 ===")
    print(classification_report(y, y_pred))

    # 混同行列も確認
    print("\n=== 混同行列 ===")
    print(confusion_matrix(y, y_pred))

    test_y_pred_value = model.predict(test_X)

    test_y_pred_label = le.inverse_transform(test_y_pred_value)
    return test_y_pred_label


def train_and_predict_agent(train_org, test_org, pred_action):
#    label_y1 = train.pop('action1')
    train = train_org.copy()
    test = test_org.copy()
    label_y2 = train.pop('action2')
    label_agent1  = train.pop('agent_1')
    label_agent2  = train.pop('agent_2')
    label_target1 = train.pop('target_1')
    label_target2 = train.pop('target_2')

    remove_cols = ['video_id', 'action1', 'action2', 'agent_1', 'agent_2', 'target_1', 'target_2']
    for col in remove_cols:
        test = test.drop(col, axis=1)

    le = LabelEncoder()
    train['action1'] = le.fit_transform(train['action1'])
    test['action1'] = pred_action
    test['action1'] = le.fit_transform(test['action1'])

    # 欠損値処理
    X = train.copy()
    X = X.drop('video_id', axis=1)
    
    X = X.fillna(X.median())
    test_X = test
    test_X = test_X.fillna(test_X.median())

#    test_X = test_X[test_X['action'] != 'none']

    # === ラベルの分布確認 ===
    y = label_agent1
    y = y.fillna(0)
    print("\nラベル分布:")
    print(y.value_counts())

    # === モデルとパラメータ探索設定 ===
    model = XGBClassifier(
        n_estimators  =  400,
        learning_rate = 0.05,
        max_depth     = 6,
        objective     = 'multi:softmax',
        num_class     = len(y.unique()),
#        eval_metric='mlogloss',
        eval_metric   = 'logloss',
        tree_method   = 'hist',  # GPU利用可なら 'gpu_hist'
        subsample     = 0.9,
        colsample_bytree = 0.9,
        use_label_encoder=False,
        n_job = -1,
        random_state=42
    )

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    X_sample = X.sample(frac=0.2, random_state=42)
    y_sample = y.loc[X_sample.index]
    print("len(y_sample)=", len(y_sample))

    print("len(X)=", len(X))
    print("len(y)=", len(y))
    print("len(X_sample)=", len(X_sample))
    print("len(y_sample)=", len(y_sample))
    print("len(y_encoded)=", len(y_sample))

    # === 学習 ===
    model.fit(X_sample, y_sample)

    # === 最終学習済みモデルで予測と評価 ===
    y_pred_value = model.predict(X)
    print("\n=== 訓練データでの性能評価 ===")
    print(classification_report(y, y_pred_value))

    # 混同行列も確認
    print("\n=== 混同行列 ===")
    print(confusion_matrix(y, y_pred_value))

    test_y_pred_value = model.predict(test_X)

    return test_y_pred_value


def train_and_predict_target(train_org, test_org, pred_action, pred_agent):
    train = train_org.copy()
    test = test_org.copy()
#    label_y1 = train.pop('action1')
    label_y2 = train.pop('action2')
#    label_agent1  = train.pop('agent_1')
    label_agent2  = train.pop('agent_2')
    label_target1 = train.pop('target_1')
    label_target2 = train.pop('target_2')

    remove_cols = ['video_id', 'action1', 'action2', 'agent_1', 'agent_2', 'target_1', 'target_2']
    for col in remove_cols:
        test = test.drop(col, axis=1)

    le = LabelEncoder()
    train['action1'] = le.fit_transform(train['action1'])
    test['action1'] = pred_action
    test['action1'] = le.fit_transform(test['action1'])
    train['agent_1'] = train['agent_1'].fillna(0)
    test['agent_1'] = pred_agent
    test['agent_1'] = test['agent_1'].fillna(0)

    # 欠損値処理
    X = train.copy()
    X = X.drop('video_id', axis=1)
    X = X.fillna(X.median())
    test_X = test.copy()
    test_X = test_X.fillna(test_X.median())

    # 欠損値処理
    test_X = test
    test_X = test_X.fillna(test_X.median())

    # === ラベルの分布確認 ===
    y = label_target1
    y = y.fillna(0)
    print("\nラベル分布:")
    print(y.value_counts())

    # === モデルとパラメータ探索設定 ===
    model = XGBClassifier(
        n_estimators  =  400,
        learning_rate = 0.05,
        max_depth     = 6,
        objective     = 'multi:softmax',
        num_class     = len(y.unique()),
#        eval_metric='mlogloss',
        eval_metric   = 'logloss',
        tree_method   = 'hist',  # GPU利用可なら 'gpu_hist'
        subsample     = 0.9,
        colsample_bytree = 0.9,
        use_label_encoder=False,
        n_job = -1,
        random_state=42
    )

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    X_sample = X.sample(frac=0.2, random_state=42)
    y_sample = y.loc[X_sample.index]
    print("len(y_sample)=", len(y_sample))

    le = LabelEncoder()
    X_sample['action1'] = le.fit_transform(X_sample['action1'])
    test_X['action1'] = le.fit_transform(test_X['action1'])

    print("len(X)=", len(X))
    print("len(y)=", len(y))
    print("len(X_sample)=", len(X_sample))
    print("len(y_sample)=", len(y_sample))

    # === 学習 ===
    model.fit(X_sample, y_sample)

    # === 最終学習済みモデルで予測と評価 ===
    y_pred_value = model.predict(X)
    print("\n=== 訓練データでの性能評価 ===")
    print(classification_report(y, y_pred_value))

    # 混同行列も確認
    print("\n=== 混同行列 ===")
    print(confusion_matrix(y, y_pred_value))

    test_y_pred_value = model.predict(test_X)

    return test_y_pred_value


def make_submission(df):
#    df = df[df['action'] != 'none']
    frames = df['video_frame'].unique()
    start_frame = -1
    prev_agent  = 0
    prev_target = 0
    prev_action = ''
    retval = pd.DataFrame()
    for frm in frames:
        tmp = df[df['video_frame']==frm].iloc[0]
        if tmp['action'] != prev_action:
            if start_frame >= 0:
                if (tmp['action'] != prev_action):
                    if (prev_action != 'none'):
                        if prev_action in solo_action:
                            target = prev_agent
                        else:
                            target = prev_target
                        tmp_df = pd.DataFrame({
                            'video_id'   : [tmp['video_id']],
                            'agent_id'   : [prev_agent],
                            'target_id'  : [target],
                            'action'     : [prev_action],
                            'start_frame': [start_frame],
                            'end_frame'  : [tmp['video_frame']-1]
                        })
                        retval = pd.concat([retval, tmp_df], ignore_index=True)
                    start_frame = tmp['video_frame']
                    prev_action = tmp['action']
                    prev_agent  = tmp['agent_id']
                    prev_target = tmp['target_id']
            else:
                start_frame = tmp['video_frame']
                prev_action = tmp['action']
                prev_agent  = tmp['agent_id']
                prev_target = tmp['target_id']
    return retval

def train_and_predict(train, test):
    actions = train_and_predict_action(train, test)
    agent = train_and_predict_agent(train, test, actions)
    target = train_and_predict_target(train, test, actions, agent)
    test['action']    = actions
    test['agent_id']  = agent
    test['target_id'] = target
    submission = make_submission(test)
    return submission

def main_kaggle():
    start_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("[START TIME]", start_time)
    train_data = pd.read_csv(INPUT_TRAIN_FILE)

    n_jobs = min(8, os.cpu_count())
    print(f"並列ワーカー数: {n_jobs}")

    output_path = PAIR_DATA_SET_FILE
    all_pairs = []  # appendモードも可
    n_jobs = 4
    futures = {}

    # trainデータ生成
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futures = {ex.submit(process_train_video, train_data.iloc[i]): i for i in range(len(train_data))}
        for fut in tqdm(as_completed(futures), total=len(futures)):
            idx = futures[fut]
            df = fut.result()
            if len(df) > 0:
                all_pairs.append(df)
                print(f"✅ processed video {idx+1}/{len(train_data)} ({len(df):,} rows)")
            try:
                df = fut.result()
                if len(df) > 0:
                    all_pairs.append(df)
                    print(f"✅ processed video {idx+1}/{len(train_data)} ({len(df):,} rows)")
            except Exception as e:
                print(f"❌ error in train video {idx}: {e}")

    if len(all_pairs) == 0:
        print("[ERROR] can't make train data.")
        print("[STOP TIME]", end_time)
        print(f"{start_time}-{end_time}: aborted")
        exit()

    tracking_features = pd.concat(all_pairs)

    # testデータ生成
    all_pairs = []  # appendモードも可
    test_data = pd.read_csv(INPUT_TEST_FILE)
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futures = {ex.submit(process_test_video, test_data.iloc[i]): i for i in range(len(test_data))}
        for fut in tqdm(as_completed(futures), total=len(futures)):
            idx = futures[fut]
            df = fut.result()
            if len(df) > 0:
                all_pairs.append(df)
                print(f"✅ processed video {idx+1}/{len(test_data)} ({len(df):,} rows)")
            try:
                df = fut.result()
                if len(df) > 0:
                    all_pairs.append(df)
                    print(f"✅ processed video {idx+1}/{len(test_data)} ({len(df):,} rows)")
            except Exception as e:
                print(f"❌ error in train video {idx}: {e}")

    test_features = pd.concat(all_pairs)
    print("test_features:")
    test_features.info()
    submission = train_and_predict(tracking_features, test_features)
    submission.to_csv(OUTPUT_FILE)

def main_local():
    start_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("[START TIME]", start_time)
    train_data = pd.read_csv(INPUT_TRAIN_FILE)

    n_jobs = min(8, os.cpu_count())
    print(f"並列ワーカー数: {n_jobs}")

    output_path = PAIR_DATA_SET_FILE
    all_pairs = []  # appendモードも可
    futures = {}

    # trainデータ生成
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futures = {ex.submit(process_train_video, train_data.iloc[i]): i for i in range(len(train_data))}
        for fut in tqdm(as_completed(futures), total=len(futures)):
            idx = futures[fut]
            try:
                df = fut.result()
                if len(df) > 0:
                    all_pairs.append(df)
                    print(f"✅ processed video {idx+1}/{len(train_data)} ({len(df):,} rows)")
            except Exception as e:
                print(f"❌ error in train video {idx}: {e}")
    if len(all_pairs) == 0:
        print("[ERROR] can't make train data.")
        print("[STOP TIME]", end_time)
        print(f"{start_time}-{end_time}: aborted")
        exit()

    tracking_features = pd.concat(all_pairs)
    tracking_features.to_csv('tracking_features-105.csv')

    # testデータ生成
    test_data = pd.read_csv(INPUT_TEST_FILE)
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futures = {ex.submit(process_test_video, test_data.iloc[i]): i for i in range(len(test_data))}
        for fut in tqdm(as_completed(futures), total=len(futures)):
            idx = futures[fut]
            try:
                df = fut.result()
                if len(df) > 0:
                    all_pairs.append(df)
                    print(f"✅ processed video {idx+1}/{len(test_data)} ({len(df):,} rows)")
            except Exception as e:
                print(f"❌ error in train video {idx}: {e}")
    if len(all_pairs) == 0:
        print("[ERROR] can't make train data.")
        print("[STOP TIME]", end_time)
        exit()

    test_features = pd.concat(all_pairs)
    submission = train_and_predict(tracking_features, test_features)
    submission.to_csv(OUTPUT_FILE)

if __name__ == "__main__":
    start_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("[START TIME]", start_time)

    if kaggle == True:
        main_kaggle()
    else:
        main_local()

    end_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    print("[STOP TIME]", end_time)
    print(f"End of complete {start_time}-{end_time}")
