import pandas as pd
import numpy as np
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import datetime
import xgboost
import xgboost as xgb
from pathlib import Path
from itertools import combinations
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tqdm import tqdm

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
    INPUT_TRAIN_FILE = DATA_DIR /"train.csv"
    INPUT_TEST_FILE = DATA_DIR /"test.csv"
    TRAIN_ANNOTATION_DIR   = DATA_DIR /"train_annotation/"
    TRAIN_TRACKING_DIR     = DATA_DIR /"train_tracking/"
    TEST_TRACKING_DIR      = DATA_DIR /"test_tracking/"
    OUTPUT_FILE            = SUBMISSION_DIR / "submission.csv"
else:
    if innter_test == True:
#        INPUT_TRAIN_FILE = "train.csv"
        INPUT_TRAIN_FILE = "train-001.csv"
    else:
        INPUT_TRAIN_FILE  = "../train.csv"
    INPUT_TEST_FILE       = "../test.csv"
    TRAIN_ANNOTATION_DIR  = "../train_annotation/"
    TRAIN_TRACKING_DIR    = "../train_tracking/"
    TEST_TRACKING_DIR     = "../test_tracking/"
    OUTPUT_FILE           = "submission.csv"

bool_map = {
    "True":  1,
    "False": 0,
}

# --- 座標差分と角度系 ---
def compute_angle(dx, dy):
    """差分から角度（度単位）を計算"""
    return np.degrees(np.arctan2(dy, dx))

# resuce row data
# trackingデータをinterval毎に間引く
# src_data : original data
# start_time : start time for reducing
# end_time   : end time for reducing
# interval   : inval time for reducting (sec)
def reduce_rows(src_data, start_time, end_time, interval):
    retval = pd.DataFrame()
    mice = src_data['mouse_id'].unique()
    bodyparts = src_data['bodypart'].unique()
    for mouse in mice:
        for bp in bodyparts:
            tmp_df = src_data[(src_data['mouse_id']==mouse) &
                              (src_data['bodypart']==bp)]
            target_times = np.arange(start_time, end_time + 1e-8, interval)  # end を含めたい場合

            # necessary for sort by key if using merge_asof
            # merge_asof を使う場合はキーでソートが必要
            sorted = tmp_df.sort_values('video_time').reset_index(drop=True)
            target_df = pd.DataFrame({'video_time': target_times})

            # merge 'nearest' line by direction='nearest
            # direction='nearest' で「最も近い」行を結合
            result = pd.merge_asof(target_df, sorted, on='video_time', direction='nearest')
            retval = pd.concat([retval, result], ignore_index=True)
    return retval

def make_tracking_feature(tracking):
    retval = pd.DataFrame()
    tracking = tracking.dropna(subset=["bodypart"])
    bodyparts = tracking["bodypart"].unique()
    bodyparts = np.sort(bodyparts)[::-1]
    pair_list = list(combinations(bodyparts, 2))
    distance_features = []
    for bp1, bp2 in pair_list:
        df1 = tracking[tracking["bodypart"] == bp1][["video_time", "mouse_id", "x_cm", "y_cm"]].rename(columns={"x_cm": f"x_cm_{bp1}", "y_cm": f"y_cm_{bp1}"})
        df2 = tracking[tracking["bodypart"] == bp2][["video_time", "mouse_id", "x_cm", "y_cm"]].rename(columns={"x_cm": f"x_cm_{bp2}", "y_cm": f"y_cm_{bp2}"})
        merged = pd.merge(df1, df2, on=["video_time", "mouse_id"], how="inner")
        merged[f"dist_{bp1}_{bp2}"] = np.sqrt((merged[f"x_cm_{bp1}"] - merged[f"x_cm_{bp2}"])**2 + (merged[f"y_cm_{bp1}"] - merged[f"y_cm_{bp2}"])**2)
            
        distance_features.append(merged[["video_time", "mouse_id", f"dist_{bp1}_{bp2}"]])

    print("len(distance_features):",len(distance_features))

    distance_df = distance_features[0]
    for df in distance_features[1:]:
        distance_df = pd.merge(distance_df, df, on=["video_time", "mouse_id"], how="outer")

    distance_df.to_csv('distance_df.csv')

    # === 統合 ===
    # bodypart単位の特徴（速度・角度など）をpivot
    tracking_pivot = tracking.pivot_table(
#        index=["video_id", "video_frame", "mouse_id"],
        index=["video_time", "mouse_id"],
        columns="bodypart",
        values=["dx_cm", "dy_cm", "speed", "angle", "d_angle"]
    )
    tracking_pivot.to_csv('tracking_pivot.csv')

    # カラム階層をフラット化
    tracking_pivot.columns = [f"{feat}_{bp}" for feat, bp in tracking_pivot.columns]
    tracking_pivot = tracking_pivot.reset_index()
    tracking_pivot.to_csv('tracking_pivot-2.csv')

    # 全ての特徴をマージ
#    tracking_features = pd.merge(tracking_pivot, distance_df, on=["video_id", "video_frame", "mouse_id"], how="left")
    tracking_features = pd.merge(tracking_pivot, distance_df, on=["video_time", "mouse_id"], how="left")
    tracking_features.to_csv('tracking_features.csv')

    # body_centerのx,yを追加する
    tmp_center = tracking[tracking['bodypart'] == 'body_center']
    tmp_center.to_csv('tmp_center.csv')

    tmp_center = tmp_center.drop(['bodypart','dx_cm','dy_cm','speed','angle','d_angle'],axis=1)
    tracking_features = pd.merge(tracking_features, tmp_center, on=['video_time', 'mouse_id'], how='left')
    tracking_features = tracking_features.rename(columns={'x_cm':'x_cm_body_center', 'y_cm':'y_cm_body_center'})

    tracking_features.to_csv('tracking_features.csv')


    retval = pd.concat([retval, tracking_features])

    return retval


# 次のデータフレームを作成する
#   video_id
#   video_frame
#   video_time
#   mouse_id
#   action
#   mouse部位情報
#   target_id
#   target mouse部位情報
#   action mouse-target mouse部位距離



# trackingファイルを読み込む
def load_tracking(train):
    retval = pd.DataFrame()
    for i in range(len(train)):
        train_row = train.iloc[i]
        vid = str(train_row['video_id'])
        lid = train_row['lab_id']
        fps  = train_row['frames_per_second']
        ppcm = train_row['pix_per_cm_approx']
        file_path = lid + "/" + vid + ".parquet"
        tracking_data_path = os.path.join(TRAIN_TRACKING_DIR, file_path)
        if os.path.exists(tracking_data_path) == False:
            print('not found ', tracking_data_path)
            continue
        trk = pd.read_parquet(tracking_data_path)
        trk['video_id'] = vid
        trk['video_time'] = trk['video_frame'] / fps
        trk['x_cm'] = trk['x'] / ppcm
        trk['y_cm'] = trk['y'] / ppcm
        start_time = trk['video_time'].min()
        end_time   = trk['video_time'].max()
        trk_reduced = reduce_rows(trk, start_time, end_time, INTERVAL)
        tracking = (
            trk_reduced
#            .groupby(["video_id", "mouse_id", "bodypart"], group_keys=False)
            .groupby(["mouse_id", "bodypart"], group_keys=False)
            .apply(lambda g: g.assign(
                dx_cm=g["x_cm"].diff().fillna(0),
                dy_cm=g["y_cm"].diff().fillna(0),
                speed=np.sqrt(g["x_cm"].diff()**2 + g["y_cm"].diff()**2).fillna(0),
                angle=compute_angle(g["x_cm"].diff(), g["y_cm"].diff())
            ))
        )
        # --- 回転角（角度変化） ---
        tracking["d_angle"] = (
#            tracking.groupby(["video_id", "mouse_id", "bodypart"])["angle"]
            tracking.groupby(["mouse_id", "bodypart"])["angle"]
            .diff()
            .fillna(0)
        )
        tracking.info()
        trk_featured = make_tracking_feature(tracking)
        retval = pd.concat([retval, trk_featured])
    return retval

def load_annotation(train):
    retval = pd.DataFrame()
    for i in range(len(train)):
        train_row = train.iloc[i]
        vid = str(train_row['video_id'])
        lid = train_row['lab_id']
        file_path = lid + "/" + vid + "/" + ".parquet"
        annotation_data_path = os.path.join(TRAIN_ANNOTATION_DIR, file_path)
        if os.path.exists(annotation_data_path) == False:
            continue
        trk = pd.read_parquet(annotation_data_path)
        retval = pd.concat([retval, trk])
    return retval


def make_feature(tracking, annotation):
    retval = pd.DataFrame()



    return retval

def merge_tracking_annotation(tracking, annotation):
    retval = pd.DataFrame()
    for (agent,act),df in annotation.groupby(['agent_id', 'action']):
        for i in range(len(df)):
            annotation_row = df.iloc[i]
            target_mouse = annotation_row['target_id']
            start_frame  = annotation_row['start_frame']
            stop_frame   = annotation_row['stop_frame']
            tmp_df = tracking[(tracking['mouse_id']    == agent) &
                              (tracking['start_frame'] >= start_frame) &
                              (tracking['stop_frame']  <= stop_frame)]
            if len(tmp_df) == 0:
                continue
            target_info = tracking[(tracking['mouse_id'] == target_mouse) &
                                   (tracking['start_frame'] >= start_frame) &
                                   (tracking['stop_frame']  <= stop_frame)]
            

    return retval

# === メイン ===
if __name__ == "__main__":
    train_data = pd.read_csv(INPUT_TRAIN_FILE)
    tracking_data = load_tracking(train_data)
    if len(tracking_data) == 0:
        print("can't read tracking_data")
        exit()
    tracking_data.info()
    annotation_data = load_annotation(train_data)
    tracking_data.to_csv("tracking.csv")
    tracking_data_merged_annotation = merge_tracking_annotation(tracking_data, annotation_data)
