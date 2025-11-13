import pandas as pd
import numpy as np

# action毎に主な部位間距離との相関係数を計算してみる

# nose,body,hip

# nose1-nose2,nose1-body2,nose1-hip2
# body1-nose2,body1-body2,body1-hip2
# hip1-nose2,hip1-body2,hip1-hip2

# 1-2,1-3,1-4,2-3,2-4,3-4,

import pandas as pd
import numpy as np
import joblib
import os
import multiprocessing
import datetime
from tqdm import tqdm
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from pathlib import Path

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
    output = pd.DataFrame()
    frame = df['video_frame'].unique()
    for (video_id,frame,mouse_id), groups in df.groupby(['video_id','video_frame','mouse_id']):
        to_be_removed = False
        nose       = get_bodypart_df(groups, 'nose').iloc[0]
        bodycenter = get_bodypart_df(groups, 'body_center').iloc[0]
        hip_left   = get_bodypart_df(groups, 'hip_left').iloc[0]
        hip_right  = get_bodypart_df(groups, 'hip_right').iloc[0]
        if (~np.isnan(hip_left['x_cm'])) & (~np.isnan(hip_right['x_cm'])):
            hip = pd.DataFrame({
                'x_cm': [(hip_left['x_cm'] + hip_right['x_cm']) / 2],
                'y_cm': [(hip_left['y_cm'] + hip_right['y_cm']) / 2]
            }).iloc[0]
        elif ~np.isnan(hip_left['x_cm']):
            hip = pd.DataFrame({
                'x_cm': [hip_left['x_cm']],
                'y_cm': [hip_left['y_cm']]
            }).iloc[0]
        elif ~np.isnan(hip_right['x_cm']):
            hip = pd.DataFrame({
                'x_cm': [hip_right['x_cm']],
                'y_cm': [hip_right['y_cm']]
            }).iloc[0]
        else:
            hip = pd.DataFrame({
                "x_cm": [np.nan],
                "y_cm": [np.nan]
            }).iloc[0]
        if (np.isnan(nose['x_cm'])) or (np.isnan(bodycenter['x_cm'])):
            to_be_removed = True

        tmp = pd.DataFrame({
            'video_id': [video_id],
            'video_frame': [frame],
            'mouse_id': [mouse_id],
            'nose_x_cm': [nose['x_cm']],
            'nose_y_cm': [nose['y_cm']],
            'bodycenter_x_cm': [bodycenter['x_cm']],
            'bodycenter_y_cm': [bodycenter['y_cm']],
            'hip_x_cm': [hip['x_cm']],
            'hip_y_cm': [hip['y_cm']],
        })
        if to_be_removed == False:
            output = pd.concat([output, tmp])
    return output


if __name__ == "__main__":
    TRAIN_FILE = "tracking-44566106.csv"
    tracking = pd.read_csv(TRAIN_FILE)
    tracking['x_cm'] = tracking['x']
    tracking['y_cm'] = tracking['y']
    tracking_shrink = reformat_tracking(tracking)
#    tracking_shrink['video_id'] = str(tracking_shrink['video_id'])
    tracking_shrink['video_frame'] = tracking_shrink['video_frame'].astype('Int64')
    tracking_shrink['mouse_id'] = tracking_shrink['mouse_id'].astype('Int64')
#    tracking_shrink = tracking_shrink[(tracking_shrink['bodycenter_x_cm']!=np.nan) & (tracking_shrink['nose_x_cm']!=np.nan)]
    tracking_shrink.to_csv('tracking_shrink.csv')
    print("complete script")
