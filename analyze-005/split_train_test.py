import pandas as pd
import numpy as np
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder

# === 定数設定 ===
INPUT_TRAIN_FILE       = "../train.csv"
TRAIN_ANNOTATION_DIR   = "../train_annotation/"
TRAIN_TRACKING_DIR     = "../train_tracking/"
INPUT_FILE             = "train_normalize_action_tracking-3.csv"
OUTPUT_TRAIN_FILE      = "train-001.csv"
OUTPUT_TEST_FILE       = "test-001.csv"
OUTPUT_TEST_Y_FILE     = "test-y-001.csv"
INTERVAL               = 0.25
TMP_DIR                = "./tmp_results"
LOG_FILE               = "log.txt"

bool_map = {
    "True":  1,
    "False": 0,
}

action_map = {
    'allogroom'       : 0,
    'approach'        : 1,
    'attack'          : 2,
    'attemptmount'    : 3,
    'avoid'           : 4,
    'biteobject'      : 5,
    'chase'           : 6,
    'chaseattack'     : 7,
    'climb'           : 8,
    'defend'          : 9,
    'dig'             : 10,
    'disengage'       : 11,
    'dominance'       : 12,
    'dominancegroom'  : 13,
    'dominancemount'  : 14,
    'ejaculate'       : 15,
    'escape'          : 16,
    'exploreobject'   : 17,
    'flinch'          : 18,
    'follow'          : 19,
    'freeze'          : 20,
    'genitalgroom'    : 21,
    'huddle'          : 22,
    'intromit'        : 23,
    'mount'           : 24,
    'rear'            : 25,
    'reciprocalsniff' : 26,
    'rest'            : 27,
    'run'             : 28,
    'selfgroom'       : 29,
    'shepherd'        : 30,
    'sniff'           : 31,
    'sniffbody'       : 32,
    'sniffface'       : 33,
    'sniffgenital'    : 34,
    'submit'          : 35,
    'tussle'          : 36,
}

all_data = pd.read_csv(INPUT_FILE)
all_data.info()

# 不要な特徴量を削除する
all_data = all_data.drop("Unnamed: 0", axis=1)
all_data = all_data.drop("center_x",   axis=1)
all_data = all_data.drop("center_y",   axis=1)
all_data = all_data.drop("nose_x",     axis=1)
all_data = all_data.drop("nose_y",     axis=1)
#all_data['agent_target'] = all_data['agent_target'].apply(lambda x: bool(int(x)) if str(x).isdigit() else bool(x))

# グループを作成（lab_id, video_id, action の組み合わせ）
all_data["group"] = all_data[["lab_id", "video_id", "mouse_id", "action", "agent_target"]].astype(str).agg("_".join, axis=1)
all_data.to_csv('foo.csv')

groups = all_data["group"]

# lab_idを数値化
le = LabelEncoder()
all_data['lab_id_num'] = le.fit_transform(all_data['lab_id'])
all_data = all_data.drop('lab_id', axis=1)

# actionを数値化
#all_data['action_num'] = all_data['action'].map(action_map)
#all_data = all_data.drop('action', axis=1)

# agent_targetを数値化
all_data['agent_target'] = all_data['agent_target'].map(bool_map)

#y = all_data.pop("action_num")
#X = all_data

y = all_data.pop("action")
X = all_data

# GroupShuffleSplitを使用
gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)

train_idx, test_idx = next(gss.split(X, y, groups=groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = [y[i] for i in train_idx], [y[i] for i in test_idx]

print("unique(y_train)", np.unique(y_train))
print("unique(y_test)", np.unique(y_test))

X_train = X_train.drop("group", axis=1)
X_test  = X_test.drop("group", axis=1)

X_train.to_csv('X_train.csv')
X_test.to_csv( 'X_test.csv')
y_train = pd.DataFrame(y_train)
y_train.to_csv('y_train.csv')
y_test = pd.DataFrame(y_test)
y_test.to_csv( 'y_test.csv')
