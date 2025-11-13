import pandas as pd
import numpy as np
import ast
import json
import os

# testのtracking_methodと同じtrainのannotationをcsvで出力する

INPUT_TRAIN_FILE       = "../train.csv"
TRAIN_ANNOTATION_DIR   = "../train_annotation/"
TRAIN_TRACKING_DIR     = "../train_tracking/"

INPUT_TEST_FILE        = "../test.csv"
TEST_TRACKING_DIR      = "../test_tracking/"

OUTPUT_FILE = "annotation.csv"

INTERVAL = 0.1

row_threshold = 10000

train_data = pd.read_csv(INPUT_TRAIN_FILE)

test_data = pd.read_csv(INPUT_TEST_FILE)
test_row = test_data.iloc[0]
tracking_method = test_row['tracking_method']

written = False
annotations = pd.DataFrame()

for i in range(len(train_data)):
    print(i, "/", len(train_data))

    # testと同じtracking_methodのみを通す
    train = train_data.iloc[i]
    if train['tracking_method'] != tracking_method:
        continue
    lab_id   = str(train['lab_id'])
    video_id = str(train['video_id'])
    fps      = train['frames_per_second']
    if train['lab_id'] != lab_id:
        continue

    # ディレクトリ、ファイル有無確認
    if os.path.isdir(TRAIN_TRACKING_DIR + lab_id)==False:
        continue
    if os.path.exists(TRAIN_TRACKING_DIR + lab_id + "/" + video_id + ".parquet") == False:
        continue
    train_tracking_file = TRAIN_TRACKING_DIR + lab_id + "/" + video_id + ".parquet"

    if os.path.isdir(TRAIN_ANNOTATION_DIR + lab_id)==False:
        continue
    if os.path.exists(TRAIN_ANNOTATION_DIR + lab_id + "/" + video_id + ".parquet") == False:
        continue
    train_anno_file = TRAIN_ANNOTATION_DIR + lab_id + "/" + video_id + ".parquet"

    train_anno = pd.read_parquet(train_anno_file)
    train_anno['start_time'] = train_anno['start_frame'] / fps
    train_anno['stop_time']  = train_anno['stop_frame'] / fps
    train_anno['solo']       = (train_anno['agent_id'] == train_anno['target_id'])

    if written == False:
        train_anno.to_csv(OUTPUT_FILE, mode="w")
        written = True
    else:
        train_anno.to_csv(OUTPUT_FILE, mode="a")
    annotations = pd.concat([annotations, train_anno], ignore_index=True)

actions = annotations['action'].unique()
solo_action=[]
for act in actions:
    tmp = annotations[annotations['action'] == act]
    if len(tmp)>0:
        solo = (tmp[tmp['solo']==True])
        line = act + ":" + str(len(solo)) + "(solo)/" + str(len(tmp))
        print(line)
        solo_action.append(act)

print(solo_action)

for act in actions:
    tmp = annotations[annotations['action'] == act]
    bodyparts = tmp['bodypart'].unique()
    for bp in bodyparts:
        print('act: ' + bp)