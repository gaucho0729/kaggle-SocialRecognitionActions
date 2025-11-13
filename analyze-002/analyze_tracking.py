import pandas as pd
import numpy as np
import ast
import json
import os

INPUT_TRAIN_FILE       = "../train.csv"
TRAIN_ANNOTATION_DIR   = "../train_annotation/"
TRAIN_TRACKING_DIR     = "../train_tracking/"

INPUT_TEST_FILE        = "../test.csv"
TEST_TRACKING_DIR      = "../test_tracking/"

OUTPUT_FILE = "analyze_tracking.csv"

train = pd.read_csv(INPUT_TRAIN_FILE)

pd.set_option('display.max_columns', train.columns.size)
pd.set_option('display.max_rows', len(train))

output = pd.DataFrame({
    'lab_id'         : [],
    'video_id'       : [],
    'mouse_id'       : [],
    'bodypart'       : [],
    'start_frame'    : [],
    'end_frame'      : [],
    'tracking_method': [],
})

output.to_csv(OUTPUT_FILE, mode='w')

for j in range(len(train)):
    print(j , '/', len(train))
    tr = train.iloc[j]
    lab_id   = str(tr['lab_id'])
    video_id = str(tr['video_id'])
    track_file = TRAIN_TRACKING_DIR + lab_id + "/" + video_id + ".parquet"
    tracking_method = tr['tracking_method']
    if os.path.isdir(TRAIN_TRACKING_DIR + lab_id)==False:
        continue
    if os.path.exists(TRAIN_TRACKING_DIR + lab_id + "/" + video_id + ".parquet") == False:
        continue
    track = pd.read_parquet(track_file)
    for m in track['mouse_id'].unique():
        mouse_id = m
        for bp in track['bodypart'].unique():
            t = track[(track['mouse_id']==mouse_id) &
                      (track['bodypart']==bp)
                  ]
            if len(t)>=1:
                start_frame = t['video_frame'].min()
                end_frame   = t['video_frame'].max()
                df = pd.DataFrame({
                    'lab_id'         : [lab_id],
                    'video_id'       : [video_id],
                    'mouse_id'       : [mouse_id],
                    'bodypart'       : [bp],
                    'start_frame'    : [start_frame],
                    'end_frame'      : [end_frame],
                    'tracking_method': [tracking_method],
                })
                df.to_csv(OUTPUT_FILE, mode='a', header=False)

