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

OUTPUT_FILE = "action_output.csv"


train = pd.read_csv(INPUT_TRAIN_FILE)

pd.set_option('display.max_columns', train.columns.size)
pd.set_option('display.max_rows', len(train))

output = pd.DataFrame({
    'lab_id'          : [],
    'video_id'        : [],
    'action'          : [],
    'duration'        : [],
    'agent_id'        : [],
    'agent_vector'    : [],
    'agent_region_w'  : [],
    'agent_region_h'  : [],
    'target_id'       : [],
    'target_vector'   : [],
    'target_region_w' : [],
    'target_region_h' : [],
    'tracking_method' : [],
})

for j in range(len(train)):
    tr = train.iloc[j]
    lab_id   = str(tr['lab_id'])
    video_id = str(tr['video_id'])
    annon_file = TRAIN_ANNOTATION_DIR + lab_id + "/" + video_id + ".parquet"
    if os.path.isdir(TRAIN_ANNOTATION_DIR + lab_id)==False:
        continue
    if os.path.exists(TRAIN_ANNOTATION_DIR + lab_id + "/" + video_id + ".parquet") == False:
        continue
    annon = pd.read_parquet(annon_file)
    ''': train_annotation/*/*.parquet
    #   Column       Non-Null Count  Dtype 
    ---  ------       --------------  ----- 
    0   agent_id     342 non-null    int8  
    1   target_id    342 non-null    int8  
    2   action       342 non-null    object
    3   start_frame  342 non-null    int16 
    4   stop_frame   342 non-null    int16 
    '''
    if os.path.isdir(TRAIN_TRACKING_DIR + lab_id)==False:
        continue
    if os.path.exists(TRAIN_TRACKING_DIR + lab_id + "/" + video_id + ".parquet") == False:
        continue
    tracking_file = TRAIN_TRACKING_DIR + lab_id + "/" + video_id + ".parquet"
    trackings = pd.read_parquet(tracking_file)
    tracking_method = tr['tracking_method']

    ''': train_tracking/*/*.parquet
    #   Column       Non-Null Count    Dtype  
    ---  ------       --------------    -----  
    0   video_frame  1087658 non-null  int16  
    1   mouse_id     1087658 non-null  int8   
    2   bodypart     1087658 non-null  object 
    3   x            1087658 non-null  float32
    4   y            1087658 non-null  float32
    '''
    for i in range(len(annon)):
        ln = annon.iloc[i]

        agent_id    = ln['agent_id']
        target_id   = ln['target_id']
        start_frame = ln['start_frame']
        stop_frame  = ln['stop_frame']
        action      = ln['action']
        duration    = (stop_frame - start_frame) / tr['frames_per_second']

        t = trackings[(trackings['video_frame']>= start_frame) & 
                      (trackings['video_frame']<= stop_frame)  &
                      (trackings['mouse_id']   == agent_id)
            ]
        if len(t) == 0:
            continue
        start = t.iloc[0]
        end   = t.iloc[len(t)-1]
        agent_region_w  = t['x'].max() - t['x'].min()
        agent_region_h  = t['y'].max() - t['y'].min()
        agent_vector    = (start['x'] - end['x']) ** 2 + (start['y'] - end['y']) ** 2
        target_region_w = 0
        target_region_h = 0
        target_vector   = 0
        if agent_id != target_id:
            t = trackings[(trackings['video_frame']>= start_frame) & 
                          (trackings['video_frame']<= stop_frame)  &
                          (trackings['mouse_id']   == target_id)
                ]
            if len(t) >= 1:
                start = t.iloc[0]
                end   = t.iloc[len(t)-1]
                target_region_w = (t['x'].max() - t['x'].min()) / tr['pix_per_cm_approx']
                target_region_h = (t['y'].max() - t['y'].min()) / tr['pix_per_cm_approx']
                target_vector   = np.sqrt((start['x'] - end['x']) ** 2 / tr['pix_per_cm_approx'] ** 2 + (start['y'] - end['y']) ** 2 / tr['pix_per_cm_approx'] ** 2)
        df = pd.DataFrame({
            'lab_id'          : [lab_id],
            'video_id'        : [video_id],
            'action'          : [action],
            'duration'        : [duration],
            'agent_id'        : [agent_id],
            'agent_vector'    : [agent_vector],
            'agent_region_w'  : [agent_region_w],
            'agent_region_h'  : [agent_region_h],
            'target_id'       : [target_id],
            'target_vector'   : [target_vector],
            'target_region_w' : [target_region_w],
            'target_region_h' : [target_region_h],
            'tracking_method' : [tracking_method],
        })
        output = pd.concat([output, df], ignore_index=True)


output.to_csv(OUTPUT_FILE)