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

OUTPUT_FILE = "valid_action_tracking.csv"

# 出力形式
#   lab_id                 from train
#   video_id               from train
#   mouse_id               from tracking
#   agent/target           BOOL
#   start_frame_action     from annotation
#   stop_frame_action      from annotation
#   action                 from annotation
#   action_duration        from annotation
#   bodypart               from train/tracking
#   tracking_method        from train
#   start_frame_tracking   from tracking
#   end_frame_tracking     from tracking

dmmy_out = pd.DataFrame({
    'lab_id'               : [],
    'video_id'             : [],
    'mouse_id'             : [],
    'agent/target'         : [],
    'start_frame_action'   : [],
    'stop_frame_action'    : [],
    'action'               : [],
    'action_duration'      : [],
    'bodypart'             : [],
    'tracking_method'      : [],
    'start_frame_tracking' : [],
    'end_frame_tracking'   : [],
})
dmmy_out.to_csv(OUTPUT_FILE, mode='w')

train = pd.read_csv(INPUT_TRAIN_FILE)
for j in range(len(train)):
    print(j , '/', len(train))
    trn = train.iloc[j]
    lab_id   = str(trn['lab_id'])
    video_id = str(trn['video_id'])
    track_file = TRAIN_TRACKING_DIR + lab_id + "/" + video_id + ".parquet"
    tracking_method = trn['tracking_method']
    if os.path.isdir(TRAIN_TRACKING_DIR + lab_id)==False:
        continue
    if os.path.exists(TRAIN_TRACKING_DIR + lab_id + "/" + video_id + ".parquet") == False:
        continue
    track = pd.read_parquet(track_file)

    annon_file = TRAIN_ANNOTATION_DIR + lab_id + "/" + video_id + ".parquet"
    if os.path.isdir(TRAIN_ANNOTATION_DIR + lab_id)==False:
        continue
    if os.path.exists(TRAIN_ANNOTATION_DIR + lab_id + "/" + video_id + ".parquet") == False:
        continue
    annon = pd.read_parquet(annon_file)

    for i in range(len(annon)):
        ann = annon.iloc[i]
        start_frame_action = ann['start_frame']
        stop_frame_action  = ann['stop_frame']   
        action             = ann['action']
        action_duration    = (stop_frame_action - start_frame_action) / trn['frames_per_second']
        tracking_method    = trn['tracking_method']
        trk_agent          = track[track['mouse_id'] == ann['agent_id']]
        trk_target         = track[track['mouse_id'] == ann['target_id']]
        bodyparts          = json.loads(trn['body_parts_tracked'].replace('""', '"'))
        for bp in bodyparts:
            if len(trk_agent) > 0:
                t_a = trk_agent[trk_agent[ 'bodypart']==bp]
                if len(t_a)>0:
                    start_frame_tracking = t_a['video_frame'].min()
                    end_frame_tracking   = t_a['video_frame'].max()
                    df = pd.DataFrame({
                        'lab_id'               : [lab_id],
                        'video_id'             : [video_id],
                        'mouse_id'             : [ann['agent_id']],
                        'agent/target'         : [True],
                        'start_frame_action'   : [start_frame_action],
                        'stop_frame_action'    : [stop_frame_action],
                        'action'               : [action],
                        'action_duration'      : [action_duration],
                        'bodypart'             : [bp],
                        'tracking_method'      : [tracking_method],
                        'start_frame_tracking' : [start_frame_tracking],
                        'end_frame_tracking'   : [end_frame_tracking],
                    })
                    df.to_csv(OUTPUT_FILE, mode='a', header=False)

            if len(trk_agent) > 0:
                t_t = trk_agent[trk_agent['bodypart']==bp]
                if (len(t_t)>0) & (ann['agent_id'] != ann['target_id']):
                    start_frame_tracking = t_t['video_frame'].min()
                    end_frame_tracking   = t_t['video_frame'].max()
                    df = pd.DataFrame({
                        'lab_id'               : [lab_id],
                        'video_id'             : [video_id],
                        'mouse_id'             : [ann['target_id']],
                        'agent/target'         : [False],
                        'start_frame_action'   : [start_frame_action],
                        'stop_frame_action'    : [stop_frame_action],
                        'action'               : [action],
                        'action_duration'      : [action_duration],
                        'bodypart'             : [bp],
                        'tracking_method'      : [tracking_method],
                        'start_frame_tracking' : [start_frame_tracking],
                        'end_frame_tracking'   : [end_frame_tracking],
                    })
                    df.to_csv(OUTPUT_FILE, mode='a', header=False)

