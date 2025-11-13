import pandas as pd
import numpy as np
import ast
import json
import os

# testのtracking_methodと同じtrainの動画から0.1sec毎の

INPUT_TRAIN_FILE  = "../train.csv"
INPUT_FILE        = "train_tracking-2.csv"
TRAIN_ANNOTATION_DIR   = "../train_annotation/"
TRAIN_TRACKING_DIR     = "../train_tracking/"

OUTPUT_FILE = "stat_train_tracking-2.csv"

INTERVAL = 0.1

row_threshold = 10000

def initDataFrame():
    retval = pd.DataFrame({
        'lab_id'     : [],
        'video_id'   : [],
#        'video_frame': [],
#        'video_time' : [],
        'mouse_id'   : [],
        'agent_target': [],
        'bodypart'   : [],
        'duration'   : [], 
        'vector'     : [],
        'region_w'   : [],
        'region_h'   : [],
    })
    return retval

train_data = pd.read_csv(INPUT_TRAIN_FILE)
tracking = pd.read_csv(INPUT_FILE)
output = initDataFrame()
output.to_csv(OUTPUT_FILE, mode='w')
bodyparts = tracking['bodypart'].unique()

#print('tracking')
#tracking.info()

for i in range(len(train_data)):
    print(i, "/", len(train_data))

    # testと同じtracking_methodのみを通す
    train = train_data.iloc[i]
    lab_id   = str(train['lab_id'])
    video_id = str(train['video_id'])
    video_id_float= float(train['video_id'])    
    fps      = train['frames_per_second']
    if train['lab_id'] != lab_id:
        continue

    if os.path.isdir(TRAIN_ANNOTATION_DIR + lab_id)==False:
        continue
    if os.path.exists(TRAIN_ANNOTATION_DIR + lab_id + "/" + video_id + ".parquet") == False:
        continue
    train_anno_file = TRAIN_ANNOTATION_DIR + lab_id + "/" + video_id + ".parquet"
    anno_file = pd.read_parquet(train_anno_file)

#    print('anno_file:')
#    anno_file.info()

    for j in range(len(anno_file)):
        anno = anno_file.iloc[j]
        start_frame = anno['start_frame']
        stop_frame  = anno['stop_frame']
        agent_id    = int(anno['agent_id'])
        target_id   = int(anno['target_id'])

        for bodypart in bodyparts:
            trackings = tracking[(tracking['lab_id']      == lab_id)         &
                                 (tracking['video_id']    == video_id_float) &
                                 (tracking['video_frame'] >= start_frame)    &
                                 (tracking['video_frame'] <= stop_frame)     &
                                 (tracking['mouse_id']    == agent_id)       &
                                 (tracking['bodypart']    == bodypart)
                                ]
            if len(trackings) >= 2:
                vector   = (trackings['x'].iloc[len(trackings)-1] -trackings['x'].iloc[0]) ** 2 + (trackings['y'].iloc[len(trackings)-1] -trackings['y'].iloc[0])
                region_w = (trackings['x'].max() - trackings['x'].min())
                region_h = (trackings['y'].max() - trackings['y'].min())
                tmp = pd.DataFrame({
                    'lab_id'      : [lab_id],
                    'video_id'    : [video_id],
    #                'video_frame': [],
    #                'video_time' : [],
                    'mouse_id'    : [agent_id],
                    'agent_target': [True],
                    'bodypart'    : [bodypart],
                    'duration'    : [anno['stop_frame'] - anno['start_frame']], 
                    'vector'      : [vector],
                    'region_w'    : [region_w],
                    'region_h'    : [region_h],
                })
                tmp.to_csv(OUTPUT_FILE, mode='a', header=False)

                if anno['agent_id'] != anno['target_id']:
                    trackings = tracking[(tracking['lab_id']     == lab_id)         &
                                        (tracking['video_id']    == video_id_float) &
                                        (tracking['video_frame'] >= start_frame)    &
                                        (tracking['video_frame'] <= stop_frame)     &
                                        (tracking['mouse_id']    == target_id)      &
                                        (tracking['bodypart']    == bodypart)
                                        ]
                    if len(trackings) >= 2:
                        vector   = (trackings['x'].iloc[len(trackings)-1] -trackings['x'].iloc[0]) ** 2 + (trackings['y'].iloc[len(trackings)-1] -trackings['y'].iloc[0])
                        region_w = (trackings['x'].max() - trackings['x'].min())
                        region_h = (trackings['y'].max() - trackings['y'].min())
                        tmp = pd.DataFrame({
                            'lab_id'      : [lab_id],
                            'video_id'    : [video_id],
            #                'video_frame': [],
            #                'video_time' : [],
                            'mouse_id'    : [target_id],
                            'agent_target': [False],
                            'bodypart'    : [bodypart],
                            'duration'    : [anno['stop_frame'] - anno['start_frame']], 
                            'vector'      : [vector],
                            'region_w'    : [region_w],
                            'region_h'    : [region_h],
                        })

                        tmp.to_csv(OUTPUT_FILE, mode='a', header=False)

#if len(output) >= 0:
#    output.to_csv(OUTPUT_FILE, mode='a', header=False)

