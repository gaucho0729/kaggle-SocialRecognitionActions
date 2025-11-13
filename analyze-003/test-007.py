import pandas as pd
import numpy as np
import ast
import json
import os

# testのtracking_methodと同じtrainの動画を0.1sec毎のデータに間引きする

INPUT_FILE = "train_tracking_tm.csv"

INPUT_TRAIN_FILE       = "../train.csv"
TRAIN_ANNOTATION_DIR   = "../train_annotation/"
TRAIN_TRACKING_DIR     = "../train_tracking/"

INPUT_TEST_FILE        = "../test.csv"
TEST_TRACKING_DIR      = "../test_tracking/"

OUTPUT_FILE = "train_tracking-2.csv"

INTERVAL = 0.1

row_threshold = 10000

train_data = pd.read_csv(INPUT_TRAIN_FILE)

test_data = pd.read_csv(INPUT_TEST_FILE)
test_row = test_data.iloc[0]
tracking_method = test_row['tracking_method']

written = False
annotations = pd.DataFrame()

def initDataFrame():
    retval = pd.DataFrame({
        'lab_id'     : [],
        'video_id'   : [],
        'video_frame': [],
        'video_time' : [],
        'mouse_id'   : [],
        'bodypart'   : [],
        'x'          : [],
        'y'          : []
    })
    return retval
'''
tracking = pd.read_csv('train_tracking_all.csv',
                       index_col=False,
                       dtype={
                            "Unnamed: 0" : "int64",
                            "lab_id"     : "string",
                            "video_id"   : "int64",
                            "video_frame": "float64",
                            "video_time" : "float64",
                            "mouse_id"   : "float64",
                            "bodypart"   : "string",
                            "x"          : "float64",
                            "y"          : "float64",
                        }
                    )
'''
tracking = pd.read_csv(INPUT_FILE, index_col=False)

output = initDataFrame()
output.to_csv(OUTPUT_FILE, mode="w")

for i in range(len(train_data)):
    print(i, "/", len(train_data))

    # testと同じtracking_methodのみを通す
    train = train_data.iloc[i]

    lab_id     = str(train['lab_id'])
    video_id   = str(train['video_id'])    
    video_id_float= float(train['video_id'])    

    # annotationを読む
    if os.path.isdir(TRAIN_ANNOTATION_DIR + lab_id)==False:
        continue
    if os.path.exists(TRAIN_ANNOTATION_DIR + lab_id + "/" + video_id + ".parquet") == False:
        continue
    train_anno_file = TRAIN_ANNOTATION_DIR + lab_id + "/" + video_id + ".parquet"
    train_anno = pd.read_parquet(train_anno_file)

    for i in range(len(train_anno)):
        anno = train_anno.iloc[i]
        start_frame = anno['start_frame']
        stop_frame  = anno['stop_frame']
        tmp = tracking[(tracking['lab_id'  ]     == lab_id)         &
                       (tracking['video_id']     == video_id_float) &
                       (tracking['video_frame']  >= start_frame)    &
                       (tracking['video_frame']  <= stop_frame)
        ]
        tmp.to_csv(OUTPUT_FILE, mode="a",  header=False)

if len(output) > 0:
    output.to_csv(OUTPUT_FILE, mode="a",  header=False)
