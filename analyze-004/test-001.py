import pandas as pd
import numpy as np
import ast
import json
import os

# testに使用される動画を0.1sec毎のデータに間引きする

INPUT_TRAIN_FILE       = "../train.csv"
TRAIN_ANNOTATION_DIR   = "../train_annotation/"
TRAIN_TRACKING_DIR     = "../train_tracking/"

INPUT_TEST_FILE        = "../test.csv"
TEST_TRACKING_DIR      = "../test_tracking/"

OUTPUT_FILE = "test_tracking.csv"

INTERVAL = 0.1

row_threshold = 10000

def initDataFrame():
    retval = pd.DataFrame({
        'video_frame': [],
        'video_time' : [],
        'mouse_id'   : [],
        'bodypart'   : [],
        'x'          : [],
        'y'          : [],
    })
    return retval

train = pd.read_csv(INPUT_TRAIN_FILE)

test_data  = pd.read_csv(INPUT_TEST_FILE)
lab_id     = str(test_data['lab_id'].iloc[0])
video_id   = str(test_data['video_id'].iloc[0])
fps        = test_data['frames_per_second'].iloc[0]
pix_per_cm = test_data['pix_per_cm_approx'].iloc[0]
test_tracking_file = TEST_TRACKING_DIR + lab_id + "/" + video_id + ".parquet"

test_tracking_data = pd.read_parquet(test_tracking_file)
test_tracking_data['video_time'] = test_tracking_data['video_frame'] / fps
test_tracking_data['x_cm']       = test_tracking_data['x'] / pix_per_cm
test_tracking_data['y_cm']       = test_tracking_data['y'] / pix_per_cm

mice = test_tracking_data['mouse_id'].unique()
bodyparts = test_tracking_data['bodypart'].unique()
output = initDataFrame()
output.to_csv(OUTPUT_FILE, mode='w')
for mouse in mice:
    tmp = test_tracking_data[test_tracking_data['mouse_id']==mouse]
    start_time = tmp['video_time'].min()
    end_time   = tmp['video_time'].max()
    target_times = np.arange(start_time, end_time + 1e-8, INTERVAL)  # end を含めたい場合

    print('len(target_times):', len(target_times))

    # merge_asof を使う場合はキーでソートが必要
    tmp_sorted = tmp.sort_values('video_time').reset_index(drop=True)
    target_df = pd.DataFrame({'video_time': target_times})

    # direction='nearest' で「最も近い」行を結合
    result = pd.merge_asof(target_df, tmp_sorted, on='video_time', direction='nearest')

    for bpart in bodyparts:
        tmp2 = result[result['bodypart']==bpart]
        df = pd.DataFrame({
            'video_frame': tmp2['video_frame'],
            'video_time' : tmp2['video_time'],
            'mouse_id'   : tmp2['mouse_id'],
            'bodypart'   : tmp2['bodypart'],
            'x'          : tmp2['x'],
            'y'          : tmp2['y'],
            'x_cm'       : tmp2['x_cm'],
            'y_cm'       : tmp2['y_cm'],
        })
        output = pd.concat([output, df], ignore_index=True)
        if len(output)>=row_threshold:
            output.to_csv(OUTPUT_FILE, mode='a', header=False)
            output = initDataFrame()
        
if len(output)>0:
    output.to_csv(OUTPUT_FILE, mode='a', header=False)

