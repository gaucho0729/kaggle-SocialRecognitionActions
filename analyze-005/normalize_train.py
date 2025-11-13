import pandas as pd
import numpy as np
import ast
import json
import os

# testのtracking_methodと同じtrainの動画を正規化する

INPUT_TRAIN_FILE       = "../train.csv"
TRAIN_ANNOTATION_DIR   = "../train_annotation/"
TRAIN_TRACKING_DIR     = "../train_tracking/"

INPUT_TEST_FILE        = "../test.csv"
train_tracking_DIR      = "../train_tracking/"

OUTPUT_FILE = "train_normalize_tracking.csv"

INTERVAL = 0.25

row_threshold = 10000

# testのtracking_methodを返す
def get_test_tracking_method():
    test_data = pd.read_csv(INPUT_TEST_FILE)
    test = test_data.iloc[0]
    return test['tracking_method']

# trackingデータを間引く
def reduce_rows(src_data, start_time, end_time, interval):
    target_times = np.arange(start_time, end_time + 1e-8, interval)  # end を含めたい場合

    # merge_asof を使う場合はキーでソートが必要
    sorted = src_data.sort_values('video_time').reset_index(drop=True)
    target_df = pd.DataFrame({'video_time': target_times})

    # direction='nearest' で「最も近い」行を結合
    result = pd.merge_asof(target_df, sorted, on='video_time', direction='nearest')
    return result

def initDataFrame():
    retval = pd.DataFrame({
        'lab_id'      : [],
        'video_id'    : [],
        'video_time'  : [],
        'video_frame' : [],
        'mouse_id'    : [],
        'bodypart'    : [],
        'x'           : [],
        'y'           : [],
        'x_cm'        : [],
        'y_cm'        : [],
        'vx_cm'       : [],
        'vy_cm'       : [],
    })
    return retval

tracking_method = get_test_tracking_method()

# trainデータを読み込む
train_data = pd.read_csv(INPUT_TRAIN_FILE)

header = True
for i in range(len(train_data)):
    print(i, '/', len(train_data))

    train = train_data.iloc[i]

    if tracking_method != train['tracking_method']:
        continue

    lab_id   = str(train['lab_id'])
    video_id = str(train['video_id'])
    fps      = train['frames_per_second']
    pps      = train['pix_per_cm_approx']

    # trainデータを読み込む
    if os.path.isdir(TRAIN_TRACKING_DIR + lab_id)==False:
        continue
    if os.path.exists(TRAIN_TRACKING_DIR + lab_id + "/" + video_id + ".parquet") == False:
        continue
    train_tracking_file = TRAIN_TRACKING_DIR + train['lab_id'] + "/" + video_id + ".parquet"
    train_tracking      = pd.read_parquet(train_tracking_file)

    # frame⇨sec変換する
    train_tracking['video_time'] = train_tracking['video_frame'] / fps

    # px→cm変換する
    train_tracking['x_cm'] = train_tracking['x'] / pps
    train_tracking['y_cm'] = train_tracking['y'] / pps

    # trackingを間引く
    start_time = train_tracking['video_time'].min()
    end_time   = train_tracking['video_time'].max()

    bodyparts = train_tracking['bodypart'].unique()
    mice      = train_tracking['mouse_id'].unique()

    output = pd.DataFrame()

    for mouse in mice:
        for bp in bodyparts:
            tmp = train_tracking[(train_tracking['mouse_id']==mouse) &
                                 (train_tracking['bodypart']==bp)
                                ]
            result = reduce_rows(tmp, start_time, end_time, INTERVAL)
            # vx/vyを求める
            result['vx_cm'] = result['x_cm'].diff()
            result['vy_cm'] = result['y_cm'].diff()
            tmp = pd.DataFrame({
                'lab_id'      : [lab_id]   * len(result),
                'video_id'    : [video_id] * len(result),
                'video_time'  : result['video_time'],
                'video_frame' : result['video_frame'],
                'mouse_id'    : result['mouse_id'],
                'bodypart'    : result['bodypart'],
                'x'           : result['x'],
                'y'           : result['y'],
                'x_cm'        : result['x_cm'],
                'y_cm'        : result['y_cm'],
                'vx_cm'       : result['vx_cm'],
                'vy_cm'       : result['vy_cm']
            })
            output = pd.concat([output, tmp], ignore_index=True)

    output = output.dropna(subset=['video_frame'])
    if header == True:
        output.to_csv(OUTPUT_FILE, mode='w')
        header = False
    else:
        output.to_csv(OUTPUT_FILE, mode='a', header=None)


