import pandas as pd
import numpy as np
import ast
import json
import os

# testのtracking_methodと同じtrainの動画を0.25sec毎のデータに間引きする

INPUT_TRAIN_FILE       = "../train.csv"
TRAIN_ANNOTATION_DIR   = "../train_annotation/"
TRAIN_TRACKING_DIR     = "../train_tracking/"

INPUT_TEST_FILE        = "../test.csv"
TEST_TRACKING_DIR      = "../test_tracking/"

OUTPUT_FILE = "train_tracking_action.csv"

INTERVAL = 0.25

row_threshold = 10000

def initDataFrame():
    retval = pd.DataFrame({
        'lab_id'      : [],
        'video_id'    : [],
        'video_frame' : [],
        'video_time'  : [],
        'agent_target': [],
        'action'      : [],
        'mouse_id'    : [],
        'bodypart'    : [],
        'x'           : [],
        'y'           : [],
        'x_cm'        : [],
        'y_cm'        : [],
    })
    return retval

output = initDataFrame()
output.to_csv(OUTPUT_FILE, mode='w')

train_data = pd.read_csv(INPUT_TRAIN_FILE)

test_data = pd.read_csv(INPUT_TEST_FILE)
test_row = test_data.iloc[0]
tracking_method = test_row['tracking_method']

for i in range(len(train_data)):
    print(i, "/", len(train_data))

    # testと同じtracking_methodのみを通す
    train = train_data.iloc[i]
    if train['tracking_method'] != tracking_method:
        continue
    lab_id   = str(train['lab_id'])
    video_id = str(train['video_id'])
    fps      = train['frames_per_second']
    pix_per_cm_approx = train['pix_per_cm_approx']
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

    # tracking読み込み
    train_tracking_data = pd.read_parquet(train_tracking_file)
    train_tracking_data['video_time'] = train_tracking_data['video_frame'] / fps
    train_tracking_data['x_cm']       = train_tracking_data['x'] / pix_per_cm_approx
    train_tracking_data['y_cm']       = train_tracking_data['y'] / pix_per_cm_approx

    # annotation読み込み
    train_anno_data = pd.read_parquet(train_anno_file)
    train_anno_data['start_time'] = train_anno_data['start_frame'] / fps
    train_anno_data['stop_time']  = train_anno_data['start_frame'] / fps


    # bodyparts毎に読み込む
    bodyparts = train_tracking_data['bodypart'].unique()
    for j in range(len(train_anno_data)):
        anno = train_anno_data.iloc[j]
        start_time = anno['start_time']
        end_time   = anno['stop_time']
        target_times = np.arange(start_time, end_time + 1e-8, INTERVAL)  # end を含めたい場合
        tmp = train_tracking_data[train_tracking_data['mouse_id']==anno['agent_id']]
        if len(tmp) > 0:

            # merge_asof を使う場合はキーでソートが必要
            tmp_sorted = tmp.sort_values('video_time').reset_index(drop=True)
            target_df = pd.DataFrame({'video_time': target_times})

            # direction='nearest' で「最も近い」行を結合
            result = pd.merge_asof(target_df, tmp_sorted, on='video_time', direction='nearest')

            for bpart in bodyparts:
                tmp2 = result[result['bodypart']==bpart]
                df = pd.DataFrame({
                    'lab_id'      : [lab_id]   * len(tmp2),
                    'video_id'    : [video_id] * len(tmp2),
                    'video_frame' : tmp2['video_frame'],
                    'video_time'  : tmp2['video_time'],
                    'agent_target': [True]    * len(tmp2),
                    'action'      : anno['action'] * len(tmp2),
                    'mouse_id'    : tmp2['mouse_id'],
                    'bodypart'    : tmp2['bodypart'],
                    'x'           : tmp2['x'],
                    'y'           : tmp2['y'],
                    'x_cm'        : tmp2['x_cm'],
                    'y_cm'        : tmp2['y_cm'],
                })
                output = pd.concat([output, df], ignore_index=True)
                if len(output)>=row_threshold:
                    output.to_csv(OUTPUT_FILE, mode='a', header=False)
                    output = initDataFrame()
        if anno['agent_id'] != anno['target_id']:
            continue
        tmp = train_tracking_data[train_tracking_data['mouse_id']==anno['target_id']]
        if len(tmp) > 0:
            # merge_asof を使う場合はキーでソートが必要
            tmp_sorted = tmp.sort_values('video_time').reset_index(drop=True)
            target_df = pd.DataFrame({'video_time': target_times})

            # direction='nearest' で「最も近い」行を結合
            result = pd.merge_asof(target_df, tmp_sorted, on='video_time', direction='nearest')

            for bpart in bodyparts:
                tmp2 = result[result['bodypart']==bpart]
                df = pd.DataFrame({
                    'lab_id'      : [lab_id]   * len(tmp2),
                    'video_id'    : [video_id] * len(tmp2),
                    'video_frame' : tmp2['video_frame'],
                    'video_time'  : tmp2['video_time'],
                    'agent_target': [False]    * len(tmp2),
                    'action'      : anno['action'] * len(tmp2),
                    'mouse_id'    : tmp2['mouse_id'],
                    'bodypart'    : tmp2['bodypart'],
                    'x'           : tmp2['x'],
                    'y'           : tmp2['y'],
                    'x_cm'        : tmp2['x_cm'],
                    'y_cm'        : tmp2['y_cm'],
                })
                output = pd.concat([output, df], ignore_index=True)
                if len(output)>=row_threshold:
                    output.to_csv(OUTPUT_FILE, mode='a', header=False)
                    output = initDataFrame()
        
if len(output)>0:
    output.to_csv(OUTPUT_FILE, mode='a', header=False)

