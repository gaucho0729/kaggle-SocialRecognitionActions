import pandas as pd
import numpy as np
import ast
import json
import os

# testの動画を正規化する

INPUT_TRAIN_FILE       = "../test.csv"
TRAIN_ANNOTATION_DIR   = "../train_annotation/"
TRAIN_TRACKING_DIR     = "../train_tracking/"

INPUT_TEST_FILE        = "../test.csv"
TEST_TRACKING_DIR      = "../test_tracking/"

OUTPUT_FILE = "test_normalize_tracking.csv"

INTERVAL = 0.25

row_threshold = 10000

def reduce_rows(src_data, start_time, end_time, interval):
    target_times = np.arange(start_time, end_time + 1e-8, interval)  # end を含めたい場合

    # merge_asof を使う場合はキーでソートが必要
    sorted = src_data.sort_values('video_time').reset_index(drop=True)
    target_df = pd.DataFrame({'video_time': target_times})

    # direction='nearest' で「最も近い」行を結合
    result = pd.merge_asof(target_df, sorted, on='video_time', direction='nearest')
    return result


# trainデータを読み込む
train_data = pd.read_csv(INPUT_TRAIN_FILE)
train = train_data.iloc[0]

lab_id   = str(train['lab_id'])
video_id = str(train['video_id'])
fps      = train['frames_per_second']
pps      = train['pix_per_cm_approx']

# testデータを読み込む
test_tracking_file = TEST_TRACKING_DIR + train['lab_id'] + "/" + video_id + ".parquet"
test_tracking      = pd.read_parquet(test_tracking_file)

print('fps:', fps)
print(len(test_tracking))

# frame⇨sec変換する
test_tracking['video_time'] = test_tracking['video_frame'] / fps

# px→cm変換する
test_tracking['x_cm'] = test_tracking['x'] / pps
test_tracking['y_cm'] = test_tracking['y'] / pps

# trackingを間引く
start_time = test_tracking['video_time'].min()
end_time   = test_tracking['video_time'].max()

bodyparts = test_tracking['bodypart'].unique()
mice      = test_tracking['mouse_id'].unique()

output = pd.DataFrame()

for mouse in mice:
    for bp in bodyparts:
        tmp = test_tracking[(test_tracking['mouse_id']==mouse) &
                            (test_tracking['bodypart']==bp)
                            ]
        result = reduce_rows(tmp, start_time, end_time, INTERVAL)
        # vx/vyを求める
        result['vx_cm'] = result['x_cm'].diff()
        result['vy_cm'] = result['y_cm'].diff()
        output = pd.concat([output, result], ignore_index=True)

output = output.dropna(subset=['video_frame'])
output.to_csv(OUTPUT_FILE)

