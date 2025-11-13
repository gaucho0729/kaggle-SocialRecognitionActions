import pandas as pd
import numpy as np
import os

# action毎のベクトルと矩形範囲を求める


INPUT_TRAIN_FILE       = "../train.csv"
TRAIN_ANNOTATION_DIR   = "../train_annotation/"
TRAIN_TRACKING_DIR     = "../train_tracking/"

INPUT_TEST_FILE        = "../test.csv"
TEST_TRACKING_DIR      = "../test_tracking/"

OUTPUT_FILE = "annotation.csv"

INTERVAL = 0.25

row_threshold = 10000

def initDataFrame():
    retval = pd.DataFrame({
        'lab_id'      : [],
        'video_id'    : [],
        'action'      : [],
        'agent_target': [],
        'mouse_id'    : [],
        'bodypart'    : [],
        'duration'    : [], 
        'vector'      : [],
        'region_w'    : [],
        'region_h'    : [],
    })
    return retval

output = initDataFrame()
output.to_csv(OUTPUT_FILE, mode='w')

test_data = pd.read_csv(INPUT_TEST_FILE)
test_row = test_data.iloc[0]
tracking_method = test_row['tracking_method']

train_data = pd.read_csv(INPUT_TRAIN_FILE)
for i in range(len(train_data)):
    print(i, "/", len(train_data))

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

    tracking_data = pd.read_parquet(train_tracking_file)
    anno_data     = pd.read_parquet(train_anno_file)

    fps        = train['frames_per_second']
    pix_per_cm = train['pix_per_cm_approx']

    tracking_data['video_time'] = tracking_data['video_frame'] / fps
    tracking_data['x_cm']       = tracking_data['x'] / pix_per_cm
    tracking_data['y_cm']       = tracking_data['y'] / pix_per_cm

    # INTERVAL間引きを行う
    start_time = tracking_data['video_time'].min()
    end_time   = tracking_data['video_time'].max()
    target_times = np.arange(start_time, end_time + 1e-8, INTERVAL)  # end を含めたい場合
    # merge_asof を使う場合はキーでソートが必要
    tmp_sorted = tracking_data.sort_values('video_time').reset_index(drop=True)
    target_df = pd.DataFrame({'video_time': target_times})

    # direction='nearest' で「最も近い」行を結合
    train_data_2 = pd.merge_asof(target_df, tmp_sorted, on='video_time', direction='nearest')

    for j in range(len(anno_data)):
        anno = anno_data.iloc[j]
        action_start = anno['start_frame']
        action_stop  = anno['stop_frame']
        fltrd_trk    = train_data_2[(train_data_2['video_frame'] >= action_start) &
                                    (train_data_2['video_frame'] <= action_stop)
                                    ]
        if len(fltrd_trk) == 0:
            continue
        
        act = anno['action']
        trk_act = fltrd_trk
        if len(trk_act) == 0:
            continue
        bodyparts = trk_act['bodypart'].unique()
        for bp in bodyparts:
            trk_bp = trk_act[(trk_act['bodypart'] == bp) &
                             (trk_act['mouse_id'] == anno['agent_id'])
                            ]
            if len(trk_bp) == 0:
                continue
            end = len(trk_bp) - 1
            vector   = (trk_bp['x_cm'].iloc[0] - trk_bp['x_cm'].iloc[end]) ** 2 + (trk_bp['x_cm'].iloc[0] - trk_bp['y_cm'].iloc[end]) ** 2
            region_w =  trk_bp['x_cm'].max()   - trk_bp['x_cm'].min()
            region_h =  trk_bp['y_cm'].max()   - trk_bp['y_cm'].min()
            tmp = pd.DataFrame({
                        'lab_id'      : [lab_id],
                        'video_id'    : [video_id],
                        'action'      : [act],
                        'agent_target': [True],
                        'mouse_id'    : [anno['agent_id']],
                        'bodypart'    : [bp],
                        'duration'    : [trk_bp['video_time'].max() - trk_bp['video_time'].min()], 
                        'vector'      : [vector],
                        'region_w'    : [region_w],
                        'region_h'    : [region_h],
                })
            output = pd.concat([output,tmp], ignore_index=True)
            if anno['agent_id'] == anno['target_id']:
                continue
            trk_bp = trk_act[(trk_act['bodypart'] == bp) &
                             (trk_act['mouse_id'] == anno['target_id'])
                             ]
            if len(trk_bp) == 0:
                continue
            end = len(trk_bp) - 1
            vector   = (trk_bp['x_cm'].iloc[0] - trk_bp['x_cm'].iloc[end]) ** 2 + (trk_bp['x_cm'].iloc[0] - trk_bp['y_cm'].iloc[end]) ** 2
            region_w = trk_bp['x_cm'].max() - trk_bp['x_cm'].min()
            region_h = trk_bp['y_cm'].max() - trk_bp['y_cm'].min()
            tmp = pd.DataFrame({
                        'lab_id'      : [lab_id],
                        'video_id'    : [video_id],
                        'action'      : [act],
                        'agent_target': [False],
                        'mouse_id'    : [anno['target_id']],
                        'bodypart'    : [bp],
                        'duration'    : [trk_bp['video_time'].max() - trk_bp['video_time'].min()], 
                        'vector'      : [vector],
                        'region_w'    : [region_w],
                        'region_h'    : [region_h],
                })
            output = pd.concat([output,tmp], ignore_index=True)

    if len(output) > row_threshold:
        output.to_csv(OUTPUT_FILE, mode='a', header=False)
        output = initDataFrame()

