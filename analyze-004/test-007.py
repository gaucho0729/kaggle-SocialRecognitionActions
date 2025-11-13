import pandas as pd
import numpy as np
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed

INPUT_FILE = "train_tracking_tm.csv"
INPUT_TRAIN_FILE = "../train.csv"
TRAIN_ANNOTATION_DIR = "../train_annotation/"
TRAIN_TRACKING_DIR = "../train_tracking/"
INPUT_TEST_FILE = "../test.csv"
TEST_TRACKING_DIR = "../test_tracking/"
OUTPUT_FILE = "train_tracking-2.csv"

INTERVAL = 0.25

def initDataFrame():
    return pd.DataFrame({
        'lab_id': [],
        'video_id': [],
        'video_frame': [],
        'video_time': [],
        'mouse_id': [],
        'bodypart': [],
        'x': [],
        'y': []
    })

# グローバル読込
train_data = pd.read_csv(INPUT_TRAIN_FILE)
test_data = pd.read_csv(INPUT_TEST_FILE)
tracking = pd.read_csv(INPUT_FILE, index_col=False)

# test と同じ tracking_method のみ対象
tracking_method = test_data.iloc[0]['tracking_method']

# 並列処理関数
def process_one(i):
    train = train_data.iloc[i]
    lab_id = str(train['lab_id'])
    video_id = str(train['video_id'])
    video_id_float = float(train['video_id'])

    anno_path = os.path.join(TRAIN_ANNOTATION_DIR, lab_id, f"{video_id}.parquet")
    if not os.path.exists(anno_path):
        return None

    train_anno = pd.read_parquet(anno_path)
    tmp_out = []

    for j in range(len(train_anno)):
        anno = train_anno.iloc[j]
        start_frame = anno['start_frame']
        stop_frame = anno['stop_frame']

        tmp = tracking[
            (tracking['lab_id'] == lab_id) &
            (tracking['video_id'] == video_id_float) &
            (tracking['video_frame'] >= start_frame) &
            (tracking['video_frame'] <= stop_frame)
        ]

        if len(tmp) > 0:
            tmp_out.append(tmp)

    if tmp_out:
        result = pd.concat(tmp_out)
        temp_path = tempfile.mktemp(suffix=".csv", prefix=f"tmp_{lab_id}_{video_id}_")
        result.to_csv(temp_path, index=False, header=False)
        return temp_path
    else:
        return None

# 並列実行
temp_files = []
with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
    futures = {executor.submit(process_one, i): i for i in range(len(train_data))}
    for future in as_completed(futures):
        i = futures[future]
        try:
            temp_path = future.result()
            if temp_path:
                temp_files.append(temp_path)
            print(f"Done {i+1}/{len(train_data)}")
        except Exception as e:
            print(f"Error in {i}: {e}")

# 一時ファイルを結合
output_header = list(initDataFrame().columns)
merged_df_list = []

for tmp_file in temp_files:
    df = pd.read_csv(tmp_file, header=None, names=output_header)
    merged_df_list.append(df)
    os.remove(tmp_file)

if merged_df_list:
    merged = pd.concat(merged_df_list, ignore_index=True)
    # ✅ lab_id, video_id, video_frameでソート
    merged = merged.sort_values(by=['lab_id', 'video_id', 'video_frame'], ignore_index=True)
    merged.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Done. Sorted output saved to {OUTPUT_FILE}")
else:
    print("⚠ No data was generated.")
