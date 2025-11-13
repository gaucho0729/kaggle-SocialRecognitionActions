import pandas as pd
import ast
import json
import os

INPUT_TRAIN_FILE       = "../train.csv"
TRAIN_ANNOTATION_DIR   = "../train_annotation/"
TRAIN_TRACKING_DIR     = "../train_tracking/"

INPUT_TEST_FILE        = "../test.csv"
TEST_TRACKING_DIR      = "../test_tracking/"

OUTPUT_FILE = "annon_output.csv"

train = pd.read_csv(INPUT_TRAIN_FILE)

pd.set_option('display.max_columns', train.columns.size)
pd.set_option('display.max_rows', len(train))

output = pd.DataFrame({
    'lab_id'   : [],
    'video_id' : [],
    'action'   : [],
    'duration' : [],
    'agent_id' : [],
    'target_id': [],
})

for j in range(len(train)):
    tr = train.iloc[j]
    annon_file = TRAIN_ANNOTATION_DIR + str(tr["lab_id"]) + "/" + str(tr["video_id"]) + ".parquet"
    if os.path.isdir(TRAIN_ANNOTATION_DIR + str(tr["lab_id"]))==False:
        continue
    if os.path.exists(TRAIN_ANNOTATION_DIR + str(tr["lab_id"]) + "/" + str(tr["video_id"]) + ".parquet") == False:
        continue
    annon = pd.read_parquet(annon_file)
    df = pd.DataFrame({
        'lab_id'   : [],
        'video_id' : [],
        'action'   : [],
        'duration' : [],
        'agent_id' : [],
        'target_id': [],
    })

    df['lab_id']    = [str(tr['lab_id'])]   * len(annon)
    df['video_id']  = [str(tr['video_id'])] * len(annon)
    df['action']    = annon['action']
    df['duration']  = (annon['stop_frame'] - annon['start_frame']) / tr['frames_per_second']
    df['agent_id']  = annon['agent_id']
    df['target_id'] = annon['target_id']

    output = pd.concat([output, df])

output.to_csv(OUTPUT_FILE)

# action ごとの duration の統計量（全体）
stats_action = output.groupby("action")["duration"].describe()

stats_filename_action = output.groupby(["lab_id", "video_id", "action"])["duration"].describe()

print("=== filename × action ごとの duration 統計量 ===")
print(stats_filename_action)

print("\n=== action ごとの duration 統計量（全体）===")
print(stats_action)

agg_stats = output.groupby(["lab_id", "video_id", "action"])["duration"].agg(
    ["count", "mean", "median", "std", "min", "max"]
)
agg_stats.to_csv('stat_by_filename.csv')

agg_stats_action = output.groupby("action")["duration"].agg(
    ["count", "mean", "median", "std", "min", "max"]
)
agg_stats_action.to_csv('stat.csv')
