import pandas as pd
import numpy as np
import os
import concurrent.futures

INPUT_TRAIN_FILE  = "../train.csv"
INPUT_FILE        = "train_tracking-2.csv"
TRAIN_ANNOTATION_DIR = "../train_annotation/"
OUTPUT_FILE       = "stat_train_tracking-2.csv"

INTERVAL = 0.1
TEMP_DIR = "./temp_results"  # 一時ファイル保存用ディレクトリ
os.makedirs(TEMP_DIR, exist_ok=True)

def process_video(i, train_row, tracking, bodyparts):
    lab_id   = str(train_row['lab_id'])
    video_id = str(train_row['video_id'])
    video_id_float = float(train_row['video_id'])

    if not os.path.isdir(TRAIN_ANNOTATION_DIR + lab_id):
        return None
    anno_path = os.path.join(TRAIN_ANNOTATION_DIR, lab_id, f"{video_id}.parquet")
    if not os.path.exists(anno_path):
        return None

    anno_file = pd.read_parquet(anno_path)
    results = []

    for _, anno in anno_file.iterrows():
        start_frame = anno['start_frame']
        stop_frame  = anno['stop_frame']
        agent_id    = int(anno['agent_id'])
        target_id   = int(anno['target_id'])

        for bodypart in bodyparts:
            # agent側
            track_agent = tracking[
                (tracking['lab_id'] == lab_id) &
                (tracking['video_id'] == video_id_float) &
                (tracking['video_frame'] >= start_frame) &
                (tracking['video_frame'] <= stop_frame) &
                (tracking['mouse_id'] == agent_id) &
                (tracking['bodypart'] == bodypart)
            ]
            if len(track_agent) >= 2:
                vector = (track_agent['x'].iloc[-1] - track_agent['x'].iloc[0])**2 + \
                         (track_agent['y'].iloc[-1] - track_agent['y'].iloc[0])
                region_w = track_agent['x'].max() - track_agent['x'].min()
                region_h = track_agent['y'].max() - track_agent['y'].min()
                results.append([
                    lab_id, video_id, agent_id, True, bodypart,
                    stop_frame - start_frame, vector, region_w, region_h
                ])

            # target側
            if agent_id != target_id:
                track_target = tracking[
                    (tracking['lab_id'] == lab_id) &
                    (tracking['video_id'] == video_id_float) &
                    (tracking['video_frame'] >= start_frame) &
                    (tracking['video_frame'] <= stop_frame) &
                    (tracking['mouse_id'] == target_id) &
                    (tracking['bodypart'] == bodypart)
                ]
                if len(track_target) >= 2:
                    vector = (track_target['x'].iloc[-1] - track_target['x'].iloc[0])**2 + \
                             (track_target['y'].iloc[-1] - track_target['y'].iloc[0])
                    region_w = track_target['x'].max() - track_target['x'].min()
                    region_h = track_target['y'].max() - track_target['y'].min()
                    results.append([
                        lab_id, video_id, target_id, False, bodypart,
                        stop_frame - start_frame, vector, region_w, region_h
                    ])

    if results:
        df = pd.DataFrame(results, columns=[
            'lab_id', 'video_id', 'mouse_id', 'agent_target', 'bodypart',
            'duration', 'vector', 'region_w', 'region_h'
        ])
        tmp_file = os.path.join(TEMP_DIR, f"part_{i}.csv")
        df.to_csv(tmp_file, index=False)
        return tmp_file
    return None


def main():
    train_data = pd.read_csv(INPUT_TRAIN_FILE)
    tracking = pd.read_csv(INPUT_FILE)
    bodyparts = tracking['bodypart'].unique()

    print(f"Processing {len(train_data)} videos in parallel...")

    temp_files = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []
        for i in range(len(train_data)):
            futures.append(executor.submit(process_video, i, train_data.iloc[i], tracking, bodyparts))

        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            res = future.result()
            if res:
                temp_files.append(res)
            print(f"[{i+1}/{len(train_data)}] done")

    # 結合処理
    print("Combining temporary results...")
    all_data = pd.concat([pd.read_csv(f) for f in temp_files if os.path.exists(f)], ignore_index=True)
    all_data.to_csv(OUTPUT_FILE, index=False)

    print(f"✅ Output saved to: {OUTPUT_FILE}")
    print("Cleaning up...")
    for f in temp_files:
        os.remove(f)

if __name__ == "__main__":
    main()
