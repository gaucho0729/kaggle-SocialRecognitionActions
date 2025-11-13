import pandas as pd
import numpy as np
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

# === 定数設定 ===
INPUT_TRAIN_FILE       = "../train.csv"
TRAIN_ANNOTATION_DIR   = "../train_annotation/"
TRAIN_TRACKING_DIR     = "../train_tracking/"
INPUT_TEST_FILE        = "../test.csv"
OUTPUT_FILE            = "train_normalize_action_tracking-2.csv"
INTERVAL               = 0.25
TMP_DIR                = "./tmp_results"
LOG_FILE               = "log.txt"

os.makedirs(TMP_DIR, exist_ok=True)


# === ログ関数 ===
def write_log(msg: str):
    """ログをファイルと標準出力の両方に書く"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# === 共通関数 ===
def get_test_tracking_method():
    test_data = pd.read_csv(INPUT_TEST_FILE)
    return test_data.iloc[0]['tracking_method']

def reduce_rows(src_data, start_time, end_time, interval):
    target_times = np.arange(start_time, end_time + 1e-8, interval)
    sorted_df = src_data.sort_values('video_time').reset_index(drop=True)
    target_df = pd.DataFrame({'video_time': target_times})
    return pd.merge_asof(target_df, sorted_df, on='video_time', direction='nearest')

def initDataFrame_2():
    return pd.DataFrame({
        'lab_id': [],
        'video_id': [],
        'video_time': [],
        'video_frame': [],
        'mouse_id': [],
        'agent_target': [],
        'action': [],
        'center_x': [],
        'center_y': [],
        'center_x_cm': [],
        'center_y_cm': [],
        'center_vx_cm': [],
        'center_vy_cm': [],
        'nose_x': [],
        'nose_y': [],
        'nose_x_cm': [],
        'nose_y_cm': [],
        'nose_vx_cm': [],
        'nose_vy_cm': [],
        'rotate': [],
    })


# === 各動画単位の処理関数 ===
def process_video(idx, train_row, tracking_method):
    try:
        lab_id   = str(train_row['lab_id'])
        video_id = str(train_row['video_id'])
        fps      = train_row['frames_per_second']
        pps      = train_row['pix_per_cm_approx']

        tracking_file = f"{TRAIN_TRACKING_DIR}{lab_id}/{video_id}.parquet"
        anno_file     = f"{TRAIN_ANNOTATION_DIR}{lab_id}/{video_id}.parquet"

        # ファイル存在確認
        if not os.path.exists(tracking_file) or not os.path.exists(anno_file):
            write_log(f"⚠️ Missing file: {lab_id}/{video_id}")
            return None

        # tracking_method不一致スキップ
        if tracking_method != train_row['tracking_method']:
            write_log(f"⏩ Skipped (tracking_method mismatch): {lab_id}/{video_id}")
            return None

        # === tracking 読み込み・正規化 ===
        tracking = pd.read_parquet(tracking_file)
        tracking['video_time'] = tracking['video_frame'] / fps
        tracking['x_cm'] = tracking['x'] / pps
        tracking['y_cm'] = tracking['y'] / pps

        start_time, end_time = tracking['video_time'].min(), tracking['video_time'].max()
        bodyparts = tracking['bodypart'].unique()
        mice = tracking['mouse_id'].unique()

        tmp_tracking = []
        for mouse in mice:
            for bp in bodyparts:
                tmp = tracking[(tracking['mouse_id']==mouse) & (tracking['bodypart']==bp)]
                result = reduce_rows(tmp, start_time, end_time, INTERVAL)
                result['vx_cm'] = result['x_cm'].diff()
                result['vy_cm'] = result['y_cm'].diff()
                tmp_tracking.append(result.assign(mouse_id=mouse, bodypart=bp))
        tracking_df = pd.concat(tmp_tracking, ignore_index=True)

        # === annotation 読み込み ===
        anno = pd.read_parquet(anno_file)
        output = initDataFrame_2()

        for _, row in anno.iterrows():
            start_frame, stop_frame = row['start_frame'], row['stop_frame']
            action, agent_id, target_id = row['action'], row['agent_id'], row['target_id']

            for mouse_id in [agent_id, target_id]:
                tmp = tracking_df[(tracking_df['video_frame'] >= start_frame) &
                                  (tracking_df['video_frame'] <= stop_frame) &
                                  (tracking_df['mouse_id'] == mouse_id)]
                if len(tmp) == 0:
                    continue

                nose = tmp[tmp['bodypart'] == 'nose']
                center = tmp[tmp['bodypart'] == 'body_center']
                if len(nose)==0 or len(center)==0:
                    continue

                n = nose.iloc[0]
                c = center.iloc[0]
                rotate = np.arctan2(c['y']-n['y'], c['x']-n['x'])

                output = pd.concat([output, pd.DataFrame({
                    'lab_id': [lab_id],
                    'video_id': [video_id],
                    'video_time': [c['video_time']],
                    'video_frame': [c['video_frame']],
                    'mouse_id': [mouse_id],
                    'agent_target': [mouse_id == agent_id],
                    'action': [action],
                    'center_x': [c['x']], 'center_y': [c['y']],
                    'center_x_cm': [c['x_cm']], 'center_y_cm': [c['y_cm']],
                    'center_vx_cm': [c['vx_cm']], 'center_vy_cm': [c['vy_cm']],
                    'nose_x': [n['x']], 'nose_y': [n['y']],
                    'nose_x_cm': [n['x_cm']], 'nose_y_cm': [n['y_cm']],
                    'nose_vx_cm': [n['vx_cm']], 'nose_vy_cm': [n['vy_cm']],
                    'rotate': [rotate],
                })], ignore_index=True)

        tmp_path = f"{TMP_DIR}/{lab_id}_{video_id}.csv"
        output.to_csv(tmp_path, index=False)
        write_log(f"✅ Completed: {lab_id}/{video_id}")
        return tmp_path

    except Exception as e:
        err = traceback.format_exc()
        write_log(f"❌ Error in {lab_id}/{video_id}: {str(e)}\n{err}")
        return None


# === メイン ===
if __name__ == "__main__":
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)  # 古いログを削除

    write_log("=== Start Processing ===")

    tracking_method = get_test_tracking_method()
    train_data = pd.read_csv(INPUT_TRAIN_FILE)

    results = []
    errors = 0

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {executor.submit(process_video, idx, row, tracking_method): idx
                   for idx, row in train_data.iterrows()}

        for future in as_completed(futures):
            res = future.result()
            if res:
                results.append(res)
            else:
                errors += 1

    # === 結果統合 ===
    if results:
        df_all = pd.concat([pd.read_csv(f) for f in results], ignore_index=True)
        df_all.to_csv(OUTPUT_FILE, index=False)
        write_log(f"🎉 All done! Output: {OUTPUT_FILE}")
    else:
        write_log("⚠️ No output generated.")

    write_log(f"Process finished. Total videos: {len(train_data)}, Errors: {errors}")
