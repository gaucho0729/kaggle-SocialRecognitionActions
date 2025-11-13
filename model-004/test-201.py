import pandas as pd
import numpy as np
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# #   Column        Dtype  
#---  ------        -----  
# 0   video_id      int64  
# 1   frame         float64
# 2   agent_id      float64
# 3   target_id     float64
# 4   label         object  <- 目的変数なので不要
# 5   a_video_time  float64
# 6   t_video_time  float64
# 7   a_bodypart    object 
# 8   t_bodypart    object 
# 9   a_x           float64
# 10  t_x           float64
# 11  a_y           float64
# 12  t_y           float64
# 13  a_x_cm        float64
# 14  t_x_cm        float64
# 15  a_y_cm        float64
# 16  t_y_cm        float64
# 17  a_dx_cm       float64
# 18  t_dx_cm       float64
# 19  a_dy_cm       float64
# 20  t_dy_cm       float64
# 21  a_speed       float64
# 22  t_speed       float64
# 23  a_angle       float64
# 24  t_angle       float64
# 25  a_d_angle     float64
# 26  t_d_angle     float64

kaggle = False
innter_test = True
enable_save_train_data = True

# Variables for environment (環境) -------------------------
if kaggle == True:
    DATA_DIR = Path("/kaggle/input/MABe-mouse-behavior-detection")
    SUBMISSION_DIR = Path("/kaggle/working")
    INPUT_TRAIN_FILE = DATA_DIR /"train.csv"
    INPUT_TEST_FILE = DATA_DIR /"test.csv"
    TRAIN_ANNOTATION_DIR   = DATA_DIR /"train_annotation/"
    TRAIN_TRACKING_DIR     = DATA_DIR /"train_tracking/"
    TEST_TRACKING_DIR      = DATA_DIR /"test_tracking/"
    OUTPUT_FILE            = SUBMISSION_DIR / "submission.csv"
else:
    if innter_test == True:
        INPUT_TRAIN_FILE = "train.csv"
    else:
        INPUT_TRAIN_FILE  = "../train.csv"
    INPUT_TEST_FILE       = "../test.csv"
    TRAIN_ANNOTATION_DIR  = "../train_annotation/"
    TRAIN_TRACKING_DIR    = "../train_tracking/"
    TEST_TRACKING_DIR     = "../test_tracking/"
    OUTPUT_FILE           = "submission.csv"

TEST_FILE = "../test.csv"
OUTPUT_FILE = "test_pair_dataset_windowed.csv"

INTERVAL = 0.25

row_threshold = 10000

# カラム名の確認・設定
VID_COL = 'video_id'
FRAME_COL = 'video_frame'
MID_COL = 'mouse_id'

# resuce row data
# trackingデータをinterval毎に間引く
# src_data : original data
# start_time : start time for reducing
# end_time   : end time for reducing
# interval   : inval time for reducting (sec)
def reduce_rows(src_data, start_time, end_time, interval):
    retval = pd.DataFrame()
    mice = src_data['mouse_id'].unique()
    bodyparts = src_data['bodypart'].unique()
    for mouse in mice:
        for bp in bodyparts:
            tmp_df = src_data[(src_data['mouse_id']==mouse) &
                              (src_data['bodypart']==bp)]
            target_times = np.arange(start_time, end_time + 1e-8, interval)  # end を含めたい場合

            # necessary for sort by key if using merge_asof
            # merge_asof を使う場合はキーでソートが必要
            sorted = tmp_df.sort_values('video_time').reset_index(drop=True)
            target_df = pd.DataFrame({'video_time': target_times})

            # merge 'nearest' line by direction='nearest
            # direction='nearest' で「最も近い」行を結合
            result = pd.merge_asof(target_df, sorted, on='video_time', direction='nearest')
            retval = pd.concat([retval, result], ignore_index=True)
    return retval


# --- 座標差分と角度系 ---
def compute_angle(dx, dy):
    """差分から角度（度単位）を計算"""
    return np.degrees(np.arctan2(dy, dx))


def make_pair_for_test(tracking_df):
    pair_rows = []
    i = 0
    for vid, df_vid in tqdm(tracking_df.groupby(VID_COL), desc="Processing videos"):
        df_vid = df_vid.sort_values(FRAME_COL)
        frames = df_vid[FRAME_COL].unique()
        
        # 各フレームで順序付きペア(agent→target)を作成
        for frame in frames:
            frame_df = df_vid[df_vid[FRAME_COL] == frame]
            mice = frame_df[MID_COL].unique()
            # 各マウスの行を辞書化
            rows = {mid: row for mid, row in frame_df.set_index(MID_COL).iterrows()}

            for agent in mice:
                for target in mice:
                    if agent == target:
                        continue

                    a = rows[agent]
                    t = rows[target]
                    feat_cols = [c for c in df_vid.columns if c not in [VID_COL, FRAME_COL, MID_COL]]

                    # 特徴量をそれぞれ接頭辞 a_ / t_ で付与
                    feats = {}
                    for c in feat_cols:
                        feats[f"a_{c}"] = a[c]
                        feats[f"t_{c}"] = t[c]

                    # 相対特徴（距離や角度）
                    if 'x_cm_body_center' in a and 'y_cm_body_center' in a:
                        dx = t['x_cm_body_center'] - a['x_cm_body_center']
                        dy = t['y_cm_body_center'] - a['y_cm_body_center']
                        feats['dist_agent_target'] = np.sqrt(dx**2 + dy**2)
                        feats['dx_agent_target'] = dx
                        feats['dy_agent_target'] = dy
                        feats['angle_to_target'] = np.degrees(np.arctan2(dy, dx))
                    else:
                        i = i + 1
                        print("not found [xy]_cm_body_center #", i)

                    # ラベル判定
#                    mask = (
#                        (annotation_df['video_id'] == vid) &
#                        (annotation_df['agent_id'] == agent) &
#                        (annotation_df['target_id'] == target) &
#                        (annotation_df['start_frame'] <= frame) &
#                        (annotation_df['stop_frame'] >= frame)
#                    )
#                    acts = annotation_df.loc[mask, 'action'].unique()
#                    label = acts[0] if len(acts) > 0 else 'none'

                    pair_rows.append({
                        'video_id': vid,
                        'frame': frame,
                        'agent_id': agent,
                        'target_id': target,
                        **feats
                    })

    pair_df = pd.DataFrame(pair_rows)
    return pair_df

def process_test_video(test_row):
    """1動画単位でpair_dfを生成"""
    vid = str(test_row["video_id"])
    lid = test_row["lab_id"]
    fps  = test_row["frames_per_second"]
    ppcm = test_row["pix_per_cm_approx"]

    # --- ファイルパス構築 ---
    tracking_path = os.path.join(TEST_TRACKING_DIR, f"{lid}/{vid}.parquet")
    if not os.path.exists(tracking_path):
        print("not found tracking or annotation file.")
        return pd.DataFrame()

    # --- データ読み込み ---
    trk = pd.read_parquet(tracking_path)
    trk["video_id"] = vid
    trk["video_time"] = trk["video_frame"] / fps
    trk["x_cm"] = trk["x"] / ppcm
    trk["y_cm"] = trk["y"] / ppcm

    # --- reduce_rows で補間 ---
    start_time, end_time = trk["video_time"].min(), trk["video_time"].max()
    trk = reduce_rows(trk, start_time, end_time, INTERVAL)

    # --- 差分計算 ---
    trk = (
        trk.groupby(["mouse_id", "bodypart"], group_keys=False)
        .apply(lambda g: g.assign(
            dx_cm=g["x_cm"].diff().fillna(0),
            dy_cm=g["y_cm"].diff().fillna(0),
            speed=np.sqrt(g["x_cm"].diff()**2 + g["y_cm"].diff()**2).fillna(0),
            angle=compute_angle(g["x_cm"].diff(), g["y_cm"].diff())
        ))
    )
    trk["d_angle"] = (
        trk.groupby(["mouse_id", "bodypart"])["angle"]
        .diff()
        .fillna(0)
    )

    # --- ペア生成 ---
    pair_df = make_pair_for_test(trk)
    return pair_df


if __name__ == "__main__":
    test_data = pd.read_csv(TEST_FILE)

    n_jobs = min(8, os.cpu_count())
    print(f"並列ワーカー数: {n_jobs}")

    output_path = OUTPUT_FILE
    all_test_pairs = []  # appendモードも可

    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futures = {ex.submit(process_test_video, test_data.iloc[i]): i for i in range(len(test_data))}
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                df = fut.result()
                if len(df) > 0:
                    all_test_pairs.append(df)
                    print(f"✅ processed video {idx+1}/{len(test_data)} ({len(df):,} rows)")
            except Exception as e:
                print(f"❌ error in video {idx}: {e}")

    if len(all_test_pairs) == 0:
        print("No pair data generated.")
        exit()

    pair_df = pd.concat(all_test_pairs, ignore_index=True)
    pair_df.to_csv(output_path, index=False)
    print(f"\n✅ 保存完了: {output_path} ({len(pair_df):,} 行)")
