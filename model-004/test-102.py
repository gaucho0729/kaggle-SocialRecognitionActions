# 4匹のマウスが登場する動画データを前提に、
# . 各フレームごとに「順序付きペア(agent→target)」を生成し、
# . annotation.csv の (agent_id, target_id, start_frame, stop_frame, action) をもとにラベル付けし、
# . さらに「±window_size フレームの統計特徴（平均・標準偏差・最小・最大）」を追加
# . 最後に pair_dataset_windowed.csv として保存します。

import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from itertools import combinations

kaggle = False
innter_test = True
enable_save_train_data = True

# 
INTERVAL = 0.25

row_threshold = 10000

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
#        INPUT_TRAIN_FILE = "train-001.csv"
    else:
        INPUT_TRAIN_FILE  = "../train.csv"
    INPUT_TEST_FILE       = "../test.csv"
    TRAIN_ANNOTATION_DIR  = "../train_annotation/"
    TRAIN_TRACKING_DIR    = "../train_tracking/"
    TEST_TRACKING_DIR     = "../test_tracking/"
    OUTPUT_FILE           = "submission.csv"

# === 設定 ===
TRACKING_FILE = "tracking_features_simplified.csv"   # tracking_features.csv でも可
ANNOTATION_FILE = "annotation.csv"
OUTPUT_FILE = "pair_dataset_windowed.csv"

window_size = 5   # 前後5フレームで統計特徴を計算
neg_sampling_ratio = 1.0  # 正例1件あたり負例を何件残すか (Noneなら全件保持)

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


def make_tracking_feature(tracking):
    retval = pd.DataFrame()
    tracking = tracking.dropna(subset=["bodypart"])
    bodyparts = tracking["bodypart"].unique()
    bodyparts = np.sort(bodyparts)[::-1]
    pair_list = list(combinations(bodyparts, 2))
    distance_features = []
#    for bp1, bp2 in pair_list:
#        df1 = tracking[tracking["bodypart"] == bp1][["video_time", "mouse_id", "x_cm", "y_cm"]].rename(columns={"x_cm": f"x_cm_{bp1}", "y_cm": f"y_cm_{bp1}"})
#        df2 = tracking[tracking["bodypart"] == bp2][["video_time", "mouse_id", "x_cm", "y_cm"]].rename(columns={"x_cm": f"x_cm_{bp2}", "y_cm": f"y_cm_{bp2}"})
#        merged = pd.merge(df1, df2, on=["video_time", "mouse_id"], how="inner")
#        merged[f"dist_{bp1}_{bp2}"] = np.sqrt((merged[f"x_cm_{bp1}"] - merged[f"x_cm_{bp2}"])**2 + (merged[f"y_cm_{bp1}"] - merged[f"y_cm_{bp2}"])**2)
            
#        distance_features.append(merged[["video_time", "mouse_id", f"dist_{bp1}_{bp2}"]])

    print("len(distance_features):",len(distance_features))

    distance_df = distance_features[0]
    for df in distance_features[1:]:
        distance_df = pd.merge(distance_df, df, on=["video_time", "mouse_id"], how="outer")

    distance_df.to_csv('distance_df.csv')

    # === 統合 ===
    # bodypart単位の特徴（速度・角度など）をpivot
    tracking_pivot = tracking.pivot_table(
#        index=["video_id", "video_frame", "mouse_id"],
        index=["video_time", "mouse_id"],
        columns="bodypart",
        values=["dx_cm", "dy_cm", "speed", "angle", "d_angle"]
    )
    tracking_pivot.to_csv('tracking_pivot.csv')

    # カラム階層をフラット化
    tracking_pivot.columns = [f"{feat}_{bp}" for feat, bp in tracking_pivot.columns]
    tracking_pivot = tracking_pivot.reset_index()
    tracking_pivot.to_csv('tracking_pivot-2.csv')

    # 全ての特徴をマージ
#    tracking_features = pd.merge(tracking_pivot, distance_df, on=["video_id", "video_frame", "mouse_id"], how="left")
    tracking_features = pd.merge(tracking_pivot, distance_df, on=["video_time", "mouse_id"], how="left")
    tracking_features.to_csv('tracking_features.csv')

    # body_centerのx,yを追加する
    tmp_center = tracking[tracking['bodypart'] == 'body_center']
    tmp_center.to_csv('tmp_center.csv')

    tmp_center = tmp_center.drop(['bodypart','dx_cm','dy_cm','speed','angle','d_angle'],axis=1)
    tracking_features = pd.merge(tracking_features, tmp_center, on=['video_time', 'mouse_id'], how='left')
    tracking_features = tracking_features.rename(columns={'x_cm':'x_cm_body_center', 'y_cm':'y_cm_body_center'})

    tracking_features.to_csv('tracking_features.csv')

    retval = pd.concat([retval, tracking_features])

    return retval

# --- 座標差分と角度系 ---
def compute_angle(dx, dy):
    """差分から角度（度単位）を計算"""
    return np.degrees(np.arctan2(dy, dx))


# trackingファイルを読み込む
def load_tracking(train):
    retval = pd.DataFrame()
    for i in range(len(train)):
        train_row = train.iloc[i]
        vid = str(train_row['video_id'])
        lid = train_row['lab_id']
        fps  = train_row['frames_per_second']
        ppcm = train_row['pix_per_cm_approx']
        file_path = lid + "/" + vid + ".parquet"
        tracking_data_path = os.path.join(TRAIN_TRACKING_DIR, file_path)
        if os.path.exists(tracking_data_path) == False:
            print('not found ', tracking_data_path)
            continue
        trk = pd.read_parquet(tracking_data_path)
        trk['video_id'] = vid
        trk['video_time'] = trk['video_frame'] / fps
        trk['x_cm'] = trk['x'] / ppcm
        trk['y_cm'] = trk['y'] / ppcm
        start_time = trk['video_time'].min()
        end_time   = trk['video_time'].max()
        trk_reduced = reduce_rows(trk, start_time, end_time, INTERVAL)
        tracking = (
            trk_reduced
#            .groupby(["video_id", "mouse_id", "bodypart"], group_keys=False)
            .groupby(["mouse_id", "bodypart"], group_keys=False)
            .apply(lambda g: g.assign(
                dx_cm=g["x_cm"].diff().fillna(0),
                dy_cm=g["y_cm"].diff().fillna(0),
                speed=np.sqrt(g["x_cm"].diff()**2 + g["y_cm"].diff()**2).fillna(0),
                angle=compute_angle(g["x_cm"].diff(), g["y_cm"].diff())
            ))
        )
        # --- 回転角（角度変化） ---
        tracking["d_angle"] = (
#            tracking.groupby(["video_id", "mouse_id", "bodypart"])["angle"]
            tracking.groupby(["mouse_id", "bodypart"])["angle"]
            .diff()
            .fillna(0)
        )
        tracking.info()
#        trk_featured = make_tracking_feature(tracking)
#        retval = pd.concat([retval, trk_featured])
        retval = pd.concat([retval, tracking])
    return retval

def load_annotation(train):
    retval = pd.DataFrame()
    for i in range(len(train)):
        train_row = train.iloc[i]
        vid = str(train_row['video_id'])
        lid = train_row['lab_id']
        file_path = lid + "/" + vid + ".parquet"
        annotation_data_path = os.path.join(TRAIN_ANNOTATION_DIR, file_path)
        if os.path.exists(annotation_data_path) == False:
            continue
        trk = pd.read_parquet(annotation_data_path)
        trk['video_id'] = vid
        retval = pd.concat([retval, trk])
    return retval

def make_pair_data(tracking_df, annotation_df):
    pair_rows = []
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

                    # ラベル判定
                    mask = (
                        (annotation_df['video_id'] == vid) &
                        (annotation_df['agent_id'] == agent) &
                        (annotation_df['target_id'] == target) &
                        (annotation_df['start_frame'] <= frame) &
                        (annotation_df['stop_frame'] >= frame)
                    )
                    acts = annotation_df.loc[mask, 'action'].unique()
                    label = acts[0] if len(acts) > 0 else 'none'

                    pair_rows.append({
                        'video_id': vid,
                        'frame': frame,
                        'agent_id': agent,
                        'target_id': target,
                        'label': label,
                        **feats
                    })

    pair_df = pd.DataFrame(pair_rows)
    return pair_df

def calc_stats_in_pair(pair_df):
    # 数値列のみを対象
    num_cols = pair_df.select_dtypes(include=[np.number]).columns
    exclude_cols = ['video_id', 'frame', 'agent_id', 'target_id']
    num_cols = [c for c in num_cols if c not in exclude_cols]

    # 統計特徴の関数
    def add_window_features(df_vid):
        df_vid = df_vid.sort_values(['frame', 'agent_id', 'target_id'])
        for c in tqdm(num_cols, desc=f"  features ({df_vid[VID_COL].iloc[0]})", leave=False):
            df_vid[f"{c}_mean"] = df_vid[c].rolling(window=2*window_size+1, center=True, min_periods=1).mean()
            df_vid[f"{c}_std"]  = df_vid[c].rolling(window=2*window_size+1, center=True, min_periods=1).std()
            df_vid[f"{c}_min"]  = df_vid[c].rolling(window=2*window_size+1, center=True, min_periods=1).min()
            df_vid[f"{c}_max"]  = df_vid[c].rolling(window=2*window_size+1, center=True, min_periods=1).max()
        return df_vid

    pair_df = pair_df.groupby(VID_COL, group_keys=False).apply(add_window_features)
    print("ウィンドウ統計特徴追加完了。")

    # === 負例サンプリング ===
    if neg_sampling_ratio is not None:
        pos_df = pair_df[pair_df['label'] != 'none']
        neg_df = pair_df[pair_df['label'] == 'none']
        n_pos = len(pos_df)
        n_neg_keep = int(n_pos * neg_sampling_ratio)
        neg_df_sample = neg_df.sample(n=min(n_neg_keep, len(neg_df)), random_state=42)
        pair_df = pd.concat([pos_df, neg_df_sample]).reset_index(drop=True)
        print(f"負例サンプリング後: {len(pair_df):,} 行 (正例 {len(pos_df)}, 負例 {len(neg_df_sample)})")
    return pair_df

# === メイン ===
if __name__ == "__main__":
    train_data = pd.read_csv(INPUT_TRAIN_FILE)
    tracking_df = load_tracking(train_data)
    if len(tracking_df) == 0:
        print("can't read tracking_data")
        exit()

    annotation_df = load_annotation(train_data)

    annotation_df.info()

    # === ペアデータ生成 ===
    pair_df = make_pair_data(tracking_df, annotation_df)
    print(f"ペアデータ生成完了: {len(pair_df):,} 行")

    # === ウィンドウ特徴量の追加 ===
    print(f"ウィンドウ統計特徴（±{window_size}フレーム）を計算中...")
    pair_df = calc_stats_in_pair(pair_df)

    # === 保存 ===
    pair_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ 保存完了: {OUTPUT_FILE}")
    print(f"カラム数: {pair_df.shape[1]}, 行数: {pair_df.shape[0]:,}")



