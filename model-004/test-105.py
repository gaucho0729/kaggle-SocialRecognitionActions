# 4匹のマウスが登場する動画データを前提に、
# . 各フレームごとに「順序付きペア(agent→target)」を生成し、
# . annotation.csv の (agent_id, target_id, start_frame, stop_frame, action) をもとにラベル付けし、
# . さらに「±window_size フレームの統計特徴（平均・標準偏差・最小・最大）」を追加
# . 最後に pair_dataset_windowed.csv として保存します。
# . test-104.pyを元にハイパーパラメータを固定化

import pandas as pd
import numpy as np
import joblib
import os
import multiprocessing
import datetime
from tqdm import tqdm
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from pathlib import Path

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
    else:
        INPUT_TRAIN_FILE  = "../train.csv"
    INPUT_TEST_FILE       = "../test.csv"
    TRAIN_ANNOTATION_DIR  = "../train_annotation/"
    TRAIN_TRACKING_DIR    = "../train_tracking/"
    TEST_TRACKING_DIR     = "../test_tracking/"
    OUTPUT_FILE           = "submission.csv"

# === 設定 ===
TRACKING_FILE      = "tracking_features_simplified.csv"   # tracking_features.csv でも可
ANNOTATION_FILE    = "annotation.csv"
PAIR_DATA_SET_FILE = "pair_dataset_windowed.csv"
MODEL_FILE         = "xgb_mouse_behavior_model.pkl"

window_size = 5   # 前後5フレームで統計特徴を計算
neg_sampling_ratio = 1.0  # 正例1件あたり負例を何件残すか (Noneなら全件保持)

# カラム名の確認・設定
VID_COL = 'video_id'
FRAME_COL = 'video_frame'
MID_COL = 'mouse_id'

solo_action = [
    'rear'
]

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


def process_tracking_row(train_row):
    """1本の動画についてtrackingデータを処理"""
    vid = str(train_row['video_id'])
    lid = train_row['lab_id']
    fps  = train_row['frames_per_second']
    ppcm = train_row['pix_per_cm_approx']
    file_path = lid + "/" + vid + ".parquet"
    tracking_data_path = os.path.join(TRAIN_TRACKING_DIR, file_path)

    if not os.path.exists(tracking_data_path):
        print(f"not found {tracking_data_path}")
        return pd.DataFrame()

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
        .groupby(["mouse_id", "bodypart"], group_keys=False)
        .apply(lambda g: g.assign(
            dx_cm=g["x_cm"].diff().fillna(0),
            dy_cm=g["y_cm"].diff().fillna(0),
            speed=np.sqrt(g["x_cm"].diff()**2 + g["y_cm"].diff()**2).fillna(0),
            angle=compute_angle(g["x_cm"].diff(), g["y_cm"].diff())
        ))
    )
    tracking["d_angle"] = (
        tracking.groupby(["mouse_id", "bodypart"])["angle"]
        .diff()
        .fillna(0)
    )
    return tracking


def load_tracking(train):
    """train全体のtrackingデータを並列読み込み"""
    n_jobs = min(multiprocessing.cpu_count(), 8)  # コア数に応じて調整
    print(f"Using {n_jobs} parallel workers")

    dfs = []
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = {executor.submit(process_tracking_row, train.iloc[i]): i for i in range(len(train))}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                df = future.result()
                if not df.empty:
                    dfs.append(df)
            except Exception as e:
                print(f"[Error] row {idx}: {e}")

    if len(dfs) == 0:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)

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
#                    if agent == target:
#                        continue

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
                    if len(acts) > 0:
                        label = acts[0]
                    else:
                        mask_2 = (
                            (annotation_df['video_id'] == vid) &
                            (annotation_df['agent_id'] == target) &
                            (annotation_df['target_id'] == agent) &
                            (annotation_df['start_frame'] <= frame) &
                            (annotation_df['stop_frame'] >= frame)
                        )
                        acts_2 = annotation_df.loc[mask_2, 'action'].unique()
                        if len(acts_2) > 0:
                            label = acts_2[0] + "-ed"
                        else:
                            label = 'none'

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

def process_single_video(train_row):
    """1動画単位でpair_dfを生成"""
    vid = str(train_row["video_id"])
    lid = train_row["lab_id"]
    fps  = train_row["frames_per_second"]
    ppcm = train_row["pix_per_cm_approx"]

    # --- ファイルパス構築 ---
    tracking_path = os.path.join(TRAIN_TRACKING_DIR, f"{lid}/{vid}.parquet")
    annotation_path = os.path.join(TRAIN_ANNOTATION_DIR, f"{lid}/{vid}.parquet")
    if not os.path.exists(tracking_path) or not os.path.exists(annotation_path):
        print("not found tracking or annotation file.")
        return pd.DataFrame()

    # --- データ読み込み ---
    trk = pd.read_parquet(tracking_path)
    ann = pd.read_parquet(annotation_path)
    trk["video_id"] = vid
    trk["video_time"] = trk["video_frame"] / fps
    trk["x_cm"] = trk["x"] / ppcm
    trk["y_cm"] = trk["y"] / ppcm
    ann["video_id"] = vid

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
    pair_df = make_pair_data(trk, ann)
    return pair_df

def make_pair_for_test(tracking_df):
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
#                    if agent == target:
#                        continue

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
#                    else:
#                        print("not found [xy]_cm_body_center")

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

def shrink_actions(df):
    def update_prev(prev_info, idx, agent, target, action, frame):
        """prev_infoを更新"""
        prev_info.iloc[idx] = [action, agent, target, frame]

    ACTION_INTERVAL = 7

    prev_info = pd.DataFrame({
        'action': [''] * 5,
        'agent_id': [0] * 5,
        'target_id': [0] * 5,
        'frame': [0] * 5
    })
    output = pd.DataFrame()
    for i in range(len(df)):
        row = df.iloc[i]
        agent = row['agent_id']
        target = row['target_id']
        action = row['action']
        frame = row['frame']
        prev_agent = prev_info.iloc[agent]
        prev_target = prev_info.iloc[target]
        if (prev_agent['agent_id'] == 0) and (prev_target['target_id'] == 0):
            update_prev(prev_info, agent, agent, target, action, frame)            
        elif (prev_agent['agent_id'] == 0):
            update_prev(prev_info, agent, agent, target, action, frame)            
        elif (prev_agent['target_id'] != target) or (prev_agent['target_id'] != action):
            if prev_agent['frame'] < (frame-ACTION_INTERVAL):
                tmp = pd.DataFrame({
                    'agent_id'   : [prev_agent['agent_id']],
                    'target_id'  : [prev_agent['target_id']],
                    'action'     : [prev_agent['action']],
                    'start_frame': [prev_agent['frame']],
                    'stop_frame' : [frame - 1]
                })
                output = pd.concat([output, tmp])
            if (prev_target['agent_id'] != 0) and (prev_agent['agent_id'] != prev_target['agent_id']):
                if prev_target['frame'] < (frame-ACTION_INTERVAL):
                    tmp = pd.DataFrame({
                        'agent_id'   : [prev_target['agent_id']],
                        'target_id'  : [prev_target['target_id']],
                        'action'     : [prev_target['action']],
                        'start_frame': [prev_target['frame']],
                        'stop_frame' : [frame - 1]
                    })
                    output = pd.concat([output, tmp], ignore_index=True)
            update_prev(prev_info, agent,  agent, target, action, frame)
            update_prev(prev_info, target, 0,     0,      0,      0)
        else:
            # no operation
            continue
    return output


def remove_passive_action(df):
    df_reduced = df[df['action'] != "none"]
    actions = df_reduced['action'].unique()
    reduced_actions = [w for w in actions if "-ed" in w]
    for act in reduced_actions:
        df_reduced = df_reduced[df_reduced['action'] != act]
    return df_reduced


if __name__ == "__main__":
    start_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("[START TIME]", start_time)

    train_data = pd.read_csv(INPUT_TRAIN_FILE)
    n_jobs = min(8, os.cpu_count())
    print(f"並列ワーカー数: {n_jobs}")

    output_path = PAIR_DATA_SET_FILE
    all_pairs = []  # appendモードも可

    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futures = {ex.submit(process_single_video, train_data.iloc[i]): i for i in range(len(train_data))}
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                df = fut.result()
                if len(df) > 0:
                    all_pairs.append(df)
                    print(f"✅ processed video {idx+1}/{len(train_data)} ({len(df):,} rows)")
            except Exception as e:
                print(f"❌ error in train video {idx}: {e}")

    if len(all_pairs) == 0:
        print("No pair data generated.")
        exit()

    pair_df = pd.concat(all_pairs, ignore_index=True)
#    pair_df.to_csv(output_path, index=False)
#    print(f"\n✅ 保存完了: {output_path} ({len(pair_df):,} 行)")

    # === データ読み込み ===
#    print("データ読み込み中...")
#    df = pd.read_csv(PAIR_DATA_SET_FILE, low_memory=False)
    df = pair_df[pair_df['label'] != 'none']

    # 不要列を除外
#    drop_cols = ['video_id', 'frame', 'agent_id', 'target_id']
    drop_cols = ['video_id', 'frame', 'agent_id', 'target_id', 'a_bodypart', 't_bodypart']
    X = df.drop(columns=drop_cols + ['label'])
    y = df['label']
    label = df['label'].unique()

    # 欠損値処理
    X = X.fillna(X.median())

    # === ラベルの分布確認 ===
    print("\nラベル分布:")
    print(y.value_counts())

    # === モデルとパラメータ探索設定 ===
    model = XGBClassifier(
        n_estimators  =  400,
        learning_rate = 0.05,
        max_depth     = 6,
        objective     = 'multi:softmax',
        num_class     = len(y.unique()),
#        eval_metric='mlogloss',
        eval_metric   = 'logloss',
        tree_method   = 'hist',  # GPU利用可なら 'gpu_hist'
        subsample     = 0.9,
        colsample_bytree = 0.9,
        use_label_encoder=False,
        n_job = -1,
        random_state=42
    )

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

#    le = LabelEncoder()
#    y = le.fit_transform(df['label'])
    y = df['label']
    X_sample = X.sample(frac=0.2, random_state=42)
    y_sample = y.loc[X_sample.index]

    le = LabelEncoder()
    y_sample = le.fit_transform(y_sample)

    # === 学習 ===
    print("\nBayesSearchCV によるハイパーパラメータ最適化中...")
#    opt.fit(X, y)
    model.fit(X_sample, y_sample)

#    print("\n最良パラメータ:")
#    print(opt.best_params_)
#    print(f"最良スコア (CV f1_macro): {opt.best_score_:.4f}")

    # === 学習済みモデルを保存 ===
#    joblib.dump(opt.best_estimator_, MODEL_FILE)
#    print(f"\n✅ モデルを保存しました: {MODEL_FILE}")

    # === 最終学習済みモデルで予測と評価 ===
    y_pred_value = model.predict(X)
    y_pred = le.inverse_transform(y_pred_value)
    print("\n=== 訓練データでの性能評価 ===")
    print(classification_report(y, y_pred))

    # 混同行列も確認
    print("\n=== 混同行列 ===")
    print(confusion_matrix(y, y_pred))

    # テストデータ読み込み
    test_data = pd.read_csv(INPUT_TEST_FILE)

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

    testX = pd.concat(all_test_pairs, ignore_index=True)
    testX.to_csv('test_pair_dataset_windowed.csv')

    print("X_sample:")
    X_sample.info()
    print("testX:")
    testX.info()
    testX_copy = testX.copy()
    testX = testX.drop(columns=['video_id','frame','agent_id','target_id','a_bodypart','t_bodypart'])

    test_y_pred_value = model.predict(testX)
    y_tmp = pd.DataFrame(test_y_pred_value)
    y_tmp.to_csv('test_y_pred_value.csv')
    test_y = le.inverse_transform(test_y_pred_value)

    # todo submission file.
    subm = pd.DataFrame({
        'video_id' : testX_copy['video_id'],
        'frame'    : testX_copy['frame'],
        'agent_id' : testX_copy['agent_id'],
        'target_id': testX_copy['target_id'],
        'action'   : test_y
    })
    subm = subm[subm['action']!='none']
    subm['agent_id'] = subm['agent_id'].astype(int)
    subm['target_id'] = subm['target_id'].astype(int)
    for act in solo_action:
        subm = subm[~((subm["action"] == act) & (df["agent_id"] != df["target_id"]))]
    subm.to_csv("submission-0.csv")
    if len(subm) > 0:
        subm = remove_passive_action(subm)
    subm = shrink_actions(subm)
    subm['start_frame'] = subm['start_frame'].astype(int)
    subm['stop_frame'] = subm['stop_frame'].astype(int)
    subm.index.name = 'row_id'
    subm.to_csv('submission.csv', index=True)

    print("Complete of script")
    end_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("[STOP TIME]", end_time)
