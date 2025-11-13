#import polars as pl
import pandas as pd
import numpy as np
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import datetime
import xgboost
import xgboost as xgb
from pathlib import Path
from itertools import combinations
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tqdm import tqdm

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
#        INPUT_TRAIN_FILE = "train.csv"
        INPUT_TRAIN_FILE = "train-001.csv"
    else:
        INPUT_TRAIN_FILE  = "../train.csv"
    INPUT_TEST_FILE       = "../test.csv"
    TRAIN_ANNOTATION_DIR  = "../train_annotation/"
    TRAIN_TRACKING_DIR    = "../train_tracking/"
    TEST_TRACKING_DIR     = "../test_tracking/"
    OUTPUT_FILE           = "submission.csv"

bool_map = {
    "True":  1,
    "False": 0,
}


# returns bodypart and action in src_data
# src_dataに記述されているbodypartとactionを返す
def find_bodypart_action(src_data):
    bodyparts = []
    actions   = []
    for i in range(len(src_data)):
        train = src_data.iloc[i]
        body_parts_tracked = json.loads(train['body_parts_tracked'].replace('""', '"'))
        for j in range(len(body_parts_tracked)):
            body_parts = body_parts_tracked[j]
            bodyparts.append(body_parts)
        if pd.isna(train['behaviors_labeled']) == False:
            behaviors_labeled  = json.loads(train['behaviors_labeled'].replace('""', '"'))
            for i in range(len(behaviors_labeled)):
                act = behaviors_labeled[i].split(',')
                act[2] = act[2].replace("'", "")
                actions.append(act[2])
    return np.unique(bodyparts), np.unique(actions)

# childrenがmotherに含まれているかどうかを確認する
# children   子
# mother     母集団
def is_including_in(children, mother):
    retval = False
    for i in range(len(children)):
        child = children[i]
        if child in mother:
            return True
    return retval

# 対象の訓練情報を作成する
#   * testと異なるtracking_methodは訓練対象から外す
def create_valid_train():
    test_data = pd.read_csv(INPUT_TEST_FILE)
    train_data = pd.read_csv(INPUT_TRAIN_FILE)

    test_tracking_method = test_data['tracking_method'].iloc[0]
    tmp_train = pd.DataFrame()

    test_bodyparts, test_actions = find_bodypart_action(test_data)

    for i in range(len(train_data)):
        train = train_data.iloc[i]
        vid = str(train['video_id'])
        lid = train['lab_id']
        
        if train['tracking_method'] != test_tracking_method:
            continue
        file_name = lid + '/' + vid + '.parquet'
        tracking_data_path = os.path.join(TRAIN_TRACKING_DIR, file_name)
        if os.path.exists(tracking_data_path) == False:
            continue
        annotation_data_path = os.path.join(TRAIN_ANNOTATION_DIR, file_name)
        if os.path.exists(annotation_data_path) == False:
            continue
        behaviors_labeled  = json.loads(train['behaviors_labeled'].replace('""', '"'))
        train_actions = []
        for i in range(len(behaviors_labeled)):
            act = behaviors_labeled[i].split(',')
            act[2] = act[2].replace("'", "")
            train_actions.append(act[2])
        train_actions = np.unique(train_actions)

        if is_including_in(train_actions,test_actions) == True:
            tmp = pd.DataFrame({
                'lab_id'             : [train['lab_id']]             * 1,
                'video_id'           : [train['video_id']]           * 1,
                'mouse1_strain'      : [train['mouse1_strain']]      * 1,
                'mouse1_color'       : [train['mouse1_color']]       * 1,
                'mouse1_sex'         : [train['mouse1_sex']]         * 1,
                'mouse1_id'          : [train['mouse1_id']]          * 1,
                'mouse1_age'         : [train['mouse1_age']]         * 1,
                'mouse1_condition'   : [train['mouse1_condition']]   * 1,
                'mouse2_strain'      : [train['mouse2_strain']]      * 1,
                'mouse2_color'       : [train['mouse2_color']]       * 1,
                'mouse2_sex'         : [train['mouse2_sex']]         * 1,
                'mouse2_id'          : [train['mouse2_id']]          * 1,
                'mouse2_age'         : [train['mouse2_age']]         * 1,
                'mouse2_condition'   : [train['mouse2_condition']]   * 1,
                'mouse3_strain'      : [train['mouse3_strain']]      * 1,
                'mouse3_color'       : [train['mouse3_color']]       * 1,
                'mouse3_sex'         : [train['mouse3_sex']]         * 1,
                'mouse3_id'          : [train['mouse3_id']]          * 1,
                'mouse3_age'         : [train['mouse3_age']]         * 1,
                'mouse3_condition'   : [train['mouse3_condition']]   * 1,
                'mouse4_strain'      : [train['mouse4_strain']]      * 1,
                'mouse4_color'       : [train['mouse4_color']]       * 1,
                'mouse4_sex'         : [train['mouse4_sex']]         * 1,
                'mouse4_id'          : [train['mouse4_id']]          * 1,
                'mouse4_age'         : [train['mouse4_age']]         * 1,
                'mouse4_condition'   : [train['mouse4_condition']]   * 1,
                'frames_per_second'  : [train['frames_per_second']]  * 1,
                'video_duration_sec' : [train['video_duration_sec']] * 1,
                'pix_per_cm_approx'  : [train['pix_per_cm_approx']]  * 1,
                'video_width_pix'    : [train['video_width_pix']]    * 1,
                'video_height_pix'   : [train['video_height_pix']]   * 1,
                'arena_width_cm'     : [train['arena_width_cm']]     * 1,
                'arena_height_cm'    : [train['arena_height_cm']]    * 1,
                'arena_shape'        : [train['arena_shape']]        * 1,
                'arena_type'         : [train['arena_type']]         * 1,
                'body_parts_tracked' : [train['body_parts_tracked']] * 1,
                'behaviors_labeled'  : [train['behaviors_labeled']]  * 1,
                'tracking_method'    : [train['tracking_method']]    * 1
            })
            tmp_train = pd.concat([tmp_train,tmp],ignore_index=True)
    return tmp_train

# --- 座標差分と角度系 ---
def compute_angle(dx, dy):
    """差分から角度（度単位）を計算"""
    return np.degrees(np.arctan2(dy, dx))

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

# trackingの特徴データを作成する
#   * train_dataを元にtrackingファイルを読み込む
#   * trackingデータのフレームを秒に変換する (video_frame -> video_time)
#   * trackingデータの座標をcmに変換する (x -> x_cm, y -> y_cm)
#   * trackingデータをinterval毎に間引く (そうしないとデータ量が膨大になり、処理速度が落ちる)
#   * マウスの移動距離(dx_cm,dy_cm)、角度(angle)、角度変化(d_angle)、速度(speed)を求める
#   * マウスの部位毎の特徴名に変更する(ex.dx_cm_nose,dy_cm_nose)
#   * 同一時間のマウスの部位情報を1行にまとめる
def make_tracking_feature(train_data, train_or_test):
    output_df = pd.DataFrame()

    for i in range(len(train_data)):
    # === 入力 ===
        train = train_data.iloc[i]
        lab_id = train['lab_id']
        video_id = str(train['video_id'])
        fps  = train['frames_per_second']
        ppcm = train['pix_per_cm_approx']
        tracking_file_name = lab_id + '/' + video_id + '.parquet'
        if train_or_test:
            tracking_file_path = os.path.join(TRAIN_TRACKING_DIR, tracking_file_name)
        else:
            tracking_file_path = os.path.join(TEST_TRACKING_DIR, tracking_file_name)
        if os.path.exists(tracking_file_path) == False:
            continue

        tracking = pd.read_parquet(tracking_file_path)

        # フレーム→時間
        tracking['video_time'] = tracking['video_frame'] / fps

        # 座標→cm
        tracking['x_cm'] = tracking['x'] / ppcm
        tracking['y_cm'] = tracking['y'] / ppcm

        # 間引く
        start_time = tracking['video_time'].min()
        end_time   = tracking['video_time'].max()
        tracking = reduce_rows(tracking, start_time, end_time, INTERVAL)

        tracking = (
            tracking
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

        # === 部位間距離 ===
        # 部位の組み合わせを作成（例：nose-bodycenter など）
        bodyparts = tracking["bodypart"].unique()
        bodyparts = np.sort(bodyparts)[::-1]
        pair_list = list(combinations(bodyparts, 2))

        distance_features = []

        for bp1, bp2 in pair_list:
#            df1 = tracking[tracking["bodypart"] == bp1][["video_id", "video_frame", "mouse_id", "x", "y"]].rename(columns={"x": f"x_{bp1}", "y": f"y_{bp1}"})
#            df2 = tracking[tracking["bodypart"] == bp2][["video_id", "video_frame", "mouse_id", "x", "y"]].rename(columns={"x": f"x_{bp2}", "y": f"y_{bp2}"})
#            merged = pd.merge(df1, df2, on=["video_id", "video_frame", "mouse_id"], how="inner")
            df1 = tracking[tracking["bodypart"] == bp1][["video_time", "mouse_id", "x_cm", "y_cm"]].rename(columns={"x_cm": f"x_cm_{bp1}", "y_cm": f"y_cm_{bp1}"})
            df2 = tracking[tracking["bodypart"] == bp2][["video_time", "mouse_id", "x_cm", "y_cm"]].rename(columns={"x_cm": f"x_cm_{bp2}", "y_cm": f"y_cm_{bp2}"})
            merged = pd.merge(df1, df2, on=["video_time", "mouse_id"], how="inner")
            merged[f"dist_{bp1}_{bp2}"] = np.sqrt((merged[f"x_cm_{bp1}"] - merged[f"x_cm_{bp2}"])**2 + (merged[f"y_cm_{bp1}"] - merged[f"y_cm_{bp2}"])**2)
            
#            distance_features.append(merged[["video_id", "video_frame", "mouse_id", f"dist_{bp1}_{bp2}"]])
            distance_features.append(merged[["video_time", "mouse_id", f"dist_{bp1}_{bp2}"]])

        # 全距離特徴を統合
        distance_df = distance_features[0]
        for df in distance_features[1:]:
#            distance_df = pd.merge(distance_df, df, on=["video_id", "video_frame", "mouse_id"], how="outer")
            distance_df = pd.merge(distance_df, df, on=["video_time", "mouse_id"], how="outer")

        # === 統合 ===
        # bodypart単位の特徴（速度・角度など）をpivot
        tracking_pivot = tracking.pivot_table(
#            index=["video_id", "video_frame", "mouse_id"],
            index=["video_time", "mouse_id"],
            columns="bodypart",
            values=["dx_cm", "dy_cm", "speed", "angle", "d_angle"]
        )

        # カラム階層をフラット化
        tracking_pivot.columns = [f"{feat}_{bp}" for feat, bp in tracking_pivot.columns]
        tracking_pivot = tracking_pivot.reset_index()

        # 全ての特徴をマージ
#        tracking_features = pd.merge(tracking_pivot, distance_df, on=["video_id", "video_frame", "mouse_id"], how="left")
        tracking_features = pd.merge(tracking_pivot, distance_df, on=["video_time", "mouse_id"], how="left")

        # body_centerのx,yを追加する
        tmp_center = tracking[tracking['bodypart'] == 'body_center']

        tmp_center = tmp_center.drop(['bodypart','dx_cm','dy_cm','speed','angle','d_angle'],axis=1)
        tracking_features = pd.merge(tracking_features, tmp_center, on=['video_time', 'mouse_id'], how='left')
        tracking_features = tracking_features.rename(columns={'x_cm':'x_cm_body_center', 'y_cm':'y_cm_body_center'})

        tracking_features['video_id'] = video_id
        # === 出力 ===
#        tracking_features.to_csv(OUTPUT_FILE, mode='a', index=False)
        output_df = pd.concat([output_df, tracking_features])
    return output_df

# === 2. カテゴリ分け関数 ===
def classify_feature(col):
    col_lower = col.lower()
    if re.search(r'(?:^|_)x|(?:^|_)y', col_lower) and not re.search('dx|dy', col_lower):
        return 'position'
    elif re.search(r'dx|dy|speed', col_lower):
        return 'velocity'
    elif re.search(r'angle|rotation', col_lower):
        return 'angle'
    elif re.search(r'dist', col_lower):
        return 'distance'
    elif re.search(r'mean|std|var|median|max|min', col_lower):
        return 'statistics'
    elif re.search(r'frame|time|delta', col_lower):
        return 'temporal'
    else:
        return 'other'

# test-107
def create_feature_category(df):
    # === 3. 全列に対して分類 ===
    feature_classes = {col: classify_feature(col) for col in df.columns}
    feature_df = pd.DataFrame.from_dict(feature_classes, orient='index', columns=['category'])

    return feature_df

# test-108.py
def create_tracking_features_reduced(df,feature_categories):
    # === 2. 使うカテゴリを指定 ===
    # feature_categories.csv から必要なカテゴリを選んでおく
    keep_categories = ['velocity', 'distance', 'angle', 'statistics']  # ← ここを調整
    keep_cols = [
        col for col, cat in feature_categories['category'].items()
        if cat in keep_categories and col in df.columns
    ]

    # === 3. 対象データ抽出 ===
    meta_cols = [c for c in df.columns if c in ['video_time', 'video_id', 'mouse_id', 'bodypart']]
    df_sel = df[meta_cols + keep_cols].copy()

    print(f"特徴量数（抽出後）: {len(keep_cols)}")

    # === 4. 相関行列を作成 ===
    numeric_df = df_sel.select_dtypes(include=[np.number]).dropna(axis=1, how='all')
    corr_matrix = numeric_df.corr().abs()

    # === 5. 高相関の列を削除 ===
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    threshold = 0.95  # 95%以上の相関を削除対象に
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    print(f"高相関特徴量（削除対象）: {len(to_drop)} 列")
    df_reduced = df_sel.drop(columns=to_drop, errors='ignore')
    return df_reduced


def create_pca(df):
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.fillna(df.mean())
    # === 2. 数値列だけ抽出 ===
    numeric_df = df.select_dtypes(include=[np.number]).dropna(axis=1, how='all')

    # === 3. 標準化 ===
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(numeric_df)

    # === 4. PCA適用 ===
    n_components = min(10, X_scaled.shape[1])  # 多すぎないように最大10成分まで
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(n_components)])

    return pca_df

# test-111.py
def create_tracking_features_clustered(df):
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.fillna(df.mean())
    # 数値列のみ抽出
    numeric_df = df.select_dtypes(include=[float, int])

    # === 2. 標準化 ===
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(numeric_df)

    # === 3. PCA適用（次元削減） ===
    pca = PCA(n_components=5)  # まずは5次元に圧縮
    X_pca = pca.fit_transform(X_scaled)

    # === 4. KMeansクラスタリング ===
    n_clusters = 6  # 仮のクラスタ数（後で調整可）
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_pca)

    # === 5. 結果をデータフレームに追加 ===
    df_clustered = df.copy()
    df_clustered['cluster'] = cluster_labels

    return df_clustered

# returns bodypart and action in src_data
# src_dataに記述されているbodypartとactionを返す
def find_bodypart_action(src_data):
    bodyparts = []
    actions   = []
    for i in range(len(src_data)):
        train = src_data.iloc[i]
        body_parts_tracked = json.loads(train['body_parts_tracked'].replace('""', '"'))
        for j in range(len(body_parts_tracked)):
            body_parts = body_parts_tracked[j]
            bodyparts.append(body_parts)
        if pd.isna(train['behaviors_labeled']) == False:
            behaviors_labeled  = json.loads(train['behaviors_labeled'].replace('""', '"'))
            for i in range(len(behaviors_labeled)):
                act = behaviors_labeled[i].split(',')
                act[2] = act[2].replace("'", "")
                actions.append(act[2])
    return np.unique(bodyparts), np.unique(actions)

# === 5. ラベル付け関数 ===
def assign_action_label(row, annotations):
    agent_id = row['mouse_id']
    time = row['video_time']
#    print('frame:', frame)
#    mask = (annotations['video_id'] == video_id) & \
    mask = \
           (annotations['agent_id'] == agent_id) & \
           (annotations['start_time'] <= time) & \
           (annotations['stop_time'] >= time)
    actions = annotations.loc[mask, 'action'].unique()
    if len(actions) > 0:
        return actions[0]
    else:
        return np.nan

# === 4. tracking frame → annotation frame に変換関数 ===
def map_frame_to_annotation(row):
    global fps_dict
    video_id = row['video_id']
    frame = row['video_frame']
    fps = fps_dict.get(video_id, 30)  # デフォルト30fps
    # annotation のフレームは整数なので、tracking のフレームを丸める
    return int(round(frame))

# Annotationファイルを読み込む
def test_114(train_data, df_features):
    df_annotation = pd.DataFrame()

#    train_data  = pd.read_csv(INPUT_TRAIN_FILE)
    test_data   = pd.read_csv(INPUT_TEST_FILE)

    train_body_parts,train_actions = find_bodypart_action(train_data)
    test_body_parts,test_actions   = find_bodypart_action(test_data)
    test_tracking_methods = test_data['tracking_method'].unique()

    # trainにあってtestにないbodypartとactionを列挙する
    unnecessary_body_parts = list(set(train_body_parts) - set(test_body_parts))
    unnecessary_actions    = list(set(train_actions)    - set(test_actions))

    for i in range(len(train_data)):
    #    print(i,'/',len(train_data))
        train = train_data.iloc[i]
        lab_id = train['lab_id']
        # testと異なるtracking_methodは採用しない
        if not train['tracking_method'] in test_tracking_methods:
    #        print("boo #1")
            continue
        video_id = str(train['video_id'])
        fps  = train['frames_per_second']
        ppcm = train['pix_per_cm_approx']
        file_name = lab_id + "/" + video_id + ".parquet"
        annotation_file_path = os.path.join(TRAIN_ANNOTATION_DIR, file_name)
        # annotation fileがないので、次のannotation fileに移る
        if os.path.exists(annotation_file_path) == False:
#            print("boo #2")
            continue
        annotation = pd.read_parquet(annotation_file_path)
        # annotationにtestに出ないactionがあれば削除する
        for action in unnecessary_actions:
            annotation = annotation[annotation['action']!=action]
        if len(annotation) == 0:
#            print("boo #3")
            continue
        annotation['video_id'] = video_id
        annotation['start_time'] = annotation['start_frame'] / fps
        annotation['stop_time'] = annotation['stop_frame'] / fps
        df_annotation = pd.concat([df_annotation,annotation])

    # === 2. 型統一 ===
    df_features['mouse_id'] = df_features['mouse_id'].astype(int)
    df_annotation['agent_id'] = df_annotation['agent_id'].astype(int)
    df_annotation['target_id'] = df_annotation['target_id'].astype(int)

    # === 3. video_id ごとの fps 辞書作成 ===
    global fps_dict
    fps_dict = train_data.set_index('video_id')['frames_per_second'].to_dict()

#    df_features['frame_anno'] = df_features.apply(map_frame_to_annotation, axis=1)

    df_features['action'] = df_features.apply(lambda r: assign_action_label(r, df_annotation), axis=1)

    # === 6. NaN を削除 ===
    df_features = df_features.dropna(subset=['action'])

    # === 7. 学習用データ保存 ===
#    feature_cols = [c for c in df_features.columns if c not in ['action', 'video_id', 'mouse_id', 'bodypart', 'frame_anno']]
    feature_cols = [c for c in df_features.columns if c not in ['action', 'video_id', 'mouse_id', 'bodypart']]
    X = df_features[feature_cols]
    y = df_features['action']
    return X, y

def test_116(features, labels):
    features['action'] = labels['action']
    return features

def test_118(df):
    # === 2. 前処理 ===
    # 欠損値補完（平均値で）
    df = df.fillna(df.mean(numeric_only=True))

    # action列を目的変数に
    if "action" not in df.columns:
        raise ValueError("Error: 'action'列が見つかりません。ラベル付け済みデータを確認してください。")

    if "Unnamed: 0" in df.columns:
        df = df.drop("Unnamed: 0", axis=1)

#    for col in COLUMNS_TO_BE_REMOVED:
#        df = df.drop(col,axis=1)

    X = df.drop(columns=["action"])
    y = df["action"]

    # actionがカテゴリ文字列の場合はエンコード
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # === 3. 学習・テスト分割 ===
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # === 4. XGBoost モデル構築 ===
    model = xgb.XGBClassifier(
        objective='multi:softmax',  # クラス分類
        num_class=len(np.unique(y_encoded)),
        eval_metric='mlogloss',
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    # === 5. 学習 ===
    model.fit(X_train, y_train)

    # === 6. 予測と評価 ===
    y_pred = model.predict(X_test)
    # ↑でここで時間が相当かかっている(9〜10時間くらい)

    print("\n=== Accuracy ===")
    print(f"{accuracy_score(y_test, y_pred):.4f}")

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # === 8. 特徴量重要度 ===
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]

    df_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    df_importance.to_csv('importance-2.csv')
    return df_importance

# XGBoostハイパーパラメータ空間の定義
xgb_param_space={
    'n_estimators': Integer(100, 1000), #森の中の木の数
    'learning_rate': Real(0.001, 0.5),  #学習率
    'max_depth': Integer(1, 15),        #各決定木の最大深さ
    'min_child_weight': Integer(1,20),  #子ノードに必要な最小サンプル重みの合
    'gamma':    Real(0.0, 1.0),         #ノードの分割に必要な最小損失減少
    'subsample': Real(0.5, 1.0),        #各決定木を構築するために使用されるサンプルの割合
    'colsample_bytree': Real(0.0001,1.0),   #各決定木を構築するために使用される特徴量の割合
    'colsample_bylevel': Real(0.0001,1.0),   #各レベルの決定木を構築するために使用される特徴量の割合
    'colsample_bynode': Real(0.0001, 1.0), #各ノードの分割に使用される特徴量の割合
    'reg_alpha': Real(0.01, 1.0),         #L1正則化項の重み
    'reg_lambda': Real(0.01, 1.0),           #L2正則化項の重み
#    'scale_pos_weight': Real(0.1, 4.0),      #正例と負例の不均衡を補正するための重み
#    'base_score': Real(0.1, 1.0),           #すべての観測値に対する初期予測確率
}

def train_predict(train, train_y, test):
#    trainY = train.pop('action')
    xgb_param_space = {
        'max_depth': (3, 8),
        'learning_rate': (0.05, 0.3, 'log-uniform'),
        'n_estimators': (100, 500),
        'subsample': (0.6, 1.0),
        'colsample_bytree': (0.6, 1.0),
        'gamma': (0, 1.0)
    }
    model = xgb.XGBClassifier()
    opt = BayesSearchCV(
        estimator=model,
        search_spaces=xgb_param_space,
        n_iter=50,
        cv=3,
        n_jobs=-1,
        scoring='balanced_accuracy',  # ← ここを変更
        random_state=42
    )
    opt.fit(train, train_y)
    predY = opt.best_estimator_.predict(test)
    return (predY, opt)

def create_submission(df, predY, opt):
    # ここでは、video_time ≒ video_idとして扱う（実際の列名に合わせて修正可）
#    df["video_id"] = df["video_time"]
    if not "video_id" in df.columns:
        print("[create_submission]: not include video_id in df")

    print('len(df):', len(df))
    print('len(predY):', len(predY))

    # === 2. 各フレームで最も近いマウスをtarget_idに設定 ===
    targets = []

    for vid, vdf in tqdm(df.groupby("video_id"), desc="Processing videos"):
        # 同じビデオ内のマウス群で処理
        mice = vdf["mouse_id"].unique()
        for mid in mice:
            mdf = vdf[vdf["mouse_id"] == mid]
            for idx, row in mdf.iterrows():
                frame = row["video_frame"]
                # 同じフレームに存在する他のマウス
                others = vdf[(vdf["video_frame"] == frame) & (vdf["mouse_id"] != mid)]
                if len(others) == 0:
                    target_id = np.nan
                else:
                    # body_center間距離で最も近い相手をtargetに
                    dx = others["x_cm_body_center"] - row["x_cm_body_center"]
                    dy = others["y_cm_body_center"] - row["y_cm_body_center"]
                    dist = np.sqrt(dx**2 + dy**2)
                    target_id = others.iloc[dist.argmin()]["mouse_id"]
                targets.append(target_id)

#    df["target_id"] = targets
    print('len(df):',      len(df))
    print('len(targets):', len(targets))

    # === 3. モデルでactionを予測 ===
    # すでに opt.best_estimator_ が学習済み想定
#    predY = opt.best_estimator_.predict(df.drop(columns=["mouse_id", "video_frame", "video_id", "target_id"]))

    # もしlabel encoderを使用していた場合
    # predY = le.inverse_transform(predY)

    df["action"] = predY

    # === 4. 連続区間をまとめて submission形式に変換 ===
    results = []
    for vid, vdf in df.groupby("video_id"):
        for mid, mdf in vdf.groupby("mouse_id"):
            mdf = mdf.sort_values("video_frame")
            prev_action = None
            start_frame = None
            prev_target = None

            for idx, row in mdf.iterrows():
                current_action = row["action"]
                if 'target_id' in row:
                    current_target = row["target_id"]
                else:
                    current_target = np.nan

                if prev_action is None:
                    prev_action = current_action
                    prev_target = current_target
                    start_frame = row["video_frame"]

                elif (current_action != prev_action) or (current_target != prev_target):
                    results.append({
                        "video_id": vid,
                        "agent_id": mid,
                        "target_id": prev_target,
                        "action": prev_action,
                        "start_frame": start_frame,
                        "stop_frame": row["video_frame"] - 1
                    })
                    prev_action = current_action
                    prev_target = current_target
                    start_frame = row["video_frame"]

            # 最後の区間
            if prev_action is not None:
                results.append({
                    "video_id": vid,
                    "agent_id": mid,
                    "target_id": prev_target,
                    "action": prev_action,
                    "start_frame": start_frame,
                    "stop_frame": mdf["video_frame"].iloc[-1]
                })

    # === 5. submission.csv 出力 ===
    submission = pd.DataFrame(results)
    submission.insert(0, "row_id", range(len(submission)))

    return submission

# === メイン ===
if __name__ == "__main__":
    pd.set_option('display.max_info_columns', 300)
    start_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    test_data = pd.read_csv(INPUT_TEST_FILE)
    test_tracking_data = make_tracking_feature(test_data, False)
#    test_tracking_data.to_csv('test_tracking_data.csv')

    test_tracking_data_org = test_tracking_data.copy()

    print(start_time,': start time')
    train_data = create_valid_train()   # test_102
#    train_data.to_csv('train_data.csv')
    tracking_feature = make_tracking_feature(train_data, True)    # test_1031

    test_cols = test_tracking_data.columns
    train_cols = tracking_feature.columns

    # train(tracking_feature)とtest(test_tracking_data)の特徴量を合わせる
    for col in test_cols:
        if not col in train_cols:
            # testにあってtrainにない場合
            test_tracking_data = test_tracking_data.drop(col, axis=1)
    for col in train_cols:
        if not col in test_cols:
            # trainにあってtestにない場合
            tracking_feature = tracking_feature.drop(col,axis=1)
    print('len(tracking_feature.columns):',len(tracking_feature.columns))
    print('len(test_tracking_data.columns):',len(test_tracking_data.columns))
    print('tracking_feature.columns:')
    print(tracking_feature.columns)
    print('test_tracking_data')
    print(test_tracking_data.columns)

    tracking_features_slim = create_feature_category(tracking_feature)  # test_107.py
    tracking_features_reduced = create_tracking_features_reduced(tracking_feature,tracking_features_slim) #test-108.py
#    print('tracking_features_reduced:')
#    tracking_features_reduced.info()
    tracking_features_clustered = create_tracking_features_clustered(tracking_features_reduced) #test-111
    train_features, train_label = test_114(train_data, tracking_features_clustered)
#    train_features.info()
#    print('len(train_features):',len(train_features))
#    print('len(train_label):',len(train_label))
    train_features['action'] = train_label
#    train_features.info()
    importance_df = test_118(train_features)
    TOP_N = 30  #
    top_features = importance_df.head(TOP_N)["feature"].tolist()
    df_top = train_features[top_features + ["action"]]
    df_top.to_csv('df_top.csv') # video_idがない

    print('len(tracking_features_slim):', len(tracking_features_slim))
    print('len(tracking_features_reduced)', len(tracking_features_reduced))
    print('len(tracking_features_clustered)', len(tracking_features_clustered))
    print('len(train_features)', len(train_features))

    if not 'video_id' in tracking_features_slim:
        print('not include in tracking_features_slim')
    if not 'video_id' in tracking_features_reduced:
        print('not include in tracking_features_reduced')
    if not 'video_id' in tracking_features_clustered:
        print('not include in tracking_features_clustered')
    if not 'video_id' in train_features:
        print('not include in train_features')

    print('len(train_features):',len(train_features))
    print('len(df_top):', len(df_top))

    test_video_id = test_tracking_data['video_id']  #9828行

    test_df = pd.DataFrame()
    for col in df_top.columns:
        if col in test_tracking_data:
#            test_df[col] = df_top[col]
            test_df[col] = test_tracking_data[col]
    test_df.to_csv('test_tracking.csv')
    print('test_dfs columns', len(test_df.columns))
    print('df_tops columns', len(df_top.columns))

    le = LabelEncoder()
    train_y = df_top.pop('action')
    train_y_le = le.fit_transform(train_y)
    pred,opt= train_predict(df_top,train_y_le,test_df)
    pred_df_le = le.inverse_transform(pred)
    pred_df = pd.DataFrame(pred_df_le)
#    pred_df.to_csv('pred.csv')

    if not 'mouse_id' in test_df.columns:
        test_df['mouse_id'] = test_tracking_data['mouse_id']
    if 'video_frame' in test_tracking_data.columns:
        if not 'video_frame' in test_df.columns:
            test_df['video_frame'] = test_tracking_data['video_frame']
    if not 'x_cm_body_center' in test_df:
        test_df['x_cm_body_center'] = test_tracking_data_org['x_cm_body_center']
    if not 'y_cm_body_center' in test_df:
        test_df['y_cm_body_center'] = test_tracking_data_org['y_cm_body_center']

    print(test_df.columns)
    test_df.to_csv('test_df.csv')
    print('len(test_df):', len(test_df))                # video_idがない
    print('len(test_video_id):',len(test_video_id))
    test_df['video_id'] = test_video_id

    submission = create_submission(test_df, pred_df_le, opt)

    print('len(submission):', len(submission))
    print('len(test_video_id):',len(test_video_id))

    submission = submission.drop_duplicates(subset=['start_frame','stop_frame','action','video_id','agent_id','target_id'])
#    submission = submission.drop('video_id', axis=1)
#    submission.info()
    # start_frameとend_frameの整数化
#    submission = submission.astype("Int64")
    submission['agent_id']    = submission['agent_id'].astype("Int64")
    submission['target_id']   = submission['target_id'].astype("Int64")
    submission['start_frame'] = submission['start_frame'].astype("Int64")
    submission['stop_frame']  = submission['stop_frame'].astype("Int64")
##    print(submission['action'].iloc[:10], submission['action'].dtype)
#    submission['action']      = submission['action'].astype("Int64")
#    submission = submission.dropna(subset=["action"])
#    submission.info()
#    submission['action']      = le.inverse_transform(submission['action'])
    submission['target_id'] = submission['target_id'].fillna(submission['agent_id'])

    submission.to_csv(OUTPUT_FILE, index=False)

    end_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(end_time, ': complete!')
