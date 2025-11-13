import pandas as pd
import numpy as np
import ast
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import glob, os

# analyze_action_tracking.pyの並列化(未完成)

INPUT_TRAIN_FILE       = "../train.csv"
TRAIN_ANNOTATION_DIR   = "../train_annotation/"
TRAIN_TRACKING_DIR     = "../train_tracking/"

INPUT_TEST_FILE        = "../test.csv"
TEST_TRACKING_DIR      = "../test_tracking/"

OUTPUT_FILE = "valid_action_tracking.csv"

INPUT_DIR = "input_csv"
FINAL_OUTPUT = "all_results.csv"

# === ファイルごとの分析処理 ===
def analyze_file(filepath):
    chunksize = 100000  # チャンクサイズ（10万行ごとに処理）
    results = []
    for chunk in pd.read_csv(filepath, chunksize=chunksize):
        # ▼ ここで好きな分析処理を行う ▼
        # 例: 行数をそのまま返す（実際はフィルタ・特徴量計算など）
        chunk["source_file"] = os.path.basename(filepath)
        results.append(chunk)
    return results

# === 読み込むファイルリストを作成する
def makeFileList():
    retval = pd.DataFrame(
        'tracking'  : [],
        'annotation': [],
    )
    train = pd.read_csv(INPUT_TRAIN_FILE)
    for j in range(len(train)):
        trn = train.iloc[j]
        lab_id   = str(trn['lab_id'])
        video_id = str(trn['video_id'])
        if os.path.isdir(TRAIN_TRACKING_DIR + lab_id)==False:
            continue
        if os.path.exists(TRAIN_TRACKING_DIR + lab_id + "/" + video_id + ".parquet") == False:
            continue
        track_file = TRAIN_TRACKING_DIR + lab_id + "/" + video_id + ".parquet"
        if os.path.isdir(TRAIN_ANNOTATION_DIR + lab_id)==False:
            continue
        if os.path.exists(TRAIN_ANNOTATION_DIR + lab_id + "/" + video_id + ".parquet") == False:
            continue
        annon_file = TRAIN_ANNOTATION_DIR + lab_id + "/" + video_id + ".parquet"
        df = pd.DataFrame(
            'tracking'  : [track_file],
            'annotation': [annon_file],
        )
        retval = pd.concat([retval, df], ignore_index=True)
    return retval

# === 並列実行 & 書き込み ===
def main():
    files = makeFileList()

    first = True
    with ProcessPoolExecutor(max_workers=4) as executor, open(FINAL_OUTPUT, "w", newline="", encoding="utf-8") as f:
        futures = {executor.submit(analyze_file, file): file for file in files}
        for future in as_completed(futures):
            try:
                results = future.result()
                for df in results:  # チャンクごとに書き込む
                    df.to_csv(f, header=first, index=False, mode="a")
                    first = False
                print(f"完了: {futures[future]}")
            except Exception as e:
                print(f"エラー: {futures[future]}, {e}")

    print(f"\n✅ 全部まとめたCSVを出力しました: {FINAL_OUTPUT}")

if __name__ == "__main__":
    main()
