import pandas as pd
from pathlib import Path

OUTPUT_FILE = "analyze-tracking.csv"

train_dirs = [
    "../train_tracking/AdaptableSnail",
    "../train_tracking/BoisterousParrot",
    "../train_tracking/CalMS21_supplemental",
    "../train_tracking/CalMS21_task1",
    "../train_tracking/CalMS21_task2",
    "../train_tracking/CautiousGiraffe",
    "../train_tracking/CRIM13",
    "../train_tracking/DeliriousFly",
    "../train_tracking/ElegantMink",
    "../train_tracking/GroovyShrew",
    "../train_tracking/InvincibleJellyfish",
    "../train_tracking/JovialSwallow",
    "../train_tracking/LyricalHare",
    "../train_tracking/MABe22_keypoints",
    "../train_tracking/MABe22_movies",
    "../train_tracking/NiftyGoldfinch",
    "../train_tracking/PleasantMeerkat",
    "../train_tracking/ReflectiveManatee",
    "../train_tracking/SparklingTapir",
    "../train_tracking/TranquilPanther",
    "../train_tracking/UppityFerret",
]

parquet_files = list()

for dir in train_dirs:
    # Parquetファイルがあるルートディレクトリ
    root_dir = Path(dir)  # 適宜変更

    # 全サブディレクトリを含めてParquetを検索
    parquet_files += list(root_dir.rglob("*.parquet"))

for file in parquet_files:
    all_data = pd.read_parquet(file)
    numeric_cols = all_data.select_dtypes(include='number')
    stats = numeric_cols.describe().T
    stats['median'] = numeric_cols.median()
    stats = stats[['min', 'max', 'mean', 'median', 'std']]
    with open(OUTPUT_FILE, mode='a') as f:
        f.write(str(file) + "\n")
    stats.to_csv(OUTPUT_FILE, mode='a')
    obj_cols = all_data.select_dtypes(include='object')
    for col in obj_cols.columns:
        unique_values = obj_cols[col].unique()  # 列のユニーク値を取得
        with open(OUTPUT_FILE, mode='a') as f:
            f.write(f"{col}: {unique_values}\n")
    with open(OUTPUT_FILE, mode='a') as f:
        f.write("\n\n")
