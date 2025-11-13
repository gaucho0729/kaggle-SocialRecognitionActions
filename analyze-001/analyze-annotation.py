import pandas as pd
from pathlib import Path

INPUT_FILE = "../train_annotation/AdaptableSnail/705948978.parquet"
OUTPUT_FILE = "analyze-annotation.csv"

train_dirs = [
    "../train_annotation/AdaptableSnail",
    "../train_annotation/BoisterousParrot",
    "../train_annotation/CalMS21_supplemental",
    "../train_annotation/CalMS21_task1",
    "../train_annotation/CalMS21_task2",
    "../train_annotation/CautiousGiraffe",
    "../train_annotation/CRIM13",
    "../train_annotation/DeliriousFly",
    "../train_annotation/ElegantMink",
    "../train_annotation/GroovyShrew",
    "../train_annotation/InvincibleJellyfish",
    "../train_annotation/JovialSwallow",
    "../train_annotation/LyricalHare",
    "../train_annotation/NiftyGoldfinch",
    "../train_annotation/PleasantMeerkat",
    "../train_annotation/ReflectiveManatee",
    "../train_annotation/SparklingTapir",
    "../train_annotation/TranquilPanther",
    "../train_annotation/UppityFerret",
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
