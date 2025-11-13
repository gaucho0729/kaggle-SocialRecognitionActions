import pandas as pd
from pathlib import Path

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

print(len(parquet_files))


