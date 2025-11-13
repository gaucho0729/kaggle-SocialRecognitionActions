import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype, is_string_dtype
from enum import Enum

# 訓練データの特徴を表示し、ファイルに保存する
## 列の表示、データ型、平均値、標準偏差、四分位、欠損値の数、外れ値の数

class OutlierType(Enum):
    quater = 1,     # 四分位
    sigma_2 = 2,    # 2sigma
    sigma_3 = 3    # 3 sigma

# 初期値
INPUT_FILE = '../train.csv'
OUTPUT_FILE = 'train-info.txt'

# 外れ値の数をカウントする
def get_qty_outlier(df, outlier_type):
    outlier_counts = {}

    for col in df.columns:
        if is_numeric_dtype(df[col]) == True:
            if outlier_type == OutlierType.quater:
                Q1 = df[col].quantile(0.25)  # 第1四分位
                Q3 = df[col].quantile(0.75)  # 第3四分位
                IQR = Q3 - Q1                # 四分位範囲
                
                # 外れ値の条件（1.5 * IQR を外れ値の基準にするのが一般的）
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outlier_counts[col] = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            elif outlier_type == OutlierType.sigma_2:
                lower = df[col].mean() - (2 * df[col].std())
                upper = df[col].mean() + (2 * df[col].std())
                outlier_counts[col] = ((df[col] < lower) | (df[col] > upper)).sum()
            elif outlier_type == OutlierType.sigma_3:
                lower = df[col].mean() - (3 * df[col].std())
                upper = df[col].mean() + (3 * df[col].std())
                outlier_counts[col] = ((df[col] < lower) | (df[col] > upper)).sum()

    return (pd.Series(outlier_counts, name="outlier_count"))

with open(OUTPUT_FILE, mode='w') as f:
    # 列の情報を表示する
    all_data = pd.read_csv(INPUT_FILE)
    print('[Column\'s information]:')
    print(all_data.dtypes)
    f.write('[Column\'s information]:\n')
    f.write(str(all_data.dtypes))

    # 欠損値を数を表示する
    print('[Numbers of None-data]:')
    print(all_data.isnull().sum())
    f.write('\n[Numbers of None-data]:\n')
    f.write(str(all_data.isnull().sum()))

    # 統計値を表示する
    print('[Stas information]:')
    print(str(all_data.describe()))
    f.write('\n[Stas information]:\n')
    f.write(str(all_data.describe()))

    # 外れ値の数を表示する
    print('[Outlier information]:')
    print('  out of quatile:')
    print(str(get_qty_outlier(all_data, OutlierType.quater)))
    f.write('\n[Outlier information]:\n')
    f.write('  out of quatile:\n')
    f.write(str(get_qty_outlier(all_data, OutlierType.quater)))

    print('  out of 2sigma:\n')
    print(get_qty_outlier(all_data, OutlierType.sigma_2))
    f.write('\n  out of 2sigma:\n')
    f.write(str(get_qty_outlier(all_data, OutlierType.sigma_2)))

    print('  out of 3sigma:')
    print(get_qty_outlier(all_data, OutlierType.sigma_3))
    f.write('\n  out of 3sigma:\n')
    f.write(str(get_qty_outlier(all_data, OutlierType.sigma_3)))

    f.close()

