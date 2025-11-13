import pandas as pd
import numpy as np

INPUT_FILE = "../train.csv"
OUTPUT_FILE = "object-feature.csv"

np.set_printoptions(threshold=np.inf)

# カテゴリ変数
all_data = pd.read_csv(INPUT_FILE)
#for col in all_data.columns:
#    with open(OUTPUT_FILE, mode='a') as f:
#        unique_values = all_data[col].unique()  # 列のユニーク値を取得
#        f.write(f"{col}: {unique_values}\n")


print(all_data["mouse1_condition"].unique())
print(all_data["mouse2_condition"].unique())
print(all_data["mouse3_condition"].unique())
print(all_data["mouse4_condition"].unique())
