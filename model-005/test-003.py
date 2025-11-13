import pandas as pd
import numpy as np

# 部位間距離の相関係数を取る

if __name__ == "__main__":
    train_features = pd.read_csv("tracking_features.csv")
    t = train_features.copy()
    t = t.drop('video_id', axis=1)
    t = t.drop('video_frame', axis=1)
    t_corr = t.corr()
    t_corr.to_csv('corr.csv')
