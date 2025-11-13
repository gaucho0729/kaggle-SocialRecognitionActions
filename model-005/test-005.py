import pandas as pd
import numpy as np

# action毎の部位間距離の相関係数を取る

if __name__ == "__main__":
    train_features = pd.read_csv("tracking_features-2.csv")
    actions = train_features['action1'].unique()
    for i in range(len(actions)):
        act = actions[i]
        train = train_features[train_features['action1']==act]
        train = train.drop(columns=['video_id','video_frame','action1','agent_1','target_1','action2','agent_2','target_2'])
        corr = train.corr()
        corr.to_csv(f'corr-{act}-1.csv')

    actions = train_features['action2'].unique()
    for i in range(len(actions)):
        act = actions[i]
        if act == 'none':
            continue
        train = train_features[train_features['action2']==act]
        train = train.drop(columns=['video_id','video_frame','action1','agent_1','target_1','action2','agent_2','target_2'])
        corr = train.corr()
        corr.to_csv(f'corr-{act}-2.csv')
