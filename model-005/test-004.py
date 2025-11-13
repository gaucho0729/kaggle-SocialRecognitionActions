import pandas as pd
import numpy as np

# test-002.pyで生成した特徴量にactionに関する特徴量を追加する

def get_action_label(features, annos):
    video_frames = features['video_frame'].unique()
    action_list_1 = []
    action_list_2 = []
    agent_list_1  = []
    agent_list_2  = []
    target_list_1  = []
    target_list_2  = []
    for (video_frame), groups in features.groupby(['video_frame']):
        action_anno = annos[(annos['start_frame']<=video_frame) & (annos['stop_frame']>=video_frame)]
        if len(action_anno) == 0:
            action_list_1.append('none')
            action_list_2.append('none')
            agent_list_1.append(np.nan)
            target_list_1.append(np.nan)
            agent_list_2.append(np.nan)
            target_list_2.append(np.nan)
            continue
        tmp = action_anno.iloc[0]
        action1 = tmp['action']
        action_list_1.append(action1)
        agent_list_1.append(tmp['agent_id'])
        target_list_1.append(tmp['target_id'])
        if len(action_anno) >= 2:
            tmp = action_anno.iloc[1]
            action2 = tmp['action']
            action_list_2.append(action2)
            agent_list_2.append(tmp['agent_id'])
            target_list_2.append(tmp['target_id'])
        else:
            action_list_2.append('none')
            agent_list_2.append(np.nan)
            target_list_2.append(np.nan)
    return action_list_1, action_list_2,agent_list_1,agent_list_2,target_list_1,target_list_2

if __name__ == "__main__":
    train_features = pd.read_csv("tracking_features.csv")
    annotations = pd.read_csv("annotation-44566106.csv")
    action_list_1, action_list_2,agent_list_1,agent_list_2,target_list_1,target_list_2 = get_action_label(train_features, annotations)
    train_features['action1'] = action_list_1
    train_features['agent_1'] = agent_list_1
    train_features['target_1'] = target_list_1
    train_features['action2'] = action_list_2
    train_features['agent_2'] = agent_list_2
    train_features['target_2'] = target_list_2

    train_features.to_csv("tracking_features-2.csv")