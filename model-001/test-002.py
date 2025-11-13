import pandas as pd
import numpy as np
from io import StringIO

INPUT_FILE = 'test_tracking_action_agent_target.csv'

# {'approach': 0, 'attack': 1, 'avoid': 2, 'chase': 3, 'chaseattack': 4, 'rear': 5, 'submit': 6}
action_remap = {0: 'approach', 1: 'attack', 2: 'avoid', 3: 'chase', 4: 'chaseattack', 5: 'rear', 6: 'submit'}


solo_action = [
    'rear'
]

all_data = pd.read_csv(INPUT_FILE)
mice = all_data['mouse_id'].unique()
actions = all_data['action'].unique()
bools   = all_data['agent_target'].unique()
frames  = all_data['video_frame'].unique()

all_data.info()

# Kaggle提出用フォーマットにする検討用コード

tmp = pd.DataFrame({
    'video_id'   : [],
    'agent_id'   : [],
    'target_id'  : [],
    'action'     : [],
    'start_frame': [],
    'stop_frame' : [],
})

target_group = pd.DataFrame()

# agent mouse
for action in actions:
    for mouse in mice:
        row_data = all_data[(all_data['action']       == action) &
                            (all_data['mouse_id']     == mouse)  &
                            (all_data['agent_target'] == True)
                            ]
        if len(row_data) > 0:
            print('woo')
            row_df = pd.DataFrame()
            if len(row_data) >= 1:
                row_df = pd.DataFrame({
                    'video_id'   : row_data['video_id'].iloc[0],
                    'agent_id'   : row_data['mouse_id'].iloc[0],
                    'target_id'  : np.nan,
                    'action'     : [action] * 1,
                    'start_frame': row_data['video_frame'].iloc[0],
                    'stop_frame' : row_data['video_frame'].iloc[len(row_data)-1],
                })

            tmp = pd.concat([tmp, row_df], ignore_index=True)

        row_data = all_data[(all_data['action']       == action) &
                            (all_data['mouse_id']     == mouse)  &
                            (all_data['agent_target'] == False)
                            ]
        if len(row_data) == 0:
            print('foo')
            continue
        if len(row_data) >= 1:
            row_df = pd.DataFrame({
                'video_id'   : row_data['video_id'].iloc[0],
                'agent_id'   : np.nan,
                'target_id'  : row_data['mouse_id'].iloc[0],
                'action'     : [action] * 1,
                'start_frame': row_data['video_frame'].iloc[0],
                'stop_frame' : row_data['video_frame'].iloc[len(row_data)-1],
            })
        tmp = pd.concat([tmp, row_df], ignore_index=True)

tmp['action'] = tmp['action'].map(action_remap)
tmp.info()
# solo_actionのagent_idとtarget_idを一致させる
for i in range(len(tmp)):
    print(i, '/', len(tmp))
    row = tmp.iloc[i]
    row.info()
    for act in solo_action:
        if row['action'] == act:
            if np.isnan(row['agent_id']):
                tmp.loc[i, 'agent_id'] = row['target_id']
            if np.isnan(row['target_id']):
                tmp.loc[i, 'target_id'] = row['agent_id']

# start_frame順に並び替える
tmp = tmp.sort_values(by=['start_frame','action'])

#frm = tmp['start_frame'].unique()
#frm = frm.
#(tmp['stop_frame'].unique())


# 1. action + start_time ごとにグループ化
paired_rows = []
for (act, start), g in tmp.groupby(['action', 'start_frame']):
    # agent_id のみある行
    agents = g[g['agent_id'].notna() & g['target_id'].isna()].copy()
    # target_id のみある行
    targets = g[g['target_id'].notna() & g['agent_id'].isna()].copy()
    
    used_targets = set()
    for i, agent_row in agents.iterrows():
        # 未使用の target を探す
        available_targets = targets[~targets.index.isin(used_targets)]
        if available_targets.empty:
            continue
        target_row = available_targets.iloc[0]  # 最初のものを対応付け
        used_targets.add(target_row.name)

        # stop_frame は小さい方を採用
        stop_frame = min(agent_row['stop_frame'], target_row['stop_frame'])

        # ペアリング情報を統合
        new_row = agent_row.copy()
        new_row['target_id'] = target_row['target_id']
        new_row['stop_frame'] = stop_frame
        paired_rows.append(new_row)

# ペアリング結果を DataFrame に
paired_df = pd.DataFrame(paired_rows)

# 2. 既存データとマージして、ペアがあるものを更新
merged = pd.merge(tmp, paired_df, on=['action', 'start_frame', 'agent_id'], how='left', suffixes=('', '_new'))

# 既にtarget_idが空欄なら新しいものを採用
merged['target_id'] = merged['target_id'].combine_first(merged['target_id_new'])
merged['stop_frame'] = merged['stop_frame_new'].combine_first(merged['stop_frame'])

# 不要列を削除
merged = merged[tmp.columns]

print(merged)

adjusted_stop = []

for i, row in merged.iterrows():
    vid = row["video_id"]

    # 同じ video 内で agent_id と target_id の次の出現を探す
    next_agent = merged[(merged["video_id"] == vid) &
                    ((merged["agent_id"] == row["agent_id"]) |
                     (merged["agent_id"] == row["target_id"]) 
                    ) &
                    (merged["start_frame"] > row["start_frame"])]
    
    next_target = merged[(merged["video_id"] == vid) &
                     ((merged["target_id"] == row["target_id"]) |
                      (merged["target_id"] == row["agent_id"])
                      ) &
                     (merged["start_frame"] > row["start_frame"])]

    # 次の出現フレーム候補を集める
    next_frames = []
    if not next_agent.empty:
        next_frames.append(next_agent["start_frame"].min())
    if not next_target.empty:
        next_frames.append(next_target["start_frame"].min())

    # stop_frame調整判定
    if next_frames:
        next_min = min(next_frames)
        if row["stop_frame"] > next_min:
            stop_frame = next_min - 1
        else:
            stop_frame = row["stop_frame"]
    else:
        stop_frame = row["stop_frame"]

    adjusted_stop.append(stop_frame)

merged["stop_frame"] = adjusted_stop

merged.to_csv('tmp.csv')

print(merged)


# tmpを
