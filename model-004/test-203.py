import pandas as pd
import numpy as np

def remove_passive_action(df):
    df_reduced = df[df['action'] != "none"]
    actions = df_reduced['action'].unique()
    reduced_actions = [w for w in actions if "-ed" in w]
    for act in reduced_actions:
        df_reduced = df_reduced[df_reduced['action'] != act]
    return df_reduced

def shrink_actions(df):
    def add_output(prev, frame):
        """prev_infoの1行分から出力を作成してoutputに追加"""
        return pd.DataFrame({
            'agent_id': [prev['agent_id']],
            'target_id': [prev['target_id']],
            'action': [prev['action']],
            'start_frame': [prev['frame']],
            'stop_frame': [frame - 1]
        })

    def update_prev(prev_info, idx, agent, target, action, frame):
        """prev_infoを更新"""
        prev_info.iloc[idx] = [action, agent, target, frame]

    def reset_prev(prev_info, idx):
        """prev_infoをリセット"""
        prev_info.iloc[idx] = ['', 0, 0, 0]

    output = []
    prev_info = pd.DataFrame({
        'action': ['', ''],
        'agent_id': [0, 0],
        'target_id': [0, 0],
        'frame': [0, 0]
    })

    for i, row in df.iterrows():
        agent, target, action, frame = row['agent_id'], row['target_id'], row['action'], row['frame']

        # 初回のみ登録
        if i == 0:
            update_prev(prev_info, 0, agent, target, action, frame)
            continue

        for idx in [0, 1]:
            p = prev_info.iloc[idx]

            # 1) 同一動作を継続中
            if (p['action'] == action) and (p['agent_id'] == agent):
                break

            # 2) agent/target 入れ替わり
            if (p['target_id'] == agent) and (p['agent_id'] == target):
                output.append(add_output(p, frame))
                update_prev(prev_info, idx, agent, target, action, frame)
                break

            # 3) action同じだがtargetが異なる
            if (p['action'] == action) and (p['agent_id'] == agent) and (p['target_id'] != target):
                output.append(add_output(p, frame))
                update_prev(prev_info, idx, agent, target, action, frame)
                break

            # 4) action同じだがagentが異なる
            if (p['action'] == action) and (p['agent_id'] != agent) and (p['target_id'] == target):
                output.append(add_output(p, frame))
                update_prev(prev_info, idx, agent, target, action, frame)
                break

            # 5) agent/target同じだがaction異なる
            if (p['action'] != action) and (p['agent_id'] == agent) and (p['target_id'] == target):
                output.append(add_output(p, frame))
                update_prev(prev_info, idx, agent, target, action, frame)
                break
        else:
            # 6) 新しい関係性
            p0, p1 = prev_info.iloc[0], prev_info.iloc[1]
            cond_agent_change = (p0['agent_id'] == agent) and (p0['target_id'] != target)
            cond_target_change = (p0['agent_id'] != agent) and (p0['target_id'] == target)

            if cond_agent_change or cond_target_change:
                # prev_info[1] が空なら1件だけ出力
                if p1['agent_id'] == 0:
                    output.append(add_output(p0, frame))
                    update_prev(prev_info, 0, agent, target, action, frame)
                else:
                    # 両方出力してprev_info[1]をリセット
                    output.append(add_output(p0, frame))
                    output.append(add_output(p1, frame))
                    update_prev(prev_info, 0, agent, target, action, frame)
                    reset_prev(prev_info, 1)

    # 出力をまとめて返す
    return pd.concat(output, ignore_index=True) if output else pd.DataFrame()

if __name__ == "__main__":
    submission_org = pd.read_csv("submission.csv")
    submission_reduced = remove_passive_action(submission_org)
    submission_tmp = shrink_actions(submission_reduced)
    submission_tmp.to_csv('submission_tmp.csv')