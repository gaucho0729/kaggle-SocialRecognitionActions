import pandas as pd
import numpy as np

INPUT_FILE_NAME = 'submission-0.csv'
OUTPUT_FILE_NAME = 'submission-2.csv'

# todo: solo_actionを自動的に生成するようにする (2025-10-14)
solo_action = [
    'rear'
]

pair_actions = [
    'approach', 'attack', 'avoid', 'chase', 'chaseattack', 'submit'
]



def build_submission_2(pred_df):
    retval = pd.DataFrame()
    tmp = pred_df.copy()
    tmp['used'] = [False] * len(pred_df)
    video_ids = pred_df['video_id'].unique()
    mice = pred_df['mouse_id'].unique()
#    mice = pred_df['agent_id'].unique()
    for video_id in video_ids:
        print('video_id:', video_id)
        for mouse in mice:
            tmp_df = pred_df[(pred_df['mouse_id']==mouse) & (pred_df['video_id']==video_id)]
#            tmp_df = pred_df[(pred_df['agent_id']==mouse) & (pred_df['video_id']==video_id)]
#            frms = tmp_df['video_frame'].unique()
            actions = tmp_df['pred_action'].unique()
#            actions = tmp_df['action'].unique()
            for action in actions:
                tmp_act = tmp_df[tmp_df['pred_action']==action]
#                tmp_act = tmp_df[tmp_df['action']==action]
                if len(tmp_act)==0:
                    continue
                prev_frame = 0
                start_frame = tmp_act['video_frame'].iloc[0]
                frms = tmp_act['video_frame'].unique()
                for frm in frms:
                    tmp = tmp_act[tmp_act['video_frame']==frm]
                    if tmp['video_frame'].iloc[0] - prev_frame >= 60:
                        if tmp['agent_target'].iloc[0] == True:
                            agent = tmp['mouse_id'].iloc[0]
#                            agent = tmp['agent_id'].iloc[0]
                            target = tmp['target_id'].iloc[0]
                            if prev_frame != 0:
                                row = pd.DataFrame({
                                        'video_id'   : [video_id]    * 1,
                                        'agent_id'   : [agent]       * 1,
                                        'target_id'  : [target]      * 1,
                                        'action'     : [action]      * 1,
                                        'start_frame': [start_frame] * 1,
                                        'stop_frame' : [prev_frame]  * 1
                                })
                                retval = pd.concat([retval, row], ignore_index=True)
#                        else:
#                            target = tmp['mouse_id'].iloc[0]
#                            agent  = tmp['target_id'].iloc[0]
                        start_frame = frm
                    prev_frame = tmp['video_frame'].iloc[0]
    return retval

def remove_self_actions(df, actions_to_remove):
    """
    自分自身への行動を削除する。

    Parameters
    ----------
    df : pd.DataFrame
        agent_id, target_id, action を含む DataFrame
    actions_to_remove : list of str
        削除対象の行動名リスト（例：["attack", "chase"]）
    """
    cond = (df["agent_id"] == df["target_id"]) & (df["action"].isin(actions_to_remove))
    removed_count = cond.sum()
    df = df[~cond].reset_index(drop=True)
    print(f"削除行数: {removed_count}")
    return df

def remove_pair_actions(df, actions_to_remove):
    """
    自分自身への行動を削除する。

    Parameters
    ----------
    df : pd.DataFrame
        agent_id, target_id, action を含む DataFrame
    actions_to_remove : list of str
        削除対象の行動名リスト（例：["attack", "chase"]）
    """
    cond = (df["agent_id"] != df["target_id"]) & (df["action"].isin(actions_to_remove))
    removed_count = cond.sum()
    df = df[~cond].reset_index(drop=True)
    print(f"削除行数: {removed_count}")
    return df

def remove_same_start_stop_frame(df):
    cond = (df['start_frame'] == df['stop_frame'])
    removed_count = cond.sum()
    df = df[~cond].reset_index(drop=True)
    print(f"削除行数: {removed_count}")
    return df

def canonical_pair(row):
    """ペアをソートして正規化"""
    a, t = sorted([row['agent_id'], row['target_id']])
    return pd.Series({'p_agent': a, 'p_target': t})

# === 攻撃側を agent に統一 ===
def unify_attack_direction(sub):
    # 攻撃がない場合は小さい方をagent
    if not (sub['action'] == 'attack').any():
        a, t = sorted([sub['p_agent'].iloc[0], sub['p_target'].iloc[0]])
        return pd.Series({'agent_id': a, 'target_id': t})

    # 攻撃がある場合 → その行のscoreを使って判定
    attacks = sub[sub['action'] == 'attack']
    if len(attacks) == 1:
        row = attacks.iloc[0]
        return pd.Series({'agent_id': row['agent_id'], 'target_id': row['target_id']})
    else:
        # 複数攻撃あり → scoreが高い方をagent
        top = attacks.loc[attacks['score'].idxmax()]
        return pd.Series({'agent_id': top['agent_id'], 'target_id': top['target_id']})


def clear_contradiction(df):
    # スコアがなければ仮で 1.0
    if "score" not in df.columns:
        df["score"] = 1.0

    # 行動優先順位（上ほど優先度高い）
    priority = ['chase', 'approach', 'attack', 'avoid', 'rear', 'submit']
    priority_map = {a: i for i, a in enumerate(priority)}

    # === 1. agent_id と target_id の逆転重複を整理 ===
    df[['p_agent', 'p_target']] = df.apply(canonical_pair, axis=1)

    # === 2. 同一ペア・重複フレームを統合 ===
    merged = []
    for (vid, a, t), sub in df.groupby(['video_id', 'p_agent', 'p_target']):
        sub = sub.sort_values(['start_frame', 'stop_frame']).reset_index(drop=True)

        # 重複区間を統合（行動優先度を考慮）
        merged_rows = []
        current = sub.iloc[0].copy()
        for _, row in sub.iloc[1:].iterrows():
            overlap = not (row['start_frame'] > current['stop_frame'])
            if overlap:
                # 行動が違う場合：スコア or 優先度で選択
                if row['score'] > current['score']:
                    current['action'] = row['action']
                    current['score'] = row['score']
                elif row['score'] == current['score']:
                    if priority_map[row['action']] < priority_map[current['action']]:
                        current['action'] = row['action']
                # フレーム結合
                current['stop_frame'] = max(current['stop_frame'], row['stop_frame'])
            else:
                merged_rows.append(current)
                current = row.copy()
        merged_rows.append(current)

    clean_df = pd.DataFrame(merged_rows)
    merged_df = pd.DataFrame(merged_rows)

    # 各ペアごとにagent-target方向を決定
    agent_target_df = merged_df.groupby(['video_id', 'p_agent', 'p_target']).apply(unify_attack_direction).reset_index()

    # === 4. 結合して方向統一 ===
    final_df = merged_df.merge(agent_target_df, on=['video_id', 'p_agent', 'p_target'], how='left')

    final_df.info()
    for col in ['agent_id', 'target_id']:
        if f'{col}_y' in final_df.columns:
            final_df[col] = final_df[f'{col}_y']
        elif f'{col}_x' in final_df.columns:
            final_df[col] = final_df[f'{col}_x']

    # === 5. 統一方向に反転 ===
    # （p_agent/p_targetと決定されたagent/targetが逆ならswap）
    swap_mask = (final_df['p_agent'] == final_df['target_id']) & (final_df['p_target'] == final_df['agent_id'])
    final_df.loc[swap_mask, ['agent_id', 'target_id']] = final_df.loc[swap_mask, ['target_id', 'agent_id']].values

    # === 6. 出力整形 ===
    final_df = final_df[['video_id', 'agent_id', 'target_id', 'action', 'start_frame', 'stop_frame', 'score']]
    final_df = final_df.sort_values(['video_id', 'start_frame']).reset_index(drop=True)
    final_df.to_csv("submission_cleaned_scored.csv", index=False)

    final_df.index.name = 'row_index'
    print("✅ 出力:", final_df.shape)
    print(final_df.head(10))

    return final_df


if __name__ == "__main__":
    all_data = pd.read_csv(INPUT_FILE_NAME)
    all_data.info()
    df = build_submission_2(all_data)
    df['agent_id']  = df['agent_id'].astype('Int64')
    df['target_id'] = df['target_id'].astype('Int64')

    df.to_csv('df.csv')

    df = df.sort_values(by=['start_frame'],ascending=[True])
    df = df.dropna(subset=['target_id'])
    df = df.dropna(subset=['agent_id'])
    df = remove_self_actions(df, pair_actions)
    df = remove_pair_actions(df, solo_action)
    df = remove_same_start_stop_frame(df)
    
    df.to_csv('submission-1.csv')

#    df = merge_overlapping_actions(df)
#    df = resolve_reversed_actions(df)
#    df = remove_conflicting_actions(df)
#    df = df.sort_values(by='start_frame')

    df = clear_contradiction(df)

#    all_data['action'] = all_data['pred_action']
#    all_data.info()
#    df = shape_data_for_submit(all_data,all_data['action'].unique(),all_data['mouse_id'].unique())
    df.to_csv(OUTPUT_FILE_NAME)
